// test_oscar_attn_decode.cpp — regression test for the OSCAR INT2-quantized
// attention decode kernel (kernels/oscar_quant.hip). This kernel had zero
// test coverage before this and shipped with two real bugs, both found by
// this test:
//
// 1. Online-softmax running state (m, l) lived in per-thread local
//    variables, updated only by a single thread (tid==0) out of every 128 —
//    every other thread read its own perpetually-unchanged -FLT_MAX/0,
//    corrupting the accumulator with inf/nan. The per-warp dot-product
//    score was also never reduced across the block's 4 warps, so for
//    HD > 32 each warp fed the softmax a partial, not the true, score.
// 2. Independent of #1: the accumulator was normalized by dividing by the
//    running softmax denominator `l` on every iteration instead of once at
//    the end (the standard flash-attention pattern) — a systematic ~15-20%
//    output bias that would exist even single-threaded, single-warp.
//
// Uses identity rotation matrices so the reference computation is exactly
// hand-verifiable: it replicates the same INT2 quantize/dequantize
// round-trip (including fp16 rounding) that the GPU path applies, isolating
// the kernel's attention math from the separate, expected, inherent
// lossiness of 2-bit quantization itself.

#include "rocm_cpp/oscar.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define HC(e) do { auto _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP err %d at %d\n", _s, __LINE__); exit(1); } } while (0)

int main() {
    int dev_count = 0;
    if (hipGetDeviceCount(&dev_count) != hipSuccess || dev_count == 0) {
        fprintf(stderr, "no HIP device available, skipping\n");
        return 77;
    }

    const int HD = 128;  // real target dim (Qwen3) — full 4 warps active, exercises the cross-warp reduction
    const int NKV = 1, NQ = 1, SEQ = 40;
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> Q(HD), K(SEQ*HD), V(SEQ*HD);
    for (auto& x : Q) x = dist(rng);
    for (auto& x : K) x = dist(rng);
    for (auto& x : V) x = dist(rng);

    // Identity rotation matrices (R_K = R_V = I), one layer.
    std::vector<float> R(2 * HD * HD, 0.0f);
    for (int i = 0; i < HD; i++) { R[i*HD+i] = 1.0f; R[HD*HD + i*HD+i] = 1.0f; }
    OscarRots rots{}; rots.head_dim = HD; rots.n_layers = 1; rots.n_heads_q = NQ; rots.n_heads_kv = NKV;
    rots.data = R.data();

    __half *dQ, *dK, *dV, *dOut;
    HC(hipMalloc(&dQ, HD*sizeof(__half)));
    HC(hipMalloc(&dK, (size_t)SEQ*HD*sizeof(__half)));
    HC(hipMalloc(&dV, (size_t)SEQ*HD*sizeof(__half)));
    HC(hipMalloc(&dOut, HD*sizeof(__half)));
    std::vector<__half> hQ(HD), hK(SEQ*HD), hV(SEQ*HD);
    for (int i = 0; i < HD; i++) hQ[i] = __float2half(Q[i]);
    for (int i = 0; i < SEQ*HD; i++) { hK[i] = __float2half(K[i]); hV[i] = __float2half(V[i]); }
    HC(hipMemcpy(dQ, hQ.data(), HD*2, hipMemcpyHostToDevice));
    HC(hipMemcpy(dK, hK.data(), SEQ*HD*2, hipMemcpyHostToDevice));
    HC(hipMemcpy(dV, hV.data(), SEQ*HD*2, hipMemcpyHostToDevice));

    void *dKidx, *dKscale, *dVidx, *dVscale;
    HC(hipMalloc(&dKidx, (size_t)SEQ*NKV*HD/4));
    HC(hipMalloc(&dKscale, (size_t)SEQ*NKV*sizeof(__half)));
    HC(hipMalloc(&dVidx, (size_t)SEQ*NKV*HD/4));
    HC(hipMalloc(&dVscale, (size_t)SEQ*NKV*sizeof(__half)));

    rcpp_kv_quantize_oscar_k(dK, dKidx, dKscale, SEQ, NKV, HD, 0, &rots, nullptr);
    rcpp_kv_quantize_oscar_v(dV, dVidx, dVscale, SEQ, NKV, HD, 0, &rots, nullptr);
    HC(hipDeviceSynchronize());

    float scale = 1.0f / sqrtf((float)HD);
    int rc = rcpp_kv_cache_attn_decode_oscar(dQ, dKidx, dKscale, dVidx, dVscale, dOut,
                                              NQ, NKV, HD, SEQ, 0, &rots, scale, nullptr);
    HC(hipDeviceSynchronize());
    if (rc != 0) { fprintf(stderr, "FAIL: launch returned %d\n", rc); return 1; }

    std::vector<__half> hOut(HD);
    HC(hipMemcpy(hOut.data(), dOut, HD*2, hipMemcpyDeviceToHost));

    // CPU reference: replicate the exact INT2 quantize/dequantize round-trip
    // (same LUT, same max-abs scale, same fp16 rounding) before computing
    // attention, so this isolates the kernel's softmax/accumulation math
    // from ordinary quantization noise.
    static const float LUT[4] = {-1.0f, -0.3333333f, 0.3333333f, 1.0f};
    auto quant_round_trip = [&](std::vector<float>& v) {
        for (int t = 0; t < SEQ; t++) {
            float max_abs = 1e-10f;
            for (int i = 0; i < HD; i++) {
                v[t*HD+i] = __half2float(__float2half(v[t*HD+i]));
                max_abs = std::max(max_abs, fabsf(v[t*HD+i]));
            }
            max_abs = __half2float(__float2half(max_abs));
            for (int i = 0; i < HD; i++) {
                float ratio = v[t*HD+i] / max_abs;
                int code = ratio < -0.6666667f ? 0 : ratio < 0.0f ? 1 : ratio < 0.6666667f ? 2 : 3;
                v[t*HD+i] = LUT[code] * max_abs;
            }
        }
    };
    std::vector<float> Kq = K, Vq = V;
    quant_round_trip(Kq);
    quant_round_trip(Vq);

    std::vector<float> scores(SEQ);
    float mx = -1e30f;
    for (int t = 0; t < SEQ; t++) {
        float s = 0;
        for (int i = 0; i < HD; i++) s += Q[i] * Kq[t*HD+i];
        scores[t] = s * scale;
        mx = std::max(mx, scores[t]);
    }
    float sum = 0;
    for (int t = 0; t < SEQ; t++) { scores[t] = expf(scores[t]-mx); sum += scores[t]; }
    std::vector<float> ref(HD, 0.0f);
    for (int t = 0; t < SEQ; t++)
        for (int i = 0; i < HD; i++) ref[i] += (scores[t]/sum) * Vq[t*HD+i];

    int nan_count = 0;
    float max_abs_diff = 0, max_abs_ref = 0;
    for (int i = 0; i < HD; i++) {
        float got = __half2float(hOut[i]);
        if (!std::isfinite(got)) { nan_count++; continue; }
        max_abs_diff = std::max(max_abs_diff, fabsf(got - ref[i]));
        max_abs_ref = std::max(max_abs_ref, fabsf(ref[i]));
    }
    fprintf(stderr, "nan/inf count: %d / %d\n", nan_count, HD);
    fprintf(stderr, "max abs diff vs quantize-aware fp32 reference: %.4f (ref range ~%.4f)\n", max_abs_diff, max_abs_ref);

    bool ok = nan_count == 0 && max_abs_diff < 0.05f * (max_abs_ref + 1.0f);
    if (ok) { printf("OK: finite output, matches quantize-aware fp32 reference closely\n"); return 0; }
    fprintf(stderr, "FAIL\n");
    return 1;
}
