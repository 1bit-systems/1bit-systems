// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
// Sherry — see LICENSE-SHERRY.md and SHERRY-FILES.txt at the repo root.
// Commercial use requires a separate license.
//
// test_sherry_e2e.cpp — minimal end-to-end smoke test for the fp16-Sherry
// dispatch path.
//
// What it checks:
//   1. rcpp_bitnet_load_h1b succeeds on the real sherry-v4 model file.
//   2. The loader resolves WeightFormat::SHERRY_FP16 from the v3 header +
//      H1B_FLAG_SHERRY_FP16 bit (cfg[8]).
//   3. One decode-token forward pass (Q/K/V/O + gate/up/down across all L
//      layers + tied LM head) completes with no HIP launch error.
//   4. The resulting fp32 logits buffer is non-trivial (non-zero, finite).
//
// What it does NOT check:
//   - Quality / coherence of the generated token. With a 12.98% forced-zero
//     flip rate from the post-hoc TQ1→Sherry requantizer this kernel path
//     will not produce state-of-the-art tokens until the Sherry-aware
//     fine-tune lands. We only care that the pipe turns over.
//
// Expected runtime: ~1 s on gfx1151 once weights are in LPDDR5 page cache.
// Binary: built as an executable under tests/ — it is NOT wired into the
// librocm_cpp target. Run by hand on the strix-halo box.

#include "rocm_cpp/bitnet_model.h"
#include "rocm_cpp/ck_gemm.h"
#include "rocm_cpp/sherry.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                (int)_s, hipGetErrorString(_s), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)
#define RC_OK(e) do { \
    rcpp_status_t _s = (e); \
    if (_s != RCPP_OK) { \
        fprintf(stderr, "rcpp err %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

// Mirrors the fp16-Sherry dispatch in tools/bitnet_decode.cpp. Kept here
// rather than factored into a shared helper to keep the loader path under
// test 1:1 with what ships in the CLI — if the CLI dispatch drifts this
// test also drifts, and we catch the divergence at build time.
__global__ static void sherry_fp16_apply_row_scale_kernel(
    __half*          __restrict__ y,
    const float*     __restrict__ row_scales,
    int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float v = (float)y[n] * row_scales[n];
    constexpr float FP16_MAX = 65504.0f;
    float vc = v;
    if (vc >  FP16_MAX) vc =  FP16_MAX;
    if (vc < -FP16_MAX) vc = -FP16_MAX;
    y[n] = __float2half(vc);
}

static void sherry_gemv(const void* packed, const void* act_fp16,
                        const float* row_scales, void* out_fp16,
                        int N, int K)
{
    sherry_ternary_gemv_launch(
        static_cast<const uint8_t*>(packed),
        static_cast<const uint16_t*>(act_fp16),
        static_cast<uint16_t*>(out_fp16),
        N, K,
        /*stream=*/nullptr);
    const int BLOCK = 128;
    dim3 grid((unsigned)((N + BLOCK - 1) / BLOCK), 1, 1);
    dim3 block((unsigned)BLOCK, 1, 1);
    hipLaunchKernelGGL(sherry_fp16_apply_row_scale_kernel,
                       grid, block, 0, nullptr,
                       static_cast<__half*>(out_fp16),
                       row_scales,
                       N);
}

int main(int argc, char** argv) {
    const char* path = (argc > 1)
        ? argv[1]
        : "/home/bcloud/halo-ai/models/halo-1bit-2b-sherry-v4.h1b";

    fprintf(stderr, "[test_sherry_e2e] loading %s\n", path);

    rcpp_bitnet_model_t m{};
    RC_OK(rcpp_bitnet_load_h1b(path, &m));

    fprintf(stderr,
            "[test_sherry_e2e] version=%d flags=0x%x weight_format=%d "
            "hs=%d is=%d L=%d nh=%d nkv=%d V=%d\n",
            m.format_version, m.flags, (int)m.weight_format,
            m.hidden_size, m.intermediate_size, m.num_layers,
            m.num_heads, m.num_kv_heads, m.vocab_size);

    if (m.weight_format != RCPP_WEIGHT_FORMAT_SHERRY_FP16) {
        fprintf(stderr,
                "[test_sherry_e2e] FAIL: expected SHERRY_FP16 dispatch tag, "
                "got %d. Flag bit parsing or model is wrong.\n",
                (int)m.weight_format);
        return 1;
    }
    if ((m.flags & H1B_FLAG_SHERRY_FP16) == 0) {
        fprintf(stderr, "[test_sherry_e2e] FAIL: H1B_FLAG_SHERRY_FP16 not set.\n");
        return 1;
    }

    const int hs = m.hidden_size;
    const int is = m.intermediate_size;
    const int nh = m.num_heads;
    const int nkv = m.num_kv_heads;
    const int hd = hs / nh;
    const int L = m.num_layers;
    const int V = m.vocab_size;
    const float scale = 1.0f / std::sqrt((float)hd);

    // Scratch buffers — decode shape (batch=1, one token).
    float*    x_fp32       = nullptr;
    _Float16* x            = nullptr;
    _Float16* normed       = nullptr;
    int8_t*   x_i8         = nullptr;  // unused on fp16 path, kept for quant launcher
    float*    x_scale_dev  = nullptr;
    _Float16* q_fp16       = nullptr;
    _Float16* k_fp16       = nullptr;
    _Float16* v_fp16       = nullptr;
    _Float16* o_fp16       = nullptr;
    _Float16* gate_fp16    = nullptr;
    _Float16* up_fp16      = nullptr;
    _Float16* down_fp16    = nullptr;
    _Float16* silu_out     = nullptr;
    int8_t*   silu_i8      = nullptr;
    float*    silu_scale_dev = nullptr;
    _Float16* K_cache      = nullptr;
    _Float16* V_cache      = nullptr;
    float*    logits       = nullptr;

    HIP_OK(hipMalloc(&x_fp32, hs * sizeof(float)));
    HIP_OK(hipMalloc(&x, hs * sizeof(_Float16)));
    HIP_OK(hipMalloc(&normed, hs * sizeof(_Float16)));
    HIP_OK(hipMalloc(&x_i8, hs));
    HIP_OK(hipMemsetAsync(x_i8, 0, hs, nullptr));
    HIP_OK(hipMalloc(&x_scale_dev, sizeof(float)));
    HIP_OK(hipMalloc(&q_fp16, nh * hd * sizeof(_Float16)));
    HIP_OK(hipMalloc(&k_fp16, nkv * hd * sizeof(_Float16)));
    HIP_OK(hipMalloc(&v_fp16, nkv * hd * sizeof(_Float16)));
    HIP_OK(hipMalloc(&o_fp16, hs * sizeof(_Float16)));
    HIP_OK(hipMalloc(&gate_fp16, is * sizeof(_Float16)));
    HIP_OK(hipMalloc(&up_fp16, is * sizeof(_Float16)));
    HIP_OK(hipMalloc(&down_fp16, hs * sizeof(_Float16)));
    HIP_OK(hipMalloc(&silu_out, is * sizeof(_Float16)));
    HIP_OK(hipMalloc(&silu_i8, is));
    HIP_OK(hipMemsetAsync(silu_i8, 0, is, nullptr));
    HIP_OK(hipMalloc(&silu_scale_dev, sizeof(float)));
    HIP_OK(hipMalloc(&logits, V * sizeof(float)));

    // One-layer-worth of KV slabs — but we need L copies for the forward pass.
    // Keep lifetime simple: allocate a vector of pointers, free at exit.
    std::vector<_Float16*> Ks(L, nullptr), Vs(L, nullptr);
    const size_t kv_slab = (size_t)nkv * hd * sizeof(_Float16);  // one slot
    for (int l = 0; l < L; ++l) {
        HIP_OK(hipMalloc(&Ks[l], kv_slab));
        HIP_OK(hipMalloc(&Vs[l], kv_slab));
    }

    // Seed position. Token id 1 is BitNet's BOS in all checked tokenizers —
    // using a single-token forward pass keeps the KV cache at size 1, which
    // is the minimum shape the decode attention accepts.
    const int token_id = 1;
    const int pos = 0;

    RC_OK(rcpp_embedding_lookup_fp16(m.embedding_dev, token_id, x, hs, nullptr));
    HIP_OK(hipMemsetAsync(x_fp32, 0, hs * sizeof(float), nullptr));
    RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, x, hs, nullptr));

    for (int l = 0; l < L; ++l) {
        rcpp_bitnet_layer_t& ly = m.layers[l];

        // Attention block.
        RC_OK(rcpp_rmsnorm_fp32_in_fp16_out(x_fp32, ly.input_norm_dev, normed,
                                            m.rms_norm_eps, hs, nullptr));
        // Not used on fp16 path, but exercised so the quant launcher doesn't rot.
        RC_OK(rcpp_quantize_fp16_to_i8(normed, x_i8, x_scale_dev, hs, nullptr));

        sherry_gemv(ly.q_packed_dev, normed, ly.q_scales_dev, q_fp16, nh*hd,  hs);
        sherry_gemv(ly.k_packed_dev, normed, ly.k_scales_dev, k_fp16, nkv*hd, hs);
        sherry_gemv(ly.v_packed_dev, normed, ly.v_scales_dev, v_fp16, nkv*hd, hs);

        RC_OK(rcpp_rope_fp16(q_fp16, pos, m.rope_theta, nh,  hd, nullptr));
        RC_OK(rcpp_rope_fp16(k_fp16, pos, m.rope_theta, nkv, hd, nullptr));

        HIP_OK(hipMemcpy(Ks[l], k_fp16, nkv*hd*2, hipMemcpyDeviceToDevice));
        HIP_OK(hipMemcpy(Vs[l], v_fp16, nkv*hd*2, hipMemcpyDeviceToDevice));
        RC_OK(rcpp_kv_cache_attn_decode_fd(q_fp16, Ks[l], Vs[l],
                                           o_fp16, nh, nkv, hd, pos+1, scale, nullptr));

        RC_OK(rcpp_rmsnorm_fp16(o_fp16, ly.attn_sub_norm_dev, normed,
                                m.rms_norm_eps, hs, nullptr));
        sherry_gemv(ly.o_packed_dev, normed, ly.o_scales_dev, o_fp16, hs, nh*hd);
        RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, o_fp16, hs, nullptr));

        // FFN block.
        RC_OK(rcpp_rmsnorm_fp32_in_fp16_out(x_fp32, ly.post_attn_norm_dev, normed,
                                            m.rms_norm_eps, hs, nullptr));
        sherry_gemv(ly.gate_packed_dev, normed, ly.gate_scales_dev, gate_fp16, is, hs);
        sherry_gemv(ly.up_packed_dev,   normed, ly.up_scales_dev,   up_fp16,   is, hs);

        RC_OK(rcpp_relu2_glu_rmsnorm_fp16(gate_fp16, up_fp16, ly.ffn_sub_norm_dev,
                                          silu_out, m.rms_norm_eps, is, nullptr));
        sherry_gemv(ly.down_packed_dev, silu_out, ly.down_scales_dev, down_fp16, hs, is);
        RC_OK(rcpp_residual_add_fp32_from_fp16(x_fp32, down_fp16, hs, nullptr));
    }

    RC_OK(rcpp_rmsnorm_fp32_in_fp16_out(x_fp32, m.final_norm_weight_dev, normed,
                                        m.rms_norm_eps, hs, nullptr));
    RC_OK(rcpp_fp16_gemv(m.embedding_dev, normed, logits, V, hs, nullptr));

    HIP_OK(hipDeviceSynchronize());
    HIP_OK(hipGetLastError());

    // Pull logits + assert non-trivial.
    std::vector<float> host_logits((size_t)V);
    HIP_OK(hipMemcpy(host_logits.data(), logits, V * sizeof(float), hipMemcpyDeviceToHost));

    bool any_nonzero = false;
    bool all_finite  = true;
    float lmax = -1e30f, lmin = 1e30f;
    int argmax_idx = 0;
    for (int i = 0; i < V; ++i) {
        const float v = host_logits[i];
        if (!std::isfinite(v)) all_finite = false;
        if (v != 0.0f) any_nonzero = true;
        if (v > lmax) { lmax = v; argmax_idx = i; }
        if (v < lmin) lmin = v;
    }

    fprintf(stderr, "[test_sherry_e2e] logits: argmax=%d max=%.3f min=%.3f\n",
            argmax_idx, lmax, lmin);

    if (!any_nonzero) {
        fprintf(stderr, "[test_sherry_e2e] FAIL: all logits are zero — pipeline returned trivial output.\n");
        return 1;
    }
    if (!all_finite) {
        fprintf(stderr, "[test_sherry_e2e] FAIL: non-finite logit(s). Check fp16 clamps + row-scale post-pass.\n");
        return 1;
    }

    // Housekeeping. HIP frees on process exit are fine, but explicit is nicer.
    // Ignore hipFree returns — if a free fails on shutdown there's nothing
    // productive to do. Cast to void to silence [[nodiscard]].
    auto hfree = [](void* p) { (void)hipFree(p); };
    for (int l = 0; l < L; ++l) { hfree(Ks[l]); hfree(Vs[l]); }
    hfree(x_fp32); hfree(x); hfree(normed); hfree(x_i8); hfree(x_scale_dev);
    hfree(q_fp16); hfree(k_fp16); hfree(v_fp16); hfree(o_fp16);
    hfree(gate_fp16); hfree(up_fp16); hfree(down_fp16);
    hfree(silu_out); hfree(silu_i8); hfree(silu_scale_dev);
    hfree(logits);
    rcpp_bitnet_free(&m);

    fprintf(stderr, "[test_sherry_e2e] PASS\n");
    return 0;
}
