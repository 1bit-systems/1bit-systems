// bench_fused_tq2_1024.cpp — Fused QKV + Gate/Up benchmark (TQ2_1024 format)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include "rocm_cpp/bonsai.h"

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP err %d at %s:%d\n", (int)_s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

extern "C" void fused_tq2_1024_qkv_launch(
    const uint8_t* w_q, const uint8_t* w_k, const uint8_t* w_v,
    const uint16_t* act,
    uint16_t* out_q, uint16_t* out_k, uint16_t* out_v,
    int N_q, int N_kv, int K, void* stream);

extern "C" void fused_tq2_1024_gu_launch(
    const uint8_t* w_gate, const uint8_t* w_up,
    const uint16_t* act,
    uint16_t* out_gate, uint16_t* out_up,
    int N_gs, int K, void* stream);

constexpr int HS = 2048, IS = 6144, NL = 28, NH = 16, NKV = 8, HD = 128;
constexpr int kBlockBytes_old = 34;

static size_t old_sz(int r, int c) { return (size_t)r * (size_t)(c / 128) * kBlockBytes_old; }

int main() {
    hipStream_t s;
    HIP_OK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));

    auto mk = [&](int r, int c) {
        uint8_t *a = nullptr, *d = nullptr;
        HIP_OK(hipMalloc(&a, old_sz(r, c)));
        HIP_OK(hipMemsetAsync(a, 0x55, old_sz(r, c), s));
        HIP_OK(hipDeviceSynchronize());
        bonsai_tq2_convert_to_1024(a, &d, r, c);
        HIP_OK(hipFree(a));
        return d;
    };

    // Store weights as: q,k,v,o,gate,up,down per layer
    std::vector<uint8_t*> w;
    for (int l = 0; l < NL; ++l) {
        w.push_back(mk(NH * HD, HS)); // q
        w.push_back(mk(NKV * HD, HS)); // k
        w.push_back(mk(NKV * HD, HS)); // v
        w.push_back(mk(HS, NH * HD)); // o
        w.push_back(mk(IS, HS)); // gate
        w.push_back(mk(IS, HS)); // up
        w.push_back(mk(HS, IS)); // down
    }

    uint16_t *a = nullptr;
    HIP_OK(hipMalloc(&a, 8192 * sizeof(uint16_t)));
    HIP_OK(hipMemsetAsync(a, 0x3C, 8192 * sizeof(uint16_t), s));

    uint16_t *out_q = nullptr, *out_k = nullptr, *out_v = nullptr;
    uint16_t *out_o = nullptr, *out_g = nullptr, *out_u = nullptr, *out_d = nullptr;
    HIP_OK(hipMalloc(&out_q, NH*HD * sizeof(uint16_t)));
    HIP_OK(hipMalloc(&out_k, NKV*HD * sizeof(uint16_t)));
    HIP_OK(hipMalloc(&out_v, NKV*HD * sizeof(uint16_t)));
    HIP_OK(hipMalloc(&out_o, HS * sizeof(uint16_t)));
    HIP_OK(hipMalloc(&out_g, IS * sizeof(uint16_t)));
    HIP_OK(hipMalloc(&out_u, IS * sizeof(uint16_t)));
    HIP_OK(hipMalloc(&out_d, HS * sizeof(uint16_t)));
    HIP_OK(hipDeviceSynchronize());

    printf("Allocated + converted weights. Warmup...\n");

    // Warmup using fused kernels
    for (int l = 0; l < NL; ++l) {
        int i = l * 7;
        fused_tq2_1024_qkv_launch(w[i], w[i+1], w[i+2], a, out_q, out_k, out_v, NH*HD, NKV*HD, HS, s);
        // O projection (individual - not fused)
        bonsai_tq2_1024_gemv_launch(w[i+3], a, out_o, HS, NH*HD, s);
        // Gate/Up fused
        fused_tq2_1024_gu_launch(w[i+4], w[i+5], a, out_g, out_u, IS, HS, s);
        // Down (individual)
        bonsai_tq2_1024_gemv_launch(w[i+6], a, out_d, HS, IS, s);
    }
    HIP_OK(hipDeviceSynchronize());
    printf("Warmup done.\n");

    // Benchmark fused vs individual
    const int N_RUNS = 10;

    // ── Fused path ──
    hipEvent_t t0, t1;
    HIP_OK(hipEventCreate(&t0));
    HIP_OK(hipEventCreate(&t1));

    HIP_OK(hipEventRecord(t0, s));
    for (int run = 0; run < N_RUNS; ++run) {
        for (int l = 0; l < NL; ++l) {
            int i = l * 7;
            fused_tq2_1024_qkv_launch(w[i], w[i+1], w[i+2], a, out_q, out_k, out_v, NH*HD, NKV*HD, HS, s);
            bonsai_tq2_1024_gemv_launch(w[i+3], a, out_o, HS, NH*HD, s);
            fused_tq2_1024_gu_launch(w[i+4], w[i+5], a, out_g, out_u, IS, HS, s);
            bonsai_tq2_1024_gemv_launch(w[i+6], a, out_d, HS, IS, s);
        }
    }
    HIP_OK(hipEventRecord(t1, s));
    HIP_OK(hipEventSynchronize(t1));

    float ms_fused;
    HIP_OK(hipEventElapsedTime(&ms_fused, t0, t1));
    double pt_fused = ms_fused / N_RUNS;
    double ts_fused = 1000.0 / pt_fused;

    // ── Individual path (baseline) ──
    HIP_OK(hipEventRecord(t0, s));
    for (int run = 0; run < N_RUNS; ++run) {
        for (int l = 0; l < NL; ++l) {
            int i = l * 7;
            bonsai_tq2_1024_gemv_launch(w[i],   a, out_q, NH*HD, HS, s);
            bonsai_tq2_1024_gemv_launch(w[i+1], a, out_k, NKV*HD, HS, s);
            bonsai_tq2_1024_gemv_launch(w[i+2], a, out_v, NKV*HD, HS, s);
            bonsai_tq2_1024_gemv_launch(w[i+3], a, out_o, HS, NH*HD, s);
            bonsai_tq2_1024_gemv_launch(w[i+4], a, out_g, IS, HS, s);
            bonsai_tq2_1024_gemv_launch(w[i+5], a, out_u, IS, HS, s);
            bonsai_tq2_1024_gemv_launch(w[i+6], a, out_d, HS, IS, s);
        }
    }
    HIP_OK(hipEventRecord(t1, s));
    HIP_OK(hipEventSynchronize(t1));

    float ms_indiv;
    HIP_OK(hipEventElapsedTime(&ms_indiv, t0, t1));
    double pt_indiv = ms_indiv / N_RUNS;
    double ts_indiv = 1000.0 / pt_indiv;

    printf("\n═══════════════════════════════════════════════\n");
    printf("  Fused TQ2_1024 — 28-layer model\n");
    printf("═══════════════════════════════════════════════\n");
    printf("  Individual:     %.3f ms/tok, %5.0f tok/s\n", pt_indiv, ts_indiv);
    printf("  Fused (QKV+GU): %.3f ms/tok, %5.0f tok/s\n", pt_fused, ts_fused);
    printf("  Speedup:        %.2f×\n", ts_fused / ts_indiv);
    printf("  Launches saved: %d → %d\n", 7 * NL, 4 * NL);
    printf("═══════════════════════════════════════════════\n");

    for (auto p : w) hipFree(p);
    hipFree(a);
    hipFree(out_q); hipFree(out_k); hipFree(out_v);
    hipFree(out_o); hipFree(out_g); hipFree(out_u); hipFree(out_d);
    HIP_OK(hipEventDestroy(t0));
    HIP_OK(hipEventDestroy(t1));
    HIP_OK(hipStreamDestroy(s));
    return 0;
}
