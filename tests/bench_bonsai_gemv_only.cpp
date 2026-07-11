// bench_bonsai_gemv_only.cpp — measure ONLY the 1-bit GEMVs for full model
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

#include "rocm_cpp/bonsai.h"

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                (int)_s, hipGetErrorString(_s), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

constexpr int kQ1BlockBytes = 18;
constexpr int HS = 2048, IS = 6144, NL = 28, NH = 16, NKV = 8, HD = 128;

static size_t q1b(int rows, int cols) {
    return (size_t)rows * (size_t)(cols / 128) * (size_t)kQ1BlockBytes;
}

int main() {
    hipStream_t stream;
    HIP_OK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    // Allocate ALL weight buffers (synthetic random Q1_0)
    auto mk_w = [&](int r, int c, const char* name) {
        size_t bytes = q1b(r, c);
        uint8_t* d = nullptr;
        HIP_OK(hipMalloc(&d, bytes));
        // Fill with random-ish data on device
        HIP_OK(hipMemsetAsync(d, 0x55, bytes, stream));
        return d;
    };

    std::vector<uint8_t*> q(NL), k(NL), v(NL), o_(NL);
    std::vector<uint8_t*> gate(NL), up(NL), down(NL);
    // lm_head skipped (151669 not divisible by 128 for Q1_0 kernel)

    for (int l = 0; l < NL; ++l) {
        q[l]    = mk_w(NH*HD, HS, "Q");
        k[l]    = mk_w(NKV*HD, HS, "K");
        v[l]    = mk_w(NKV*HD, HS, "V");
        o_[l]   = mk_w(HS, NH*HD, "O");
        gate[l] = mk_w(IS, HS, "Gate");
        up[l]   = mk_w(IS, HS, "Up");
        down[l] = mk_w(HS, IS, "Down");
    }
    // lm_head skipped

    // Activation buffers
    uint16_t *d_act = nullptr;
    HIP_OK(hipMalloc(&d_act, 8192 * sizeof(uint16_t)));
    HIP_OK(hipMemsetAsync(d_act, 0, 8192 * sizeof(uint16_t), stream));

    // Output buffers
    uint16_t *d_out = nullptr;
    HIP_OK(hipMalloc(&d_out, 16384 * sizeof(uint16_t)));
    HIP_OK(hipMemsetAsync(d_out, 0, 16384 * sizeof(uint16_t), stream));

    HIP_OK(hipDeviceSynchronize());

    printf("Allocated %zu GEMV weight buffers. Warming up...\n", size_t(7*NL+1));

    // Warmup: all GEMVs once
    for (int l = 0; l < NL; ++l) {
        bonsai_q1_gemv_launch(q[l],    d_act, d_out, NH*HD, HS, stream);
        bonsai_q1_gemv_launch(k[l],    d_act, d_out, NKV*HD, HS, stream);
        bonsai_q1_gemv_launch(v[l],    d_act, d_out, NKV*HD, HS, stream);
        bonsai_q1_gemv_launch(o_[l],   d_act, d_out, HS, NH*HD, stream);
        bonsai_q1_gemv_launch(gate[l], d_act, d_out, IS, HS, stream);
        bonsai_q1_gemv_launch(up[l],   d_act, d_out, IS, HS, stream);
        bonsai_q1_gemv_launch(down[l], d_act, d_out, HS, IS, stream);
    }
    HIP_OK(hipDeviceSynchronize());
    printf("Warmup done.\n");

    // Benchmark: chain all 197 GEMVs per layer in a tight loop
    const int N_RUNS = 5;
    printf("Benchmarking %d runs of 197 GEMVs each...\n", N_RUNS);

    hipEvent_t t0, t1;
    HIP_OK(hipEventCreate(&t0));
    HIP_OK(hipEventCreate(&t1));

    HIP_OK(hipEventRecord(t0, stream));
    for (int run = 0; run < N_RUNS; ++run) {
        for (int l = 0; l < NL; ++l) {
            bonsai_q1_gemv_launch(q[l],    d_act, d_out, NH*HD, HS, stream);
            bonsai_q1_gemv_launch(k[l],    d_act, d_out, NKV*HD, HS, stream);
            bonsai_q1_gemv_launch(v[l],    d_act, d_out, NKV*HD, HS, stream);
            bonsai_q1_gemv_launch(o_[l],   d_act, d_out, HS, NH*HD, stream);
            bonsai_q1_gemv_launch(gate[l], d_act, d_out, IS, HS, stream);
            bonsai_q1_gemv_launch(up[l],   d_act, d_out, IS, HS, stream);
            bonsai_q1_gemv_launch(down[l], d_act, d_out, HS, IS, stream);
        }
        // lm_head skipped
    }
    HIP_OK(hipEventRecord(t1, stream));
    HIP_OK(hipEventSynchronize(t1));

    float ms;
    HIP_OK(hipEventElapsedTime(&ms, t0, t1));
    double per_token = ms / (double)N_RUNS;
    double tok_s = 1000.0 / per_token;

    // Calculate total weight bytes read
    size_t total_bytes = 0;
    for (int l = 0; l < NL; ++l) {
        total_bytes += q1b(NH*HD, HS);   // Q
        total_bytes += q1b(NKV*HD, HS);  // K
        total_bytes += q1b(NKV*HD, HS);  // V
        total_bytes += q1b(HS, NH*HD);   // O
        total_bytes += q1b(IS, HS);      // gate
        total_bytes += q1b(IS, HS);      // up
        total_bytes += q1b(HS, IS);      // down
    }
    // lm_head skipped (estimated ~44 MB additional at ~1200 tok/s)
    double total_mb = total_bytes / 1e6;
    double bw = total_bytes / (per_token / 1000.0) / 1e9;

    printf("\n════════════════════════════════════════════\n");
    printf("  1-bit GEMVs only (no norms/attention)\n");
    printf("════════════════════════════════════════════\n");
    printf("  Model weights:  %.0f MB (Q1_0 1-bit)\n", total_mb);
    printf("  Total time:     %.1f ms over %d runs\n", ms, N_RUNS);
    printf("  Per token:      %.3f ms\n", per_token);
    printf("  Throughput:     %.1f tok/s\n", tok_s);
    printf("  Effective BW:   %.0f GB/s\n", bw);
    printf("  Peak BW util:   %.0f%%\n", bw / 273.0 * 100.0);
    printf("════════════════════════════════════════════\n");
    printf("  With norms+attn (~1.5 ms): ~%.0f tok/s\n", 1000.0/(per_token + 1.5));
    printf("════════════════════════════════════════════\n");

    // Cleanup
    for (int l = 0; l < NL; ++l) {
        hipFree(q[l]); hipFree(k[l]); hipFree(v[l]); hipFree(o_[l]);
        hipFree(gate[l]); hipFree(up[l]); hipFree(down[l]);
    }
    hipFree(d_act); hipFree(d_out);
    HIP_OK(hipEventDestroy(t0));
    HIP_OK(hipEventDestroy(t1));
    HIP_OK(hipStreamDestroy(stream));
    return 0;
}
