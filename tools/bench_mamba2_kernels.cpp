// bench_mamba2_kernels.cpp — Microbenchmark for Mamba2 ROCm kernels
//
// Measures per-kernel latency and token throughput for the tuned kernels
// vs naive baseline kernels on Zamba2-2.7B shapes.
//
// Build & run:
//   cd build && cmake .. && make bench_mamba2_kernels -j$(nproc)
//   ./bench_mamba2_kernels [iterations=100]

#include "../src/mamba2_kernels.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { \
    fprintf(stderr, "HIP Error %d at %s:%d: %s\n", _s, __FILE__, __LINE__, hipGetErrorString(_s)); \
    exit(1); }} while(0)

// Zamba2-2.7B
static const int D_MODEL   = 2560;
static const int D_INNER   = 5120;
static const int D_STATE   = 64;
static const int D_CONV    = 4;
static const int N_HEAD    = 80;
static const int N_GROUP   = 1;
static const int HEAD_DIM  = 64;
static const int CONV_DIM  = D_INNER + 2 * N_GROUP * D_STATE;
static const int DIM_IN_PROJ = D_INNER + CONV_DIM + N_HEAD;

static std::vector<float> randvec(int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> v(n);
    for (int i = 0; i < n; ++i) v[i] = dist(rng);
    return v;
}

static float* dev_alloc(hipStream_t st, const std::vector<float>& src) {
    float* d = nullptr;
    HIP_CHECK(hipMalloc(&d, src.size() * sizeof(float)));
    HIP_CHECK(hipMemcpyAsync(d, src.data(), src.size() * sizeof(float), hipMemcpyHostToDevice, st));
    return d;
}

int main(int argc, char** argv) {
    int iterations = argc > 1 ? atoi(argv[1]) : 100;
    if (iterations < 10) iterations = 10;

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Mamba2 ROCm Kernel Benchmark Suite                 ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");
    printf("Config: d_model=%d d_inner=%d d_state=%d d_conv=%d\n"
           "        n_head=%d n_group=%d head_dim=%d\n"
           "        conv_dim=%d dim_in_proj=%d\n"
           "        iterations=%d\n\n",
           D_MODEL, D_INNER, D_STATE, D_CONV, N_HEAD, N_GROUP, HEAD_DIM,
           CONV_DIM, DIM_IN_PROJ, iterations);

    std::mt19937 rng(42);
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // ── Allocate device data ──
    auto h_x       = randvec(D_MODEL, rng);
    auto h_w_inp   = randvec(DIM_IN_PROJ * D_MODEL, rng);
    auto h_w_conv  = randvec(D_CONV * CONV_DIM, rng);
    auto h_b_conv  = randvec(CONV_DIM, rng);
    auto h_dt_bias = randvec(N_HEAD, rng);
    auto h_A_log   = randvec(N_HEAD, rng);
    auto h_D       = randvec(N_HEAD, rng);
    auto h_w_out   = randvec(D_MODEL * D_INNER, rng);
    auto h_norm_w  = randvec((D_INNER/N_GROUP)*(D_INNER/N_GROUP), rng);

    float *d_x, *d_w_inp, *d_w_conv, *d_b_conv, *d_dt_bias;
    float *d_A_log, *d_D, *d_w_out, *d_norm_w, *d_cs, *d_ss, *d_y, *d_tmp;
    hipEvent_t t0, t1;

    d_x      = dev_alloc(stream, h_x);
    d_w_inp  = dev_alloc(stream, h_w_inp);
    d_w_conv = dev_alloc(stream, h_w_conv);
    d_b_conv = dev_alloc(stream, h_b_conv);
    d_dt_bias= dev_alloc(stream, h_dt_bias);
    d_A_log  = dev_alloc(stream, h_A_log);
    d_D      = dev_alloc(stream, h_D);
    d_w_out  = dev_alloc(stream, h_w_out);
    d_norm_w = dev_alloc(stream, h_norm_w);

    std::vector<float> zero_cs((D_CONV-1)*CONV_DIM, 0.0f);
    std::vector<float> zero_ss(D_STATE*D_INNER, 0.0f);
    d_cs = dev_alloc(stream, zero_cs);
    d_ss = dev_alloc(stream, zero_ss);

    int tmp_size = std::max(DIM_IN_PROJ, std::max(CONV_DIM, D_INNER));
    HIP_CHECK(hipMalloc(&d_tmp, tmp_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, D_MODEL * sizeof(float)));
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));
    HIP_CHECK(hipStreamSynchronize(stream));

    // ── Warmup (10 iterations) ──
    for (int i = 0; i < 10; i++) {
        mamba2_gpu_decode_block_tuned(
            d_x, d_w_inp, d_w_conv, d_b_conv, d_dt_bias, d_A_log, d_D, nullptr, d_w_out,
            d_cs, d_ss, d_y, d_tmp, D_MODEL, D_INNER, D_STATE, D_CONV,
            N_HEAD, N_GROUP, HEAD_DIM, CONV_DIM, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    printf("Warmup complete.\n\n");

    // ════════════════════════════════════════════════════════
    // BENCHMARK 1: Full decode block (tuned)
    // ════════════════════════════════════════════════════════
    printf("── Benchmark 1: Tuned Decode Block (Zamba2-2.7B) ──\n");
    float tuned_ms = 0.0f;
    HIP_CHECK(hipEventRecord(t0, stream));
    for (int i = 0; i < iterations; i++) {
        mamba2_gpu_decode_block_tuned(
            d_x, d_w_inp, d_w_conv, d_b_conv, d_dt_bias, d_A_log, d_D, nullptr, d_w_out,
            d_cs, d_ss, d_y, d_tmp, D_MODEL, D_INNER, D_STATE, D_CONV,
            N_HEAD, N_GROUP, HEAD_DIM, CONV_DIM, stream);
    }
    HIP_CHECK(hipEventRecord(t1, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipEventElapsedTime(&tuned_ms, t0, t1));
    tuned_ms /= iterations;

    printf("  Avg per-token:     %.3f ms\n", tuned_ms);
    printf("  Tokens/sec:        %.0f\n", 1000.0f / tuned_ms);
    printf("  Total %d tokens:   %.1f ms\n", iterations, tuned_ms * iterations);

    // ════════════════════════════════════════════════════════
    // BENCHMARK 2: Sub-kernel breakdown
    // ════════════════════════════════════════════════════════

    // 2a. Tiled GEMV (in_proj) — 2 rows/block
    printf("\n── Benchmark 2a: Tiled GEMV (in_proj) ──\n");
    {
        float* d_buf;
        HIP_CHECK(hipMalloc(&d_buf, DIM_IN_PROJ * sizeof(float)));
        int gemv_blocks = (DIM_IN_PROJ + 2 - 1) / 2;
        float ms = 0.0f;
        HIP_CHECK(hipEventRecord(t0, stream));
        for (int i = 0; i < iterations; i++) {
            hipLaunchKernelGGL(mamba2_tiled_gemv_kernel,
                dim3(gemv_blocks), dim3(256), 0, stream,
                d_w_inp, d_x, d_buf, DIM_IN_PROJ, D_MODEL);
        }
        HIP_CHECK(hipEventRecord(t1, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        printf("  Avg: %.4f ms (%.0f tok/s)\n", ms/iterations, 1000.0f*iterations/ms);
        HIP_CHECK(hipFree(d_buf));
    }

    // 2b. Conv1D tuned
    printf("\n── Benchmark 2b: Conv1D (tuned, decode) ──\n");
    {
        int conv_tiles = (CONV_DIM + 128 - 1) / 128;
        float ms = 0.0f;
        HIP_CHECK(hipEventRecord(t0, stream));
        for (int i = 0; i < iterations; i++) {
            hipLaunchKernelGGL(mamba2_conv1d_tuned_kernel,
                dim3(conv_tiles, 1), dim3(256), 0, stream,
                d_tmp, d_w_conv, d_b_conv, d_tmp, d_cs,
                1, 1, D_CONV, CONV_DIM);
        }
        HIP_CHECK(hipEventRecord(t1, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        printf("  Avg: %.4f ms (%.0f tok/s)\n", ms/iterations, 1000.0f*iterations/ms);
    }

    // 2c. Fused selective scan
    printf("\n── Benchmark 2c: Fused Selective Scan ──\n");
    {
        float ms = 0.0f;
        dim3 scan_grid(N_HEAD, 1, 1);
        dim3 scan_block(64, 1, 1);
        HIP_CHECK(hipEventRecord(t0, stream));
        for (int i = 0; i < iterations; i++) {
            hipLaunchKernelGGL(mamba2_scan_fused_kernel,
                scan_grid, scan_block, 0, stream,
                d_tmp, d_tmp, d_dt_bias, d_A_log,
                d_tmp, d_tmp, d_D,
                d_y, d_ss,
                1, 1, D_INNER, D_STATE, N_HEAD, N_GROUP, HEAD_DIM);
        }
        HIP_CHECK(hipEventRecord(t1, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        printf("  Avg: %.4f ms (%.0f tok/s)\n", ms/iterations, 1000.0f*iterations/ms);
    }

    // 2d. Tiled GEMV (out_proj)
    printf("\n── Benchmark 2d: Tiled GEMV (out_proj) ──\n");
    {
        int gemv_blocks = (D_MODEL + 2 - 1) / 2;
        float ms = 0.0f;
        HIP_CHECK(hipEventRecord(t0, stream));
        for (int i = 0; i < iterations; i++) {
            hipLaunchKernelGGL(mamba2_tiled_gemv_kernel,
                dim3(gemv_blocks), dim3(256), 0, stream,
                d_w_out, d_y, d_y, D_MODEL, D_INNER);
        }
        HIP_CHECK(hipEventRecord(t1, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        printf("  Avg: %.4f ms (%.0f tok/s)\n", ms/iterations, 1000.0f*iterations/ms);
    }

    // ════════════════════════════════════════════════════════
    // BENCHMARK 3: Speedup vs naive baseline
    // ════════════════════════════════════════════════════════
    // Re-measure GEMV with the tiled kernel specifically for comparison
    printf("\n── Benchmark 3: Speedup vs baseline (naive 1-thread/row GEMV) ──\n");
    float tiled_gemv_ms, legacy_gemv_ms;
    {
        float* d_buf;
        HIP_CHECK(hipMalloc(&d_buf, DIM_IN_PROJ * sizeof(float)));
        int gemv_blocks = (DIM_IN_PROJ + 2 - 1) / 2;
        // Tiled GEMV
        float ms_t = 0.0f;
        HIP_CHECK(hipEventRecord(t0, stream));
        for (int i = 0; i < iterations; i++) {
            hipLaunchKernelGGL(mamba2_tiled_gemv_kernel,
                dim3(gemv_blocks), dim3(256), 0, stream,
                d_w_inp, d_x, d_buf, DIM_IN_PROJ, D_MODEL);
        }
        HIP_CHECK(hipEventRecord(t1, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipEventElapsedTime(&ms_t, t0, t1));
        tiled_gemv_ms = ms_t / iterations;
        // Legacy GEMV
        int threads = 256;
        int blocks = (DIM_IN_PROJ + threads - 1) / threads;
        float ms_l = 0.0f;
        HIP_CHECK(hipEventRecord(t0, stream));
        for (int i = 0; i < iterations; i++) {
            hipLaunchKernelGGL(gemv_kernel,
                dim3(blocks), dim3(threads), 0, stream,
                d_w_inp, d_x, d_buf, DIM_IN_PROJ, D_MODEL);
        }
        HIP_CHECK(hipEventRecord(t1, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipEventElapsedTime(&ms_l, t0, t1));
        legacy_gemv_ms = ms_l / iterations;
        printf("  GEMV (d_in_proj=%d x d_model=%d):\n", DIM_IN_PROJ, D_MODEL);
        printf("    Legacy (1-thread/row):  %.4f ms\n", legacy_gemv_ms);
        printf("    Tiled (LDS-cached):     %.4f ms\n", tiled_gemv_ms);
        printf("    Speedup:                %.1f×\n", legacy_gemv_ms / tiled_gemv_ms);
        HIP_CHECK(hipFree(d_buf));
    }

    // ════════════════════════════════════════════════════════
    // SUMMARY
    // ════════════════════════════════════════════════════════
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║  Benchmark Results                               ║\n");
    printf("╠══════════════════════════════════════════════════╣\n");
    printf("║  Zamba2-2.7B decode:                             ║\n");
    printf("║    Per-token:      %7.3f ms                      ║\n", tuned_ms);
    printf("║    Tokens/sec:     %7.0f tok/s                   ║\n", 1000.0f/tuned_ms);
    printf("║                                                    ║\n");
    printf("║  Estimated full-model (45 Mamba2 layers):        ║\n");
    float full_model = tuned_ms * 45;
    printf("║    Per-token:      %7.3f ms                      ║\n", full_model);
    printf("║    Tokens/sec:     %7.0f tok/s                   ║\n", 1000.0f/full_model);
    printf("╚══════════════════════════════════════════════════╝\n");

    // ── Cleanup ──
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipEventDestroy(t0));
    HIP_CHECK(hipEventDestroy(t1));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_w_inp));
    HIP_CHECK(hipFree(d_w_conv));
    HIP_CHECK(hipFree(d_b_conv));
    HIP_CHECK(hipFree(d_dt_bias));
    HIP_CHECK(hipFree(d_A_log));
    HIP_CHECK(hipFree(d_D));
    HIP_CHECK(hipFree(d_w_out));
    HIP_CHECK(hipFree(d_cs));
    HIP_CHECK(hipFree(d_ss));
    HIP_CHECK(hipFree(d_tmp));
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
