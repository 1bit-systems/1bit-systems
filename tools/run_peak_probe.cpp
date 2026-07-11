// Quick peak probe runner — measures raw WMMA throughput ceiling
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" {
void rcpp_peak_probe_launch(const void*, const void*, void*, int, int, void*);
void rcpp_peak_probe_launch_i8(const void*, const void*, void*, int, int, void*);
}

int main() {
    hipStream_t stream;
    hipStreamCreate(&stream);

    int n_blocks = 40 * 32;  // 40 CUs × 32 waves/CU (keep all WGP fully loaded)
    int n_inner = 10000;

    size_t seed_bytes = 32 * 4 * 16 * sizeof(__half);
    size_t sink_fp32_bytes = n_blocks * 32 * sizeof(float);
    size_t sink_i32_bytes = n_blocks * 32 * sizeof(int32_t);

    __half *A_d, *B_d;
    float *sink_d;
    int8_t *Ai8_d, *Bi8_d;
    int32_t *sink_i32_d;

    hipMalloc(&A_d, seed_bytes);
    hipMalloc(&B_d, seed_bytes);
    hipMalloc(&sink_d, sink_fp32_bytes);
    hipMalloc(&Ai8_d, seed_bytes);
    hipMalloc(&Bi8_d, seed_bytes);
    hipMalloc(&sink_i32_d, sink_i32_bytes);

    // FP16 WMMA peak
    printf("=== FP16 WMMA Peak Probe ===\n");
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Warmup
    rcpp_peak_probe_launch(A_d, B_d, sink_d, n_blocks, n_inner, stream);
    hipStreamSynchronize(stream);

    hipEventRecord(start, stream);
    rcpp_peak_probe_launch(A_d, B_d, sink_d, n_blocks, n_inner, stream);
    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);

    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);

    // Each wave does 4×4=16 WMMA f32_16x16x16_f16 ops per inner iter
    // Each WMMA = 16×16×16 = 4096 FMA ops = 8192 FLOPs
    // Total: n_blocks × 32 waves/block × 16 WMMA/wave/inner × n_inner × 8192 FLOPs
    double total_flops = (double)n_blocks * 32.0 * 16.0 * (double)n_inner * 8192.0;
    double tflops = total_flops / (ms * 1e-3) / 1e12;
    printf("  n_blocks=%d n_inner=%d time=%.3f ms\n", n_blocks, n_inner, ms);
    printf("  FP16 WMMA peak: %.2f TFLOPS\n\n", tflops);

    // INT8 WMMA peak
    printf("=== INT8 WMMA Peak Probe ===\n");

    rcpp_peak_probe_launch_i8(Ai8_d, Bi8_d, sink_i32_d, n_blocks, n_inner, stream);
    hipStreamSynchronize(stream);

    hipEventRecord(start, stream);
    rcpp_peak_probe_launch_i8(Ai8_d, Bi8_d, sink_i32_d, n_blocks, n_inner, stream);
    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);

    ms = 0;
    hipEventElapsedTime(&ms, start, stop);

    // INT8 WMMA: 4×4=16 WMMA i32_16x16x16_iu8 per wave per inner iter
    // INT8 ops count: 16×16×16 = 4096 ops per WMMA
    double total_ops = (double)n_blocks * 32.0 * 16.0 * (double)n_inner * 4096.0;
    double tops = total_ops / (ms * 1e-3) / 1e12;
    printf("  n_blocks=%d n_inner=%d time=%.3f ms\n", n_blocks, n_inner, ms);
    printf("  INT8 WMMA peak: %.2f TOPS\n\n", tops);

    // Also run with smaller n_inner to check if there's overhead
    for (int ni : {1000, 5000, 10000, 50000}) {
        hipStreamSynchronize(stream);
        hipEventRecord(start, stream);
        rcpp_peak_probe_launch_i8(Ai8_d, Bi8_d, sink_i32_d, n_blocks, ni, stream);
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);
        ms = 0;
        hipEventElapsedTime(&ms, start, stop);
        total_ops = (double)n_blocks * 32.0 * 16.0 * (double)ni * 4096.0;
        tops = total_ops / (ms * 1e-3) / 1e12;
        printf("  INT8 n_inner=%5d:  time=%.3f ms  TOPS=%.2f\n", ni, ms, tops);
    }

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(A_d); hipFree(B_d); hipFree(sink_d);
    hipFree(Ai8_d); hipFree(Bi8_d); hipFree(sink_i32_d);
    hipStreamDestroy(stream);
    return 0;
}
