#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP %d %s:%d\n", _s, __FILE__, __LINE__); std::abort(); } } while(0)

extern "C" {
void rcpp_peak_probe_launch(const void*, const void*, void*, int, int, void*);
void rcpp_peak_probe_launch_i8(const void*, const void*, void*, int, int, void*);
}

int main() {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    int n_blocks = 40 * 32;
    int n_inner = 10000;
    size_t seed_bytes = 32 * 4 * 16 * sizeof(__half);
    size_t sink_fp32_bytes = n_blocks * 32 * sizeof(float);
    size_t sink_i32_bytes = n_blocks * 32 * sizeof(int32_t);
    __half *A_d, *B_d; float *sink_d;
    int8_t *Ai8_d, *Bi8_d; int32_t *sink_i32_d;
    HIP_CHECK(hipMalloc(&A_d, seed_bytes));
    HIP_CHECK(hipMalloc(&B_d, seed_bytes));
    HIP_CHECK(hipMalloc(&sink_d, sink_fp32_bytes));
    HIP_CHECK(hipMalloc(&Ai8_d, seed_bytes));
    HIP_CHECK(hipMalloc(&Bi8_d, seed_bytes));
    HIP_CHECK(hipMalloc(&sink_i32_d, sink_i32_bytes));

    printf("=== FP16 WMMA Peak Probe ===\n");
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    rcpp_peak_probe_launch(A_d, B_d, sink_d, n_blocks, n_inner, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipEventRecord(start, stream));
    rcpp_peak_probe_launch(A_d, B_d, sink_d, n_blocks, n_inner, stream);
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    double total_flops = (double)n_blocks * 32.0 * 16.0 * (double)n_inner * 8192.0;
    printf("  n_blocks=%d n_inner=%d time=%.3f ms  FP16 WMMA: %.2f TFLOPS\n\n", n_blocks, n_inner, ms, total_flops / (ms * 1e-3) / 1e12);

    printf("=== INT8 WMMA Peak Probe ===\n");
    rcpp_peak_probe_launch_i8(Ai8_d, Bi8_d, sink_i32_d, n_blocks, n_inner, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipEventRecord(start, stream));
    rcpp_peak_probe_launch_i8(Ai8_d, Bi8_d, sink_i32_d, n_blocks, n_inner, stream);
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    double total_ops = (double)n_blocks * 32.0 * 16.0 * (double)n_inner * 4096.0;
    printf("  n_blocks=%d n_inner=%d time=%.3f ms  INT8 WMMA: %.2f TOPS\n\n", n_blocks, n_inner, ms, total_ops / (ms * 1e-3) / 1e12);

    for (int ni : {1000, 5000, 10000, 50000}) {
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipEventRecord(start, stream));
        rcpp_peak_probe_launch_i8(Ai8_d, Bi8_d, sink_i32_d, n_blocks, ni, stream);
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));
        ms = 0;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        total_ops = (double)n_blocks * 32.0 * 16.0 * (double)ni * 4096.0;
        printf("  INT8 n_inner=%5d: time=%.3f ms  TOPS=%.2f\n", ni, ms, total_ops / (ms * 1e-3) / 1e12);
    }

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(A_d)); HIP_CHECK(hipFree(B_d)); HIP_CHECK(hipFree(sink_d));
    HIP_CHECK(hipFree(Ai8_d)); HIP_CHECK(hipFree(Bi8_d)); HIP_CHECK(hipFree(sink_i32_d));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}