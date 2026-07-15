// bench_prefill_variants -- benchmark all ternary prefill GEMM kernels
#include "rocm_cpp/prefill_tuner.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

#define HIP_CHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(_e)); exit(1); } } while(0)

using launch_fn = void (*)(const void*, const void*, void*, int, int, int, void*);

extern "C" {
void rcpp_standalone_launch_wmma_4x4_vec(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_4x4_cached(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_lds_pingpong(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_2x2wave(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_fp16b(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_2x4(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_4x4(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_4x4_vec_i8_tiled(const void*, const void*, void*, int, int, int, void*);
void rcpp_standalone_launch_wmma_4x4_vec_i8_apre_tiled(const void*, const void*, void*, int, int, int, void*);
}

double bench_one(launch_fn fn, const void* A, const void* B, void* C,
                 int M, int N, int K, int warmup, int timed, hipStream_t stream) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    for (int i = 0; i < warmup; ++i) fn(A, B, C, M, N, K, stream);
    HIP_CHECK(hipEventRecord(start, stream));
    for (int i = 0; i < timed; ++i) fn(A, B, C, M, N, K, stream);
    HIP_CHECK(hipEventRecord(stop, stream));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return (double)ms / timed;
}

int main(int argc, char** argv) {
    int M = 2560, N = 6912, K = 2560, warmup = 5, timed = 20;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    if (argc >= 5) warmup = atoi(argv[4]);
    if (argc >= 6) timed = atoi(argv[5]);
    printf("=== Prefill Variant Benchmark (%d x %d x %d) ===\n", M, N, K);
    printf("Warmup: %d, Timed: %d\n\n", warmup, timed);
    // ... rest of benchmark logic unchanged
    printf("\n");
    // Find winner
    double best = 1e30; int best_i = -1;
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].valid) continue;
        if (results[i].ms < best) { best = results[i].ms; best_i = i; }
    }
    if (best_i >= 0)
        printf("--- Winner ---\nFastest variant: %d = %s (%.2f TFLOPS)\n", best_i, results[best_i].name, 2.0*M*N*K/(best*1e-3)/1e12);
    return 0;
}