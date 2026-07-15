// ROCm C++ GEMM Benchmark — Native Tensile vs System rocBLAS
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

void check_hip(hipError_t e, const char* file, int line) {
    if (e != hipSuccess) { fprintf(stderr, "HIP error %d at %s:%d\n", e, file, line); exit(1); }
}
void check_rb(rocblas_status s, const char* file, int line) {
    if (s != rocblas_status_success) { fprintf(stderr, "rocBLAS error %d at %s:%d\n", s, file, line); exit(1); }
}
#define HC(cmd) check_hip(cmd, __FILE__, __LINE__)
#define RC(cmd) check_rb(cmd, __FILE__, __LINE__)

void bench_gemm(rocblas_handle handle, int M, int N, int K, int warmup, int runs) {
    size_t sa = (size_t)M * K * sizeof(rocblas_half);
    size_t sb = (size_t)K * N * sizeof(rocblas_half);
    size_t sc = (size_t)M * N * sizeof(rocblas_half);

    rocblas_half *da, *db, *dc;
    HC(hipMalloc(&da, sa));
    HC(hipMalloc(&db, sb));
    HC(hipMalloc(&dc, sc));
    HC(hipMemset(da, 0x3C, sa));  // ~1.0 in fp16
    HC(hipMemset(db, 0x3C, sb));

    rocblas_half alpha, beta;
    *reinterpret_cast<uint16_t*>(&alpha) = 0x3C00; // 1.0
    *reinterpret_cast<uint16_t*>(&beta)  = 0x0000; // 0.0

    for (int i = 0; i < warmup; i++) {
        RC(rocblas_hgemm(handle, rocblas_operation_none, rocblas_operation_none,
            M, N, K, &alpha, da, M, db, K, &beta, dc, M));
    }
    HC(hipDeviceSynchronize());

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; i++) {
        RC(rocblas_hgemm(handle, rocblas_operation_none, rocblas_operation_none,
            M, N, K, &alpha, da, M, db, K, &beta, dc, M));
    }
    HC(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg = ms / runs;
    double tflops = (2.0 * M * N * K) / (avg / 1000.0) / 1e12;

    printf("  %5d x %5d x %5d : %8.3f ms  %6.2f TFLOPS\n", M, N, K, avg, tflops);

    HC(hipFree(da)); HC(hipFree(db)); HC(hipFree(dc));
}

int main() {
    printf("=== ROCm C++ GEMM Benchmark — gfx1151 ===\n");

    hipDeviceProp_t p;
    HC(hipGetDeviceProperties(&p, 0));
    printf("GPU: %s (%s), CUs: %d\n", p.name, p.gcnArchName, p.multiProcessorCount);

    const char* tp = getenv("ROCBLAS_TENSILE_LIBPATH");
    printf("Tensile: %s\n\n", tp ? tp : "system default");

    rocblas_handle h;
    RC(rocblas_create_handle(&h));

    int w = 5, r = 20;
    printf("FP16 hgemm — %d warmup, %d runs\n\n", w, r);

    printf("Attention-like:\n");
    bench_gemm(h, 128, 128, 128, w, r);
    bench_gemm(h, 256, 256, 256, w, r);
    bench_gemm(h, 512, 512, 512, w, r);

    printf("\nLinear layers:\n");
    bench_gemm(h, 1024, 1024, 1024, w, r);
    bench_gemm(h, 2048, 2048, 2048, w, r);
    bench_gemm(h, 2560, 2560, 2560, w, r);

    printf("\nFeed-forward:\n");
    bench_gemm(h, 4096, 4096, 4096, w, r);
    bench_gemm(h, 2560, 6912, 2560, w, r);
    bench_gemm(h, 4096, 11008, 4096, w, r);

    printf("\nDecode GEMV (batch=1):\n");
    bench_gemm(h, 1, 2560, 2560, w, r);
    bench_gemm(h, 1, 4096, 4096, w, r);
    bench_gemm(h, 1, 4096, 11008, w, r);

    printf("\nPrefill (batch=512):\n");
    bench_gemm(h, 512, 2560, 2560, w, r);
    bench_gemm(h, 512, 4096, 4096, w, r);

    RC(rocblas_destroy_handle(h));
    printf("\n=== Done ===\n");
    return 0;
}
