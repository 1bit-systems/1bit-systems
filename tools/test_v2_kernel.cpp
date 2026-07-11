// Quick test for v2 ternary GEMV kernel
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cmath>

extern "C" {
void ternary_gemv(const uint32_t* packed, const float* x, const float* scales,
                  float* y, int M, int K, hipStream_t stream);
void ternary_gemv_v2(const uint32_t* packed_bits, const float* x, const float* scales,
                     float* y, int M, int K, hipStream_t stream);
}

#define HC(cmd) do { hipError_t e = cmd; if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %d at %s:%d\n", e, __FILE__, __LINE__); exit(1); } } while(0)

void bench(const char* name, void(*fn)(const uint32_t*,const float*,const float*,float*,int,int,hipStream_t),
           int M, int K, int warmup, int runs) {
    int packed_k = K / 16;  // v1 uses 2-bit packing (16 per u32)

    uint32_t *d_packed; float *d_x, *d_scales, *d_y;
    HC(hipMalloc(&d_packed, M * packed_k * sizeof(uint32_t)));
    HC(hipMalloc(&d_x, K * sizeof(float)));
    HC(hipMalloc(&d_scales, M * sizeof(float)));
    HC(hipMalloc(&d_y, M * sizeof(float)));
    HC(hipMemset(d_packed, 0x55, M * packed_k * sizeof(uint32_t)));  // all +1
    HC(hipMemset(d_x, 0x3f, K * sizeof(float)));  // ~1.0

    // Set scales to 1.0
    std::vector<float> h_scales(M, 1.0f);
    HC(hipMemcpy(d_scales, h_scales.data(), M * sizeof(float), hipMemcpyHostToDevice));

    hipStream_t stream;
    HC(hipStreamCreate(&stream));

    for (int i = 0; i < warmup; i++) fn(d_packed, d_x, d_scales, d_y, M, K, stream);
    HC(hipStreamSynchronize(stream));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; i++) fn(d_packed, d_x, d_scales, d_y, M, K, stream);
    HC(hipStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double avg_us = (ms / runs) * 1000.0;
    printf("  %s %4dx%4d : %7.1f us\n", name, M, K, avg_us);

    HC(hipFree(d_packed)); HC(hipFree(d_x)); HC(hipFree(d_scales)); HC(hipFree(d_y));
    HC(hipStreamDestroy(stream));
}

int main() {
    printf("=== Ternary GEMV v1 vs v2 — gfx1151 ===\n\n");

    hipDeviceProp_t p;
    HC(hipGetDeviceProperties(&p, 0));
    printf("GPU: %s, CUs: %d\n\n", p.name, p.multiProcessorCount);

    int w = 10, r = 100;

    printf("BitNet-2B shapes:\n");
    bench("v1", ternary_gemv, 2560, 2560, w, r);
    bench("v2", ternary_gemv_v2, 2560, 2560, w, r);
    printf("\n");
    bench("v1", ternary_gemv, 6912, 2560, w, r);
    bench("v2", ternary_gemv_v2, 6912, 2560, w, r);
    printf("\n");
    bench("v1", ternary_gemv, 2560, 6912, w, r);
    bench("v2", ternary_gemv_v2, 2560, 6912, w, r);

    printf("\n=== Done ===\n");
    return 0;
}
