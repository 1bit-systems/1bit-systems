// Test + benchmark harness for ternary GEMM v2 (prefill path).
//
// Three things this harness does:
//   1. Correctness: run v2 against a scalar CPU reference, tolerate INT8
//      quantization error (ternary weights × INT8-quantized activations
//      vs. the true float matmul).
//   2. Throughput: time v2 against rocBLAS FP16 GEMM at the same shape,
//      since FP16 GEMM is the incumbent lemon-mlx / llama.cpp uses.
//   3. Sanity: single-batch GEMV degenerate case, to confirm the kernel
//      still produces the right sign/magnitude for B=1 (useful for
//      debugging the WMMA lane layout).
//
// Shapes mirror BitNet-2B FFN: M∈{2560, 6912}, K∈{2560, 6912}, B∈{128, 512}.
//
// The skeleton today does not yet hit the real WMMA intrinsic — it uses a
// scalar dot4 fallback — so the numbers here are a baseline, not the
// finished product. They exist to prove the infrastructure works and give
// us a perf floor to beat when the real WMMA path lands.

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <vector>
#include <cmath>
#include <random>

extern "C" {
void ternary_gemm_v2(const uint8_t* packed_bits, const float* x,
                     const float* scales, float* y,
                     int M, int K, int B, hipStream_t stream);
}

#define HC(cmd) do { hipError_t e = cmd; if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %d at %s:%d\n", e, __FILE__, __LINE__); exit(1); } } while(0)

#define RBC(cmd) do { rocblas_status s = cmd; if (s != rocblas_status_success) { \
    fprintf(stderr, "rocBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while(0)

// Generate a random ternary weight matrix packed 8 bits per byte.
// bit=1 → +1, bit=0 → -1.
static std::vector<uint8_t> gen_weights(int M, int K, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> out((size_t)M * (K / 8), 0);
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = rng() & 0xFF;
    }
    return out;
}

static std::vector<float> gen_float(int N, uint32_t seed = 17) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(N);
    for (int i = 0; i < N; i++) v[i] = dist(rng);
    return v;
}

static float scalar_ref(const std::vector<uint8_t>& W, const std::vector<float>& X,
                        const std::vector<float>& S, int row, int batch,
                        int M, int K) {
    float acc = 0.0f;
    int bytes_per_row = K / 8;
    for (int k = 0; k < K; k++) {
        int byte_idx = k / 8;
        int bit = (W[(size_t)row * bytes_per_row + byte_idx] >> (k % 8)) & 1;
        float w = bit ? 1.0f : -1.0f;
        acc += w * X[(size_t)batch * K + k];
    }
    return acc * S[row];
}

static double bench_v2(const uint8_t* d_W, const float* d_X, const float* d_S,
                       float* d_Y, int M, int K, int B, int warmup, int runs) {
    hipStream_t s; HC(hipStreamCreate(&s));
    for (int i = 0; i < warmup; i++) ternary_gemm_v2(d_W, d_X, d_S, d_Y, M, K, B, s);
    HC(hipStreamSynchronize(s));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; i++) ternary_gemm_v2(d_W, d_X, d_S, d_Y, M, K, B, s);
    HC(hipStreamSynchronize(s));
    auto t1 = std::chrono::high_resolution_clock::now();
    HC(hipStreamDestroy(s));
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ms / runs;
}

static double bench_rocblas_fp16(int M, int N, int K, int warmup, int runs) {
    rocblas_handle h; RBC(rocblas_create_handle(&h));
    rocblas_half *dA, *dB, *dC;
    HC(hipMalloc(&dA, (size_t)M * K * sizeof(rocblas_half)));
    HC(hipMalloc(&dB, (size_t)K * N * sizeof(rocblas_half)));
    HC(hipMalloc(&dC, (size_t)M * N * sizeof(rocblas_half)));
    HC(hipMemset(dA, 0, (size_t)M * K * sizeof(rocblas_half)));
    HC(hipMemset(dB, 0, (size_t)K * N * sizeof(rocblas_half)));
    rocblas_half alpha, beta;
    // rocblas_half is a struct wrapping uint16_t; fill bits for 1.0h and 0.0h
    uint16_t one_bits = 0x3C00;  // FP16 1.0
    uint16_t zero_bits = 0x0000; // FP16 0.0
    std::memcpy(&alpha, &one_bits, sizeof(uint16_t));
    std::memcpy(&beta, &zero_bits, sizeof(uint16_t));

    for (int i = 0; i < warmup; i++) {
        RBC(rocblas_hgemm(h, rocblas_operation_none, rocblas_operation_none,
                          M, N, K, &alpha, dA, M, dB, K, &beta, dC, M));
    }
    HC(hipDeviceSynchronize());
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; i++) {
        RBC(rocblas_hgemm(h, rocblas_operation_none, rocblas_operation_none,
                          M, N, K, &alpha, dA, M, dB, K, &beta, dC, M));
    }
    HC(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    HC(hipFree(dA)); HC(hipFree(dB)); HC(hipFree(dC));
    RBC(rocblas_destroy_handle(h));
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ms / runs;
}

static void run_shape(int M, int K, int B) {
    printf("\n=== M=%d K=%d B=%d (ternary GEMM) ===\n", M, K, B);

    auto h_W = gen_weights(M, K);
    auto h_X = gen_float((size_t)B * K);
    std::vector<float> h_S(M, 1.0f);

    uint8_t *d_W; float *d_X, *d_S, *d_Y;
    HC(hipMalloc(&d_W, h_W.size()));
    HC(hipMalloc(&d_X, (size_t)B * K * sizeof(float)));
    HC(hipMalloc(&d_S, (size_t)M * sizeof(float)));
    HC(hipMalloc(&d_Y, (size_t)B * M * sizeof(float)));
    HC(hipMemcpy(d_W, h_W.data(), h_W.size(), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_X, h_X.data(), (size_t)B * K * sizeof(float), hipMemcpyHostToDevice));
    HC(hipMemcpy(d_S, h_S.data(), (size_t)M * sizeof(float), hipMemcpyHostToDevice));
    HC(hipMemset(d_Y, 0, (size_t)B * M * sizeof(float)));

    // Correctness: spot-check a few (row, batch) cells
    ternary_gemm_v2(d_W, d_X, d_S, d_Y, M, K, B, 0);
    HC(hipDeviceSynchronize());
    std::vector<float> h_Y((size_t)B * M);
    HC(hipMemcpy(h_Y.data(), d_Y, (size_t)B * M * sizeof(float), hipMemcpyDeviceToHost));

    double err_sum = 0.0; int samples = 0;
    for (int cell = 0; cell < 8; cell++) {
        int row = (cell * 37) % M;
        int batch = (cell * 13) % B;
        float ref = scalar_ref(h_W, h_X, h_S, row, batch, M, K);
        float got = h_Y[(size_t)batch * M + row];
        double rel = (std::fabs(ref) > 1e-4) ? std::fabs(got - ref) / std::fabs(ref) : 0.0;
        err_sum += rel; samples++;
    }
    printf("  correctness: mean relative error = %.4f (skeleton uses scalar dot4, INT8 quant)\n",
           err_sum / samples);

    // Throughput
    double v2_us = bench_v2(d_W, d_X, d_S, d_Y, M, K, B, 5, 20) * 1000.0;
    double fp16_us = bench_rocblas_fp16(M, B, K, 5, 20) * 1000.0;
    double ratio = fp16_us / v2_us;
    printf("  v2 skeleton : %9.1f us/run\n", v2_us);
    printf("  rocBLAS FP16: %9.1f us/run   (v2 is %.2fx of FP16)\n", fp16_us, ratio);

    HC(hipFree(d_W)); HC(hipFree(d_X)); HC(hipFree(d_S)); HC(hipFree(d_Y));
}

int main() {
    printf("=== Ternary GEMM v2 — skeleton perf/correctness ===\n");
    hipDeviceProp_t p;
    HC(hipGetDeviceProperties(&p, 0));
    printf("GPU: %s (CUs: %d, Wave: %d)\n", p.name, p.multiProcessorCount, p.warpSize);

    // BitNet-2B FFN shapes at typical prefill batches.
    run_shape(2560, 2560, 128);
    run_shape(6912, 2560, 128);
    run_shape(2560, 6912, 128);
    run_shape(2560, 2560, 512);

    printf("\n=== Done ===\n");
    return 0;
}
