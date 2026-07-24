// bench_ternary_new.cpp — Benchmark new ternary/binary GPU kernels
// Build: hipcc -O3 -o bench_ternary_new bench_ternary_new.cpp -I../include
// Run:   ./bench_ternary_new

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "../include/rocm_cpp/ck_gemm.h"

#define HIP_CHECK(e) do { auto s=e; if(s!=hipSuccess){fprintf(stderr,"HIP err %d\n",s); exit(1);}} while(0)

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Generate synthetic ternary weights
static void gen_tq2(uint8_t* w, int M, int K) {
    for (int i = 0; i < M * K / 4; i++) {
        uint8_t byte = 0;
        for (int j = 0; j < 4; j++) {
            int code = rand() % 3;  // 0=-1, 1=0, 2=+1
            byte |= (code << (j * 2));
        }
        w[i] = byte;
    }
}

static void gen_binary_q1(uint8_t* w, int M, int K) {
    int n_blocks = (K + 127) / 128;
    int row_bytes = n_blocks * 18;
    for (int r = 0; r < M; r++) {
        // fp16 scale
        __half scale = __float2half(1.0f);
        memcpy(w + r * row_bytes, &scale, 2);
        // sign bits
        for (int b = 0; b < n_blocks; b++) {
            for (int i = 0; i < 16; i++) {
                w[r * row_bytes + 2 + b * 16 + i] = rand() & 0xFF;
            }
        }
    }
}

static void gen_bitnet_tq2_0(uint8_t* w, int M, int K) {
    int n_blocks = (K + 255) / 256;
    int row_bytes = n_blocks * 66;
    for (int r = 0; r < M; r++) {
        for (int b = 0; b < n_blocks; b++) {
            for (int i = 0; i < 64; i++) w[r * row_bytes + b * 66 + i] = rand() & 0xFF;
            __half scale = __float2half(1.0f);
            memcpy(w + r * row_bytes + b * 66 + 64, &scale, 2);
        }
    }
}

template<typename F>
void bench(const char* name, F fn, int M, int K, int iters) {
    double t0 = now_ms();
    for (int i = 0; i < iters; i++) fn();
    double ms = (now_ms() - t0) / iters;
    double gbs = (double)M * K * 2 / (ms * 1e6);  // bytes read (weights) per sec
    printf("  %-30s M=%-5d K=%-6d %8.2f µs  %8.1f GB/s\n",
           name, M, K, ms * 1000.0, gbs);
}

int main() {
    printf("=== Ternary/Binary GPU Kernel Benchmarks ===\n\n");

    int M = 6912, K = 2560, iters = 256;

    // Allocate host memory
    size_t tq2_bytes = (size_t)M * K / 4;
    size_t q1_bytes = (size_t)M * ((K + 127) / 128) * 18;
    size_t bt_bytes = (size_t)M * ((K + 255) / 256) * 66;
    size_t act_bytes = (size_t)M * K;
    size_t out_bytes = (size_t)M * sizeof(__half);
    size_t scale_bytes = (size_t)M * sizeof(float);

    uint8_t *h_tq2, *h_q1, *h_bt, *h_act, *h_out;
    float *h_scales;
    hipHostMalloc(&h_tq2, tq2_bytes); gen_tq2(h_tq2, M, K);
    hipHostMalloc(&h_q1, q1_bytes); gen_binary_q1(h_q1, M, K);
    hipHostMalloc(&h_bt, bt_bytes); gen_bitnet_tq2_0(h_bt, M, K);
    hipHostMalloc(&h_act, act_bytes);
    hipHostMalloc(&h_out, out_bytes);
    hipHostMalloc(&h_scales, scale_bytes);
    for (int i = 0; i < M * K; i++) h_act[i] = 1;  // dummy activations
    for (int i = 0; i < M; i++) h_scales[i] = 1.0f;

    uint8_t *d_tq2, *d_q1, *d_bt;
    int8_t *d_act;
    __half *d_out;
    float *d_scales;
    HIP_CHECK(hipMalloc(&d_tq2, tq2_bytes));
    HIP_CHECK(hipMalloc(&d_q1, q1_bytes));
    HIP_CHECK(hipMalloc(&d_bt, bt_bytes));
    HIP_CHECK(hipMalloc(&d_act, act_bytes));
    HIP_CHECK(hipMalloc(&d_out, out_bytes));
    HIP_CHECK(hipMalloc(&d_scales, scale_bytes));

    HIP_CHECK(hipMemcpy(d_tq2, h_tq2, tq2_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_q1, h_q1, q1_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bt, h_bt, bt_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_act, h_act, act_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_out, h_out, out_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_scales, h_scales, scale_bytes, hipMemcpyHostToDevice));

    printf("Config: M=%d K=%d iters=%d\n\n", M, K, iters);

    // Q1_0 binary (1-bit)
    bench("Q1_0 binary (1-bit)", [&]() {
        rcpp_ternary_gemv_q1_0_f16(d_q1, d_act, 1.0f, d_scales, d_out, M, K, nullptr);
    }, M, K, iters);

    // BitNet TQ2_0 (GGUF native, 2.06 bpw)
    bench("BitNet TQ2_0 (2.06 bpw)", [&]() {
        rcpp_bitnet_gemv_tq2_0_f16(d_bt, d_act, 1.0f, d_scales, d_out, M, K, nullptr);
    }, M, K, iters);

    // Existing TQ1 for comparison
    int K_padded = ((K + 19) / 20) * 20;
    bench("TQ1 (1.58-bit, reference)", [&]() {
        rcpp_ternary_gemv_tq1_halo_f16(d_tq2, d_act, 1.0f, d_scales, d_out, M, K_padded, nullptr);
    }, M, K, iters);

    printf("\n=== Done ===\n");

    hipFree(d_tq2); hipFree(d_q1); hipFree(d_bt);
    hipFree(d_act); hipFree(d_out); hipFree(d_scales);
    hipHostFree(h_tq2); hipHostFree(h_q1); hipHostFree(h_bt);
    hipHostFree(h_act); hipHostFree(h_out); hipHostFree(h_scales);
    return 0;
}
