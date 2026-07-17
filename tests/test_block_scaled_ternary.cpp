// test_block_scaled_ternary.cpp — functional test for block-scaled ternary GEMV.
// Compares GPU kernel output against CPU reference for MxK matrices.
// Tests K aligned to 16 (normal case) and K not aligned (partial tail).

#include "block_scaled_ternary.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP ERR %d at %s:%d\n", _s, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// CPU reference: ternary dot product with per-block FP8 scaling
extern "C" float ref_bst_gemv(const uint8_t* packed, const int8_t* x, int K) {
    int nb = (K + BST_BLOCK_K - 1) / BST_BLOCK_K;
    double acc = 0.0;
    for (int b = 0; b < nb; ++b) {
        uint32_t w;
        std::memcpy(&w, packed + b * BST_BLOCK_BYTES, 4);
        float scale = fp8e4m3_to_fp32(packed[b * BST_BLOCK_BYTES + 4]);
        int32_t block_acc = 0;
        for (int v = 0; v < BST_BLOCK_K; ++v) {
            int idx = b * BST_BLOCK_K + v;
            int xv = (idx < K) ? (int)x[idx] : 0;
            uint32_t bits = (w >> (v * 2)) & 3;
            int sign = (int)(bits == 1) - (int)(bits == 2);
            block_acc += sign * xv;
        }
        acc += (double)block_acc * (double)scale;
    }
    return (float)acc;
}

extern "C" void ternary_gemv_block_scaled_launch(
    const void* packed, const void* x_i8, float x_scale,
    void* y, int M, int K, void* stream);

static int run_test(int M, int K, const char* label) {
    printf("Test: %s (M=%d K=%d, K%%16=%d)\n", label, M, K, K % 16);
    int nb = (K + BST_BLOCK_K - 1) / BST_BLOCK_K;

    // Generate random weights and activations
    std::vector<float> wt(M * K);
    std::vector<int8_t> act(K);
    for (int i = 0; i < M * K; ++i)
        wt[i] = ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < K; ++i)
        act[i] = (int8_t)(rand() % 256 - 128);
    float x_scale = 0.1f;

    // Pack to block-scaled ternary
    std::vector<uint8_t> packed(M * nb * BST_BLOCK_BYTES);
    for (int r = 0; r < M; ++r)
        block_scaled_ternary_pack_row(
            wt.data() + r * K, packed.data() + r * nb * BST_BLOCK_BYTES, K);

    // CPU reference
    std::vector<float> ref_y(M);
    for (int r = 0; r < M; ++r)
        ref_y[r] = ref_bst_gemv(packed.data() + r * nb * BST_BLOCK_BYTES, act.data(), K) * x_scale;

    // GPU kernel
    uint8_t* d_packed;
    int8_t* d_act;
    float* d_y;
    HIP_OK(hipMalloc(&d_packed, packed.size()));
    HIP_OK(hipMalloc(&d_act, K));
    HIP_OK(hipMalloc(&d_y, M * sizeof(float)));
    HIP_OK(hipMemcpy(d_packed, packed.data(), packed.size(), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(d_act, act.data(), K, hipMemcpyHostToDevice));

    ternary_gemv_block_scaled_launch(d_packed, d_act, x_scale, d_y, M, K, nullptr);

    std::vector<float> gpu_y(M);
    HIP_OK(hipMemcpy(gpu_y.data(), d_y, M * sizeof(float), hipMemcpyDeviceToHost));

    HIP_OK(hipFree(d_packed));
    HIP_OK(hipFree(d_act));
    HIP_OK(hipFree(d_y));

    // Compare
    int errors = 0;
    for (int r = 0; r < M; ++r) {
        float diff = fabsf(gpu_y[r] - ref_y[r]);
        if (diff > 0.001f) {
            fprintf(stderr, "  row %d: ref=%.2f gpu=%.2f diff=%.4f\n", r, ref_y[r], gpu_y[r], diff);
            errors++;
        }
    }
    printf("  %s (%d/%d rows OK)\n", errors ? "FAIL" : "PASS", M - errors, M);
    return errors;
}

int main() {
    int total = 0;
    total += run_test(4, 1024, "aligned K");
    total += run_test(4, 520, "unaligned K");
    total += run_test(1, 128, "single row");
    total += run_test(8, 4096, "large MxK");
    printf("\n%s\n", total ? "FAIL" : "PASS");
    return total ? 1 : 0;
}
