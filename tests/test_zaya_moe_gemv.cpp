// Zaya MoE Ternary GEMV — functional correctness test
//
// Build & run:
//   cd ~/1bit/build-rocm
//   cmake --build . --target test_zaya_moe_gemv -j8
//   ./test_zaya_moe_gemv

#include "rocm_cpp/zaya_moe.h"
#include "rocm_cpp/ck_gemm.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cassert>

#define HIP_CHECK(e) do { hipError_t _s = (e); if (_s != hipSuccess) { fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s)); abort(); } } while(0)

// Reference: naive CPU ternary MoE (matches kernel: includes x_scale)
static void ref_zaya_moe(
    const int8_t* weights,       // [num_experts][K]
    const int8_t* x,             // [K]
    float         x_scale,       // activation quantization scale
    const float*  expert_scales, // [num_experts]
    const int*    expert_indices, // [top_k]
    const float*  expert_weights, // [top_k]
    float*        y,             // [num_experts]
    int num_experts, int top_k, int K)
{
    // Zero output
    memset(y, 0, num_experts * sizeof(float));

    for (int e = 0; e < top_k; e++) {
        int idx = expert_indices[e];
        if (idx < 0 || idx >= num_experts) continue;
        float rw = expert_weights[e];
        float s = expert_scales[idx];

        int32_t dot = 0;
        for (int k = 0; k < K; k++) {
            dot += (int32_t)weights[(size_t)idx * K + k] * (int32_t)x[k];
        }
        y[idx] = (float)dot * x_scale * s * rw;
    }
}

// Pack into TQ1 format: 5 ternary values per byte, 4 bytes per u32
// Each u32 encodes 20 ternary values
static void pack_tq1(const int8_t* ternary, uint32_t* packed, int K, int K_padded) {
    int n_u32s = K_padded / 20;
    for (int u = 0; u < n_u32s; u++) {
        uint32_t word = 0;
        for (int b = 0; b < 4; b++) {
            int byte_idx = u * 4 + b;
            int val = 0;
            int mul = 1;
            for (int i = 0; i < 5; i++) {
                int k = byte_idx * 5 + i;
                int d = (k < K) ? (ternary[k] + 1) : 0;
                if (d < 0) d = 0;
                if (d > 2) d = 2;
                val += d * mul;
                mul *= 3;
            }
            word |= ((uint32_t)val) << (b * 8);
        }
        packed[u] = word;
    }
}

int main() {
    HIP_CHECK(hipSetDevice(0));

    // Test parameters — match small Zaya MoE layer
    const int num_experts = 8;
    const int top_k = 2;
    const int K = 2560;        // head_dim * n_heads (e.g., 128 * 20)
    const int tokens = 4;
    const int K_padded = ((K + 19) / 20) * 20;  // round up to mult of 20

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    printf("Zaya MoE Ternary GEMV test\n");
    printf("  num_experts=%d top_k=%d K=%d K_padded=%d tokens=%d\n",
           num_experts, top_k, K, K_padded, tokens);

    const size_t expert_weight_bytes = (size_t)num_experts * K * sizeof(int8_t);
    const int n_u32_per_expert = K_padded / 20;
    const size_t packed_bytes = (size_t)num_experts * n_u32_per_expert * sizeof(uint32_t);

    // Host data
    std::vector<int8_t> h_weights(num_experts * K);
    std::vector<int8_t> h_x(tokens * K);
    std::vector<float>  h_scales(num_experts);
    std::vector<int>    h_indices(tokens * top_k);
    std::vector<float>  h_weights_moe(tokens * top_k);
    std::vector<float>  h_ref(num_experts);

    // Fill with deterministic data
    srand(42);
    for (size_t i = 0; i < h_weights.size(); i++)
        h_weights[i] = (int8_t)((rand() % 3) - 1);  // -1, 0, +1
    for (size_t i = 0; i < h_x.size(); i++)
        h_x[i] = (int8_t)((rand() % 256) - 128);
    for (int e = 0; e < num_experts; e++)
        h_scales[e] = 0.5f + (float)(rand() % 100) / 100.0f;

    // EDA router selects top_k experts per token
    for (int t = 0; t < tokens; t++) {
        // Simple round-robin expert selection
        int used[8] = {0};
        for (int e = 0; e < top_k; e++) {
            int idx;
            do { idx = rand() % num_experts; } while (used[idx]);
            used[idx] = 1;
            h_indices[t * top_k + e] = idx;
            h_weights_moe[t * top_k + e] = 0.5f + (float)(rand() % 100) / 200.0f;
        }
    }

    // Device memory
    uint32_t* d_packed;
    int8_t*   d_x;
    float*    d_scales;
    int32_t*  d_indices;
    float*    d_weights_moe;
    __half*   d_y;

    HIP_CHECK(hipMalloc(&d_packed, packed_bytes));
    HIP_CHECK(hipMalloc(&d_x, tokens * K_padded * sizeof(int8_t)));
    HIP_CHECK(hipMalloc(&d_scales, num_experts * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_indices, tokens * top_k * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_weights_moe, tokens * top_k * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, (size_t)tokens * num_experts * sizeof(__half)));

    // Pack weights into TQ1 format
    std::vector<uint32_t> h_packed(num_experts * n_u32_per_expert, 0);
    for (int e = 0; e < num_experts; e++) {
        pack_tq1(h_weights.data() + (size_t)e * K,
                 h_packed.data() + (size_t)e * n_u32_per_expert, K, K_padded);
    }

    // Pad activation to K_padded
    std::vector<int8_t> h_x_padded(tokens * K_padded, 0);
    for (int t = 0; t < tokens; t++) {
        memcpy(h_x_padded.data() + (size_t)t * K_padded,
               h_x.data() + (size_t)t * K, K * sizeof(int8_t));
    }

    // Copy to device
    HIP_CHECK(hipMemcpy(d_packed, h_packed.data(), packed_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x_padded.data(), tokens * K_padded * sizeof(int8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_scales, h_scales.data(), num_experts * sizeof(float), hipMemcpyHostToDevice));

    std::vector<int32_t> h_indices_i32(tokens * top_k);
    for (int i = 0; i < tokens * top_k; i++)
        h_indices_i32[i] = h_indices[i];
    HIP_CHECK(hipMemcpy(d_indices, h_indices_i32.data(), tokens * top_k * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weights_moe, h_weights_moe.data(), tokens * top_k * sizeof(float), hipMemcpyHostToDevice));

    // Run kernel
    const float x_scale = 1.0f / 127.0f;
    zaya_moe_ternary_gemv_batched_launch(
        d_packed, d_x, x_scale, d_scales,
        d_indices, d_weights_moe, d_y,
        tokens, num_experts, top_k, K_padded, stream);

    HIP_CHECK(hipStreamSynchronize(stream));

    // Read back results
    std::vector<__half> h_y(tokens * num_experts);
    HIP_CHECK(hipMemcpy(h_y.data(), d_y, (size_t)tokens * num_experts * sizeof(__half), hipMemcpyDeviceToHost));

    // Reference computation (token 0 only)
    printf("\nToken 0 results:\n");
    ref_zaya_moe(h_weights.data(), h_x.data(), x_scale, h_scales.data(),
                 h_indices.data(), h_weights_moe.data(),
                 h_ref.data(), num_experts, top_k, K);

    bool pass = true;
    int checked = 0;
    for (int e = 0; e < num_experts; e++) {
        float ref_val = h_ref[e];
        float gpu_val = __half2float(h_y[e]);
        float abs_err = fabsf(ref_val - gpu_val);
        float rel_err = (fabsf(ref_val) > 1e-6f) ? abs_err / fabsf(ref_val) : abs_err;

        if (ref_val != 0.0f || gpu_val != 0.0f) {
            checked++;
            const float tolerance = 30.0f;  // INT8 quantization + TQ1 decode tolerance
            // Note: kernel uses fp16 intermediates, ref uses fp32. Allow 30 ULPs.
            if (abs_err > tolerance) {
                printf("  FAIL expert %d: ref=%.2f gpu=%.2f err=%.4f\n",
                       e, ref_val, gpu_val, abs_err);
                pass = false;
            } else {
                printf("  OK   expert %d: ref=%.2f gpu=%.2f err=%.4f\n",
                       e, ref_val, gpu_val, abs_err);
            }
        }
    }

    // Also check non-selected experts are zero (within fp16 denorm)
    for (int e = 0; e < num_experts; e++) {
        float gpu_val = __half2float(h_y[e]);
        bool selected = false;
        for (int ee = 0; ee < top_k; ee++) {
            if (h_indices[ee] == e) { selected = true; break; }
        }
        if (!selected && fabsf(gpu_val) > 1e-6f) {
            printf("  FAIL expert %d (unselected): non-zero gpu=%.6f\n", e, gpu_val);
            pass = false;
        }
    }

    // Check all tokens
    printf("\nAll tokens:\n");
    for (int t = 0; t < tokens; t++) {
        ref_zaya_moe(h_weights.data(), h_x.data() + (size_t)t * K,
                     x_scale, h_scales.data(),
                     h_indices.data() + (size_t)t * top_k,
                     h_weights_moe.data() + (size_t)t * top_k,
                     h_ref.data(), num_experts, top_k, K);

        for (int e = 0; e < num_experts; e++) {
            float ref_val = h_ref[e];
            float gpu_val = __half2float(h_y[(size_t)t * num_experts + e]);
            float abs_err = fabsf(ref_val - gpu_val);
            if ((ref_val != 0.0f || gpu_val != 0.0f) && abs_err > 1.0f) {
                printf("  FAIL token %d expert %d: ref=%.2f gpu=%.2f err=%.4f\n",
                       t, e, ref_val, gpu_val, abs_err);
                pass = false;
            }
        }
    }

    // Cleanup
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_packed));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_scales));
    HIP_CHECK(hipFree(d_indices));
    HIP_CHECK(hipFree(d_weights_moe));
    HIP_CHECK(hipFree(d_y));

    printf("\n%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
