// bench_bonsai_full.cpp — benchmark full Bonsai-1.7B forward pass with custom
// 1-bit GEMV kernels. Measures end-to-end tok/s across all 28 layers + LM head.
//
// Uses synthetic weights (Q1_0 format) matching Bonsai-1.7B dimensions:
//   hs=2048, is=6144, L=28, nh=16, nkv=8, hd=128, V=151669
//
// Compile: hipcc -O3 -ffast-math -munsafe-fp-atomics \
//              -o bench_bonsai_full bench_bonsai_full.cpp \
//              -L../build -lrocm_cpp \
//              -I../include
// Run:    LD_LIBRARY_PATH=../build ./bench_bonsai_full

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

#include "rocm_cpp/bonsai.h"

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                (int)_s, hipGetErrorString(_s), __FILE__, __LINE__); \
        std::exit(1); \
    } \
} while (0)

// ── Q1_0 block constants ──
constexpr int kBlockSize    = 128;
constexpr int kQ1BlockBytes = 18;   // 2B fp16 d + 16B sign bits

// ── Bonsai-1.7B dimensions ──
constexpr int HS  = 2048;     // hidden size
constexpr int IS  = 6144;     // intermediate size (FFN)
constexpr int NL  = 28;       // layers
constexpr int NH  = 16;       // num heads
constexpr int NKV = 8;        // num kv heads
constexpr int HD  = 128;      // head dim
constexpr int V   = 151669;   // vocab size
constexpr float ROPE_THETA = 1000000.0f;
constexpr float EPS = 1e-6f;

// Forward declare prim kernels
extern "C" void rcpp_rmsnorm_fp32_in_fp16_out(
    const float* x_f32, const __half* w, __half* y, float eps, int N, void* stream);
extern "C" void rcpp_rmsnorm_fp16(
    const __half* x, const __half* w, __half* y, float eps, int N, void* stream);
extern "C" void rcpp_residual_add_fp32_from_fp16(
    float* residual, const __half* update, int N, void* stream);
extern "C" void rcpp_rope_fp16(
    __half* qk, int pos, float theta, int num_heads, int head_dim, void* stream);
extern "C" void rcpp_kv_cache_attn_decode_fd(
    const __half* q, const __half* K, const __half* V,
    __half* out, int nh, int nkv, int hd, int seq_len, float scale, void* stream);
extern "C" void rcpp_silu_glu_fp16(
    const __half* gate, const __half* up, __half* out, int N, void* stream);
// Skipping rcpp_fp16_gemv for now — lm_head benchmarked separately

// ── Q1_0 synthetic weight generator (host) ──
static size_t q1_weight_bytes(int rows, int cols) {
    int blocks_per_row = cols / kBlockSize;
    return (size_t)rows * (size_t)blocks_per_row * (size_t)kQ1BlockBytes;
}

static void fill_q1_weights(uint8_t* dst, int rows, int cols, unsigned seed) {
    int blocks_per_row = cols / kBlockSize;
    size_t row_bytes = (size_t)blocks_per_row * kQ1BlockBytes;
    std::srand(seed);
    for (int r = 0; r < rows; ++r) {
        uint8_t* row = dst + (size_t)r * row_bytes;
        for (int b = 0; b < blocks_per_row; ++b) {
            uint8_t* blk = row + (size_t)b * kQ1BlockBytes;
            // Random fp16 scale in [0.5, 2.0] * 1e-3
            float d = (0.5f + (float)std::rand() / (float)RAND_MAX * 1.5f) * 1e-3f;
            __half d_h = __float2half(d);
            std::memcpy(blk, &d_h, 2);
            // Random sign bits
            for (int i = 0; i < 16; ++i)
                blk[2 + i] = (uint8_t)(std::rand() & 0xFF);
        }
    }
}

// ── Benchmark ──
int main() {
    hipStream_t stream;
    HIP_OK(hipStreamCreate(&stream));

    // ---- Allocate synthetic Q1_0 weights on GPU ----
    printf("Generating synthetic Q1_0 weights for Bonsai-1.7B...\n");

    // Per-layer weight sizes
    auto alloc_q1 = [&](int rows, int cols, const char* name, uint8_t** dev) {
        size_t bytes = q1_weight_bytes(rows, cols);
        std::vector<uint8_t> host(bytes);
        fill_q1_weights(host.data(), rows, cols, (unsigned)(uintptr_t)name);
        HIP_OK(hipMalloc(dev, bytes));
        HIP_OK(hipMemcpyAsync(*dev, host.data(), bytes, hipMemcpyHostToDevice, stream));
        printf("  %-20s %6d x %-6d = %8zu KB\n", name, rows, cols, bytes / 1024);
    };

    uint8_t *d_q[NL], *d_k[NL], *d_v[NL], *d_o[NL];
    uint8_t *d_gate[NL], *d_up[NL], *d_down[NL];

    __half *d_q_norm[NL], *d_k_norm[NL], *d_in_norm[NL], *d_post_norm[NL];

    for (int l = 0; l < NL; ++l) {
        char buf[64];
        snprintf(buf, sizeof(buf), "L%d.Q", l); alloc_q1(NH*HD, HS, buf, &d_q[l]);
        snprintf(buf, sizeof(buf), "L%d.K", l); alloc_q1(NKV*HD, HS, buf, &d_k[l]);
        snprintf(buf, sizeof(buf), "L%d.V", l); alloc_q1(NKV*HD, HS, buf, &d_v[l]);
        snprintf(buf, sizeof(buf), "L%d.O", l); alloc_q1(HS, NH*HD, buf, &d_o[l]);
        snprintf(buf, sizeof(buf), "L%d.gate", l); alloc_q1(IS, HS, buf, &d_gate[l]);
        snprintf(buf, sizeof(buf), "L%d.up", l);   alloc_q1(IS, HS, buf, &d_up[l]);
        snprintf(buf, sizeof(buf), "L%d.down", l); alloc_q1(HS, IS, buf, &d_down[l]);

        // Norm weights (FP16)
        auto alloc_norm = [&](int n, const char* name, __half** dev) {
            std::vector<__half> h(n, __float2half(1.0f));
            HIP_OK(hipMalloc(dev, n * sizeof(__half)));
            HIP_OK(hipMemcpyAsync(*dev, h.data(), n * sizeof(__half),
                                  hipMemcpyHostToDevice, stream));
        };
        alloc_norm(HS, "L*.input_norm", &d_in_norm[l]);
        alloc_norm(HS, "L*.post_norm", &d_post_norm[l]);
        alloc_norm(HD, "L*.q_norm", &d_q_norm[l]);
        alloc_norm(HD, "L*.k_norm", &d_k_norm[l]);
    }

    // Embedding (lm_head uses same weights for tied embedding)
    __half *d_embed = nullptr;
    {
        std::vector<__half> h((size_t)V * HS, __float2half(0.01f));
        HIP_OK(hipMalloc(&d_embed, (size_t)V * HS * sizeof(__half)));
        HIP_OK(hipMemcpyAsync(d_embed, h.data(), (size_t)V * HS * sizeof(__half),
                              hipMemcpyHostToDevice, stream));
    }

    __half *d_final_norm = nullptr;
    {
        std::vector<__half> h(HS, __float2half(1.0f));
        HIP_OK(hipMalloc(&d_final_norm, HS * sizeof(__half)));
        HIP_OK(hipMemcpyAsync(d_final_norm, h.data(), HS * sizeof(__half),
                              hipMemcpyHostToDevice, stream));
    }

    // ---- Allocate scratch buffers ----
    __half *d_x = nullptr, *d_normed = nullptr;
    __half *d_q_f16 = nullptr, *d_k_f16 = nullptr, *d_v_f16 = nullptr;
    __half *d_o_f16 = nullptr, *d_gate_f16 = nullptr, *d_up_f16 = nullptr;
    __half *d_down_f16 = nullptr, *d_silu_out = nullptr;
    float  *d_x_f32 = nullptr, *d_logits = nullptr;

    HIP_OK(hipMalloc(&d_x,        HS * sizeof(__half)));
    HIP_OK(hipMalloc(&d_normed,   HS * sizeof(__half)));
    HIP_OK(hipMalloc(&d_q_f16,    NH * HD * sizeof(__half)));
    HIP_OK(hipMalloc(&d_k_f16,    NKV * HD * sizeof(__half)));
    HIP_OK(hipMalloc(&d_v_f16,    NKV * HD * sizeof(__half)));
    HIP_OK(hipMalloc(&d_o_f16,    HS * sizeof(__half)));
    HIP_OK(hipMalloc(&d_gate_f16, IS * sizeof(__half)));
    HIP_OK(hipMalloc(&d_up_f16,   IS * sizeof(__half)));
    HIP_OK(hipMalloc(&d_down_f16, HS * sizeof(__half)));
    HIP_OK(hipMalloc(&d_silu_out, IS * sizeof(__half)));
    HIP_OK(hipMalloc(&d_x_f32,    HS * sizeof(float)));
    HIP_OK(hipMalloc(&d_logits,   V * sizeof(float)));

    // Initial activations (random fp16)
    {
        std::vector<__half> h(HS);
        for (int i = 0; i < HS; ++i) h[i] = __float2half(0.01f * (float)std::rand() / RAND_MAX);
        HIP_OK(hipMemcpyAsync(d_x, h.data(), HS * sizeof(__half),
                              hipMemcpyHostToDevice, stream));
    }
    HIP_OK(hipMemsetAsync(d_x_f32, 0, HS * sizeof(float), stream));
    HIP_OK(hipDeviceSynchronize());

    // KV cache (one position)
    __half *d_K_cache = nullptr, *d_V_cache = nullptr;
    HIP_OK(hipMalloc(&d_K_cache, (size_t)NKV * HD * sizeof(__half)));
    HIP_OK(hipMalloc(&d_V_cache, (size_t)NKV * HD * sizeof(__half)));

    // ---- Warmup ----
    printf("\nWarmup: 1 forward pass...\n");
    for (int l = 0; l < NL; ++l) {
        // RMSNorm → Q/K/V
        rcpp_rmsnorm_fp32_in_fp16_out(d_x_f32, d_in_norm[l], d_normed, EPS, HS, stream);
        bonsai_q1_gemv_launch(d_q[l], (uint16_t*)d_normed, (uint16_t*)d_q_f16, NH*HD, HS, stream);
        bonsai_q1_gemv_launch(d_k[l], (uint16_t*)d_normed, (uint16_t*)d_k_f16, NKV*HD, HS, stream);
        bonsai_q1_gemv_launch(d_v[l], (uint16_t*)d_normed, (uint16_t*)d_v_f16, NKV*HD, HS, stream);

        // Per-head Q/K norms
        for (int h = 0; h < NH; ++h)
            rcpp_rmsnorm_fp16(d_q_f16 + h * HD, d_q_norm[l], d_q_f16 + h * HD, EPS, HD, stream);
        for (int h = 0; h < NKV; ++h)
            rcpp_rmsnorm_fp16(d_k_f16 + h * HD, d_k_norm[l], d_k_f16 + h * HD, EPS, HD, stream);

        // RoPE
        rcpp_rope_fp16(d_q_f16, l, ROPE_THETA, NH, HD, stream);
        rcpp_rope_fp16(d_k_f16, l, ROPE_THETA, NKV, HD, stream);

        // KV cache store
        HIP_OK(hipMemcpyAsync(d_K_cache, d_k_f16, (size_t)NKV*HD*sizeof(__half),
                              hipMemcpyDeviceToDevice, stream));
        HIP_OK(hipMemcpyAsync(d_V_cache, d_v_f16, (size_t)NKV*HD*sizeof(__half),
                              hipMemcpyDeviceToDevice, stream));

        // Attention
        rcpp_kv_cache_attn_decode_fd(d_q_f16, d_K_cache, d_V_cache,
                                     d_o_f16, NH, NKV, HD, l+1, 1.0f/sqrtf(HD), stream);

        // O proj + residual
        bonsai_q1_gemv_launch(d_o[l], (uint16_t*)d_o_f16, (uint16_t*)d_normed, HS, NH*HD, stream);
        rcpp_residual_add_fp32_from_fp16(d_x_f32, d_normed, HS, stream);

        // FFN
        rcpp_rmsnorm_fp32_in_fp16_out(d_x_f32, d_post_norm[l], d_normed, EPS, HS, stream);
        bonsai_q1_gemv_launch(d_gate[l], (uint16_t*)d_normed, (uint16_t*)d_gate_f16, IS, HS, stream);
        bonsai_q1_gemv_launch(d_up[l], (uint16_t*)d_normed, (uint16_t*)d_up_f16, IS, HS, stream);
        rcpp_silu_glu_fp16(d_gate_f16, d_up_f16, d_silu_out, IS, stream);
        bonsai_q1_gemv_launch(d_down[l], (uint16_t*)d_silu_out, (uint16_t*)d_down_f16, HS, IS, stream);
        rcpp_residual_add_fp32_from_fp16(d_x_f32, d_down_f16, HS, stream);
    }

    // Final norm only (skip lm_head for benchmark)
    rcpp_rmsnorm_fp32_in_fp16_out(d_x_f32, d_final_norm, d_normed, EPS, HS, stream);

    HIP_OK(hipDeviceSynchronize());
    printf("Warmup done.\n");

    // ---- Timed runs ----
    const int N_RUNS = 10;
    printf("\nBenchmarking %d forward passes...\n", N_RUNS);

    hipEvent_t t0, t1;
    HIP_OK(hipEventCreate(&t0));
    HIP_OK(hipEventCreate(&t1));

    HIP_OK(hipEventRecord(t0, stream));
    for (int run = 0; run < N_RUNS; ++run) {
        int pos = run;
        // Re-init residual
        HIP_OK(hipMemsetAsync(d_x_f32, 0, HS * sizeof(float), stream));

        for (int l = 0; l < NL; ++l) {
            rcpp_rmsnorm_fp32_in_fp16_out(d_x_f32, d_in_norm[l], d_normed, EPS, HS, stream);
            bonsai_q1_gemv_launch(d_q[l], (uint16_t*)d_normed, (uint16_t*)d_q_f16, NH*HD, HS, stream);
            bonsai_q1_gemv_launch(d_k[l], (uint16_t*)d_normed, (uint16_t*)d_k_f16, NKV*HD, HS, stream);
            bonsai_q1_gemv_launch(d_v[l], (uint16_t*)d_normed, (uint16_t*)d_v_f16, NKV*HD, HS, stream);

            for (int h = 0; h < NH; ++h)
                rcpp_rmsnorm_fp16(d_q_f16 + h * HD, d_q_norm[l], d_q_f16 + h * HD, EPS, HD, stream);
            for (int h = 0; h < NKV; ++h)
                rcpp_rmsnorm_fp16(d_k_f16 + h * HD, d_k_norm[l], d_k_f16 + h * HD, EPS, HD, stream);

            rcpp_rope_fp16(d_q_f16, pos, ROPE_THETA, NH, HD, stream);
            rcpp_rope_fp16(d_k_f16, pos, ROPE_THETA, NKV, HD, stream);

            HIP_OK(hipMemcpyAsync(d_K_cache, d_k_f16, (size_t)NKV*HD*sizeof(__half),
                                  hipMemcpyDeviceToDevice, stream));
            HIP_OK(hipMemcpyAsync(d_V_cache, d_v_f16, (size_t)NKV*HD*sizeof(__half),
                                  hipMemcpyDeviceToDevice, stream));

            rcpp_kv_cache_attn_decode_fd(d_q_f16, d_K_cache, d_V_cache,
                                         d_o_f16, NH, NKV, HD, pos+1, 1.0f/sqrtf(HD), stream);

            bonsai_q1_gemv_launch(d_o[l], (uint16_t*)d_o_f16, (uint16_t*)d_normed, HS, NH*HD, stream);
            rcpp_residual_add_fp32_from_fp16(d_x_f32, d_normed, HS, stream);

            rcpp_rmsnorm_fp32_in_fp16_out(d_x_f32, d_post_norm[l], d_normed, EPS, HS, stream);
            bonsai_q1_gemv_launch(d_gate[l], (uint16_t*)d_normed, (uint16_t*)d_gate_f16, IS, HS, stream);
            bonsai_q1_gemv_launch(d_up[l], (uint16_t*)d_normed, (uint16_t*)d_up_f16, IS, HS, stream);
            rcpp_silu_glu_fp16(d_gate_f16, d_up_f16, d_silu_out, IS, stream);
            bonsai_q1_gemv_launch(d_down[l], (uint16_t*)d_silu_out, (uint16_t*)d_down_f16, HS, IS, stream);
            rcpp_residual_add_fp32_from_fp16(d_x_f32, d_down_f16, HS, stream);
        }

        rcpp_rmsnorm_fp32_in_fp16_out(d_x_f32, d_final_norm, d_normed, EPS, HS, stream);
    }
    HIP_OK(hipEventRecord(t1, stream));
    HIP_OK(hipEventSynchronize(t1));

    float ms_total;
    HIP_OK(hipEventElapsedTime(&ms_total, t0, t1));
    double ms_per_token = ms_total / (double)N_RUNS;
    double tok_per_sec = 1000.0 / ms_per_token;

    printf("\n═══════════════════════════════════════════\n");
    printf("  Runs:           %d\n", N_RUNS);
    printf("  Total time:     %.1f ms\n", ms_total);
    printf("  Per token:      %.3f ms\n", ms_per_token);
    printf("  Throughput:     %.1f tok/s\n", tok_per_sec);
    printf("═══════════════════════════════════════════\n");

    // Cleanup
    for (int l = 0; l < NL; ++l) {
        hipFree(d_q[l]); hipFree(d_k[l]); hipFree(d_v[l]); hipFree(d_o[l]);
        hipFree(d_gate[l]); hipFree(d_up[l]); hipFree(d_down[l]);
        hipFree(d_in_norm[l]); hipFree(d_post_norm[l]);
        hipFree(d_q_norm[l]); hipFree(d_k_norm[l]);
    }
    hipFree(d_embed); hipFree(d_final_norm);
    hipFree(d_x); hipFree(d_normed);
    hipFree(d_q_f16); hipFree(d_k_f16); hipFree(d_v_f16);
    hipFree(d_o_f16); hipFree(d_gate_f16); hipFree(d_up_f16);
    hipFree(d_down_f16); hipFree(d_silu_out);
    hipFree(d_x_f32); hipFree(d_logits);
    hipFree(d_K_cache); hipFree(d_V_cache);
    HIP_OK(hipEventDestroy(t0));
    HIP_OK(hipEventDestroy(t1));
    HIP_OK(hipStreamDestroy(stream));

    return 0;
}
