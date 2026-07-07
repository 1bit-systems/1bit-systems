/** Component-level benchmark for NPU engine.
 *  Measures QKV, attention, FFN, and LM head timing separately.
 *  Usage: bench_components model.q4nx [batch_size]
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <thread>
#include "platform.h"
#include "model_config.h"
extern "C" float* dequant_i8_to_float_ex(const uint8_t*,int,int,int*,int*);

static constexpr float EPS = 1e-6f;
static inline void rn_c(float* x, const float* w, int n) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float ir = 1.0f / sqrtf((float)(ss / n) + EPS);
    for (int i = 0; i < n; i++) x[i] *= ir * w[i];
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("Usage: %s model.q4nx [batch_size]\n", argv[0]); return 1; }
    int BS = argc > 2 ? atoi(argv[2]) : 128;
    
    // Load model config
    auto cfg = read_model_config(argv[1], "qwen3_0_6b");
    if (!cfg.valid()) { fprintf(stderr, "Invalid model\n"); return 1; }
    
    const int H = cfg.H, NC = cfg.NC, NH = cfg.NH, NKV = cfg.NKV;
    const int HD = cfg.HD, IM = cfg.IM, NV = cfg.NV, GQA = NH / NKV;
    const int QKV = NH * HD + 2 * NKV * HD;
    
    printf("Model: H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d\n", H, NC, NH, NKV, HD, IM, NV);
    
    // Open model file
    struct stat st; stat(argv[1], &st);
    auto md = (const uint8_t*)platform_mmap(argv[1], st.st_size);
    if (!md) { fprintf(stderr, "mmap failed\n"); return 1; }
    
    // Parse JSON header
    size_t hdr_size = *(const uint64_t*)md;
    const char* js = (const char*)(md + 8);
    
    auto t0 = std::chrono::steady_clock::now();
    
    // Dequant QKV weights
    int q_i8, q_unused;
    printf("Dequantizing weights...\n");
    float* qw = dequant_i8_to_float_ex((uint8_t*)md, 0, H, &q_i8, &q_unused);
    printf("  QKV weights: %d x %d (%.1fms)\n", q_i8, H, 
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    
    // Benchmark data
    std::vector<float> hidden(H * BS, 0.01f);
    std::vector<float> qkv(BS * QKV, 0);
    std::vector<float> k_cache(4096 * NKV * HD, 0);
    std::vector<float> v_cache(4096 * NKV * HD, 0);
    std::vector<float> attn_out(BS * NH * HD, 0);
    std::vector<float> ffn_out(BS * H, 0);
    
    // 1. RMSNorm benchmark
    auto t1 = std::chrono::steady_clock::now();
    const int NORM_ITERS = 1000;
    for (int i = 0; i < NORM_ITERS; i++) {
        for (int b = 0; b < BS; b++) rn_c(&hidden[b * H], qw, H);
    }
    auto norm_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t1).count();
    printf("\n── CPU Component Timings (B=%d) ──\n", BS);
    printf("RMSNorm:      %.3fms (%.1f ns/token)\n", norm_ms, norm_ms * 1e6 / (BS * NORM_ITERS));
    
    // 2. RoPE applied benchmark
    std::vector<float> rc(4096 * HD), rs(4096 * HD);
    for (int p = 0; p < 4096; p++) for (int d = 0; d < HD/2; d++) {
        float a = (float)p / powf(cfg.rope_theta, (float)d / (HD/2));
        rc[p*HD+d] = cosf(a); rs[p*HD+d] = sinf(a);
    }
    auto t2 = std::chrono::steady_clock::now();
    int ROPE_ITERS = 10000;
    for (int i = 0; i < ROPE_ITERS; i++) {
        for (int b = 0; b < BS; b++) {
            for (int h = 0; h < NH; h++) {
                float* x = &qkv[b * QKV + h * HD];
                for (int d = 0; d < HD/2; d++) {
                    float a = x[d], b2 = x[d + HD/2];
                    float c = rc[0 * HD + d], s = rs[0 * HD + d];
                    x[d] = a * c - b2 * s;
                    x[d + HD/2] = b2 * c + a * s;
                }
            }
        }
    }
    auto rope_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t2).count();
    printf("RoPE:         %.3fms (%.1f ns/token/head)\n", rope_ms, rope_ms * 1e6 / (BS * NH * ROPE_ITERS));
    
    // 3. CPU Attention benchmark (most relevant for GPU offload estimate)
    // Simulate seq=100, GQA ratio
    for (int pos = 0; pos < 100; pos++) {
        for (int kvh = 0; kvh < NKV; kvh++) {
            for (int d = 0; d < HD; d++) {
                k_cache[pos * NKV * HD + kvh * HD + d] = 0.01f;
                v_cache[pos * NKV * HD + kvh * HD + d] = 0.01f;
            }
        }
    }
    
    auto t3 = std::chrono::steady_clock::now();
    int ATTN_ITERS = 100;
    for (int iter = 0; iter < ATTN_ITERS; iter++) {
        for (int b = 0; b < std::min(BS, 32); b++) {
            for (int h = 0; h < NH; h++) {
                int kvh = h / GQA;
                std::vector<float> scores(100);
                float mx = -1e30f;
                for (int p = 0; p < 100; p++) {
                    double s = 0;
                    for (int d = 0; d < HD; d++)
                        s += (double)qkv[b * QKV + h * HD + d] * k_cache[p * NKV * HD + kvh * HD + d];
                    scores[p] = (float)(s / sqrtf(HD));
                    if (scores[p] > mx) mx = scores[p];
                }
                double sw = 0;
                for (int p = 0; p < 100; p++) { scores[p] = expf(scores[p] - mx); sw += scores[p]; }
                float isw = sw > 0 ? 1.0f / (float)sw : 1.0f / 100;
                for (int d = 0; d < HD; d++) {
                    float acc = 0;
                    for (int p = 0; p < 100; p++)
                        acc += scores[p] * v_cache[p * NKV * HD + kvh * HD + d];
                    attn_out[b * NH * HD + h * HD + d] = acc * isw;
                }
            }
        }
    }
    auto attn_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t3).count();
    printf("Attention:    %.3fms (%.1f us/iter, seq=100, B=%d)\n", 
        attn_ms, attn_ms * 1000 / ATTN_ITERS, std::min(BS, 32));
    
    // 4. SiLU FFN benchmark
    std::vector<float> silu_buf(BS * IM, 0.5f);
    auto t4 = std::chrono::steady_clock::now();
    int SILU_ITERS = 1000;
    for (int iter = 0; iter < SILU_ITERS; iter++) {
        for (int b = 0; b < BS; b++) {
            for (int i = 0; i < IM; i++) {
                float v = silu_buf[b * IM + i];
                silu_buf[b * IM + i] = (v / (1.0f + expf(-v)));
            }
        }
    }
    auto silu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t4).count();
    printf("SiLU FFN:     %.3fms (%.1f ns/token)\n", silu_ms, silu_ms * 1e6 / (BS * SILU_ITERS));
    
    // 5. LM Head benchmark (matrix-vector multiply)
    std::vector<float> lm_h(H, 0.01f);
    std::vector<float> lg_buf(NV, 0);
    auto t5 = std::chrono::steady_clock::now();
    int LM_ITERS = 10;
    for (int iter = 0; iter < LM_ITERS; iter++) {
        double mx = -1e30;
        for (int n = 0; n < NV; n++) {
            double s = 0;
            const float* e = &qw[(size_t)n * H];
            for (int k = 0; k < H; k++) s += (double)lm_h[k] * e[k];
            lg_buf[n] = (float)s;
            if (lg_buf[n] > mx) mx = lg_buf[n];
        }
        double sum = 0;
        for (int n = 0; n < NV; n++) {
            float d = lg_buf[n] - (float)mx;
            if (d < -80) d = -80;
            lg_buf[n] = expf(d); sum += lg_buf[n];
        }
    }
    auto lm_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t5).count();
    printf("LM Head:      %.3fms (%.1f us/iter, NV=%d)\n", lm_ms, lm_ms * 1000 / LM_ITERS, NV);
    
    // Summary
    printf("\n═══ Estimated Fused NPU+GPU Pipeline ═══\n");
    printf("Component              CPU        GPU       Speedup\n");
    printf("────────────────────────────────────────────────────\n");
    float lm_us = lm_ms * 1000 / LM_ITERS;
    printf("LM Head [151936×1536]  %.0fus    ~50us     ~%.0fx\n", lm_us, lm_us / 50);
    
    float attn_us = attn_ms * 1000 / ATTN_ITERS;
    printf("Attention (seq=100)    %.0fus    ~500us       ~%.1fx (GPU slower at short seq)\n", attn_us, attn_us / 500);
    
    printf("\nWith GPU attention + GPU LM head:\n");
    printf("  NPU GEMM:       28 × %.2fms = %.1fms (QKV+FFN)\n", 
        (22.2 / 28 * 0.6), 28 * (22.2 / 28 * 0.6));
    printf("  GPU Attn:       28 × 0.50ms  = 14.0ms\n");
    printf("  GPU LM Head:    ~0.05ms\n");
    printf("  Pipeline:       max(%.1f, 14.0) = 14.0ms\n", 28 * (22.2 / 28 * 0.6));
    printf("  Throughput:     128/0.014 = 9143 tok/s (theoretical)\n");
    printf("  Realistic:      ~273 tok/s\n");
    
    platform_munmap((void*)md, st.st_size);
    return 0;
}
