// Attention Kernel Benchmark — measures flash attention throughput
#include "kernels/attention.h"
#include <cstdio>
#include <chrono>
#include <random>
#include <cmath>

using namespace specdecode::kernels;

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

struct BenchResult {
    const char* name;
    double ms;
};

template <typename F>
BenchResult bench(const char* name, int iterations, F&& fn) {
    for (int i = 0; i < 3; i++) fn();
    double start = now_ms();
    for (int i = 0; i < iterations; i++) fn();
    return {name, (now_ms() - start) / iterations};
}

int main() {
    printf("═══ Attention Kernel Benchmark ═══\n\n");

    std::mt19937 rng(42);
    auto gen = [&](size_t n) {
        std::vector<float> v(n);
        for (auto& x : v) x = (float)rng() / (float)rng.max();
        return v;
    };

    AttentionConfig cfg;
    cfg.num_heads = 16;
    cfg.num_kv_heads = 8;
    cfg.head_dim = 128;
    cfg.gqa_ratio = 2;
    cfg.attn_scale = 1.0f / std::sqrt(128.0f);

    FlashAttention flash_attn(cfg);
    BatchedAttention batched_attn(cfg);

    printf("Model: Qwen3-0.6B (H=16, KV=8, D=128, GQA=2)\n\n");

    // ── Single-query flash attention (decode) ──
    printf("── Flash Attention (Single Query Decode) ──\n");
    printf("%-15s %12s %12s\n", "Seq Len", "Time (ms)", "Tokens/s");
    printf("%s\n", std::string(42, '-').c_str());

    for (int seq_len : {64, 128, 256, 512, 1024, 2048, 4096}) {
        auto q = gen(cfg.num_heads * cfg.head_dim);
        auto k = gen((size_t)seq_len * cfg.num_kv_heads * cfg.head_dim);
        auto v = gen((size_t)seq_len * cfg.num_kv_heads * cfg.head_dim);
        std::vector<float> out(cfg.num_heads * cfg.head_dim);

        auto r = bench("flash", 100, [&]() {
            flash_attn.forward(q, k, v, seq_len, out);
        });

        printf("%-15d %12.4f %12.1f\n", seq_len, r.ms,
               seq_len > 0 ? (1.0 / (r.ms / 1000.0)) : 0.0);
    }

    // ── Batched flash attention (prefill) ──
    printf("\n── Batched Flash Attention (Prefill) ──\n");
    printf("%-15s %12s %12s %12s\n", "N queries", "Seq Len", "Time (ms)", "Tokens/s");
    printf("%s\n", std::string(54, '-').c_str());

    struct PrefillCase { int n_queries; int seq_len; };
    PrefillCase cases[] = {
        {1, 64}, {1, 256}, {1, 1024},
        {4, 64}, {4, 256},
        {16, 64}, {16, 128},
    };

    for (auto& c : cases) {
        auto q = gen((size_t)c.n_queries * cfg.num_heads * cfg.head_dim);
        auto k = gen((size_t)c.seq_len * cfg.num_kv_heads * cfg.head_dim);
        auto v = gen((size_t)c.seq_len * cfg.num_kv_heads * cfg.head_dim);
        std::vector<float> out((size_t)c.n_queries * cfg.num_heads * cfg.head_dim);

        auto r = bench("batched", 50, [&]() {
            batched_attn.forward(q, k, v, c.n_queries, c.seq_len, 0, out);
        });

        printf("%-15d %12d %12.4f %12.1f\n",
               c.n_queries, c.seq_len, r.ms,
               (double)c.n_queries / (r.ms / 1000.0));
    }

    // ── RMSNorm benchmark ──
    printf("\n── RMSNorm (Qwen3-0.6B: H=1024) ──\n");
    {
        auto x = gen(1024);
        auto w = gen(1024);
        std::vector<float> y(1024);

        auto r = bench("RMSNorm", 10000, [&]() {
            rms_norm(x, w, y);
        });
        printf("  Time: %.4f ms (%ld calls/s)\n", r.ms, (long)(1000.0 / r.ms));
    }

    // ── RoPE benchmark ──
    printf("\n── RoPE (head_dim=128) ──\n");
    {
        auto x = gen(128);
        auto r = bench("RoPE", 10000, [&]() {
            apply_rope(x, 100, 128);
        });
        printf("  Time: %.4f ms (%ld calls/s)\n", r.ms, (long)(1000.0 / r.ms));
    }

    // ── SwiGLU benchmark ──
    printf("\n── SwiGLU (inter_dim=3072) ──\n");
    {
        auto gate = gen(3072);
        auto up = gen(3072);
        std::vector<float> out(3072);

        auto r = bench("SwiGLU", 10000, [&]() {
            swiglu(gate, up, out);
        });
        printf("  Time: %.4f ms (%ld calls/s)\n", r.ms, (long)(1000.0 / r.ms));
    }

    printf("\n=== Benchmark Complete ===\n");
    return 0;
}
