// bench_zamba2_e2e.cpp — End-to-end Zamba2 GPU inference benchmark
//
// Loads a real Zamba2 model and measures actual token generation speed
// using the GPU pipeline (tuned Mamba2 + hybrid layers + LM head).
//
// Build & run:
//   cd build && cmake .. && make bench_zamba2_e2e -j$(nproc)
//   ./bench_zamba2_e2e ../models/zamba2-1.2b-instruct-v2-q4_0.gguf [tokens]

#include "zamba2_engine.h"
#include "gguf_zamba2_loader.cpp"
#include <cstdio>
#include <cstring>
#include <chrono>
#include <vector>

// Forward declare GPU state struct (defined in zamba2_engine_hip.hip)
struct Zamba2HIPState;

extern "C" {
    Zamba2HIPState* zamba2_hip_init(Zamba2Model& model);
    void zamba2_hip_forward(Zamba2HIPState*, Zamba2Model&, int, float*, int);
    void zamba2_hip_destroy(Zamba2HIPState*);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [tokens=20] [warmup=5]\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    int n_tokens = argc > 2 ? atoi(argv[2]) : 20;
    int n_warmup = argc > 3 ? atoi(argv[3]) : 5;
    if (n_tokens < 1) n_tokens = 1;

    fprintf(stderr, "\n╔══════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  Zamba2 End-to-End GPU Benchmark              ║\n");
    fprintf(stderr, "╚══════════════════════════════════════════════════╝\n\n");

    // ── Load model ──
    fprintf(stderr, "  Loading model: %s\n", model_path);
    Zamba2Model model;
    if (!load_zamba2_from_gguf(model_path, model)) {
        fprintf(stderr, "  ❌ Failed to load model\n");
        return 1;
    }

    auto& cfg = model.cfg;
    int d_ff = 2 * cfg.d_model;  // Zamba2 FFN intermediate size
    fprintf(stderr, "  Model: %s\n", model_path);
    fprintf(stderr, "  Config: H=%d L=%d d_state=%d d_inner=%d\n",
            cfg.d_model, cfg.n_layers, cfg.d_state, cfg.d_inner);
    fprintf(stderr, "  Hybrid: %d layers  Attn: NH=%d NKV=%d HD=%d\n",
            (int)model.hybrid_layers.size(), cfg.n_attn_heads, cfg.n_kv_heads, cfg.attn_head_dim);
    fprintf(stderr, "  Mamba:  %d layers  SSM: n_head=%d head_dim=%d\n",
            (int)model.mamba_layers.size(), cfg.n_head, cfg.head_dim);
    fprintf(stderr, "  Tokens: %d  Warmup: %d\n\n", n_tokens, n_warmup);

    // ── Init GPU state ──
    fprintf(stderr, "  Initializing GPU...\n");
    model.init_state();
    Zamba2HIPState* gpu = zamba2_hip_init(model);
    if (!gpu) {
        fprintf(stderr, "  ❌ Failed to init GPU state\n");
        return 1;
    }
    fprintf(stderr, "  ✅ GPU initialized\n\n");

    // ── Warmup ──
    std::vector<float> logits(cfg.vocab_size);
    fprintf(stderr, "  Warming up (%d tokens)...\n", n_warmup);
    int token = 1;  // BOS
    for (int i = 0; i < n_warmup; i++) {
        zamba2_hip_forward(gpu, model, token, logits.data(), i);
        token = 0;
        for (int v = 1; v < cfg.vocab_size; v++)
            if (logits[v] > logits[token]) token = v;
    }
    model.reset();
    zamba2_hip_destroy(gpu);
    // Re-init GPU for clean state
    gpu = zamba2_hip_init(model);
    // Reset KV cache
    model.reset();

    // ── Benchmark ──
    fprintf(stderr, "  Benchmarking (%d tokens)...\n", n_tokens);
    token = 1;  // BOS
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_tokens; i++) {
        zamba2_hip_forward(gpu, model, token, logits.data(), i);
        // Argmax for next token
        token = 0;
        for (int v = 1; v < cfg.vocab_size; v++)
            if (logits[v] > logits[token]) token = v;
    }

    float ms = std::chrono::duration<float, std::milli>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // ── Results ──
    fprintf(stderr, "\n╔══════════════════════════════════════════════════╗\n");
    fprintf(stderr, "║  RESULTS                                      ║\n");
    fprintf(stderr, "╠══════════════════════════════════════════════════╣\n");
    fprintf(stderr, "║  Total tokens:  %7d                           ║\n", n_tokens);
    fprintf(stderr, "║  Total time:    %7.1f ms                       ║\n", ms);
    fprintf(stderr, "║  Per-token:     %7.3f ms                       ║\n", ms / n_tokens);
    fprintf(stderr, "║  Tokens/sec:    %7.1f                          ║\n", 1000.0f * n_tokens / ms);
    fprintf(stderr, "║  Model:         %d layers (%d mamba + %d hybrid)║\n",
            cfg.n_layers, (int)model.mamba_layers.size(), (int)model.hybrid_layers.size());
    fprintf(stderr, "╚══════════════════════════════════════════════════╝\n");

    // ── Cleanup ──
    zamba2_hip_destroy(gpu);
    return 0;
}
