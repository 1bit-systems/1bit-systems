// test_mamba1_backend.cpp — Direct Mamba1 GPU backend test
// Loads a BlackMamba GGUF and runs inference through the Mamba1 HIP backend.
// Bypasses the HTTP server and ZINC pipeline entirely.
#include "backend.h"
#include "gguf_reader.h"
// Route is handled inline — we call create_mamba1_backend directly
#include <cstdio>
#include <chrono>
#include <vector>
#include <string>

// From src/backend.h
extern "C" Backend* create_mamba1_backend();

// Minimal GGUF metadata reader for Mamba config
static bool read_mamba_config(const std::string& path, ModelConfig& cfg) {
    GgufReader r;
    if (!r.open(path)) {
        fprintf(stderr, "Failed to open GGUF: %s\n", path.c_str());
        return false;
    }

    cfg.architecture = r.architecture();
    cfg.arch = rcpp_arch_from_string(cfg.architecture.c_str());
    cfg.model_path = path;
    cfg.format = ModelFormat::GGUF;

    auto slash = path.find_last_of('/');
    auto dot = path.find_last_of('.');
    cfg.model_name = path.substr(slash + 1, dot - slash - 1);

    // Read Mamba-specific metadata
    uint32_t u32;
    if (r.get_u32("mamba.block_count", u32)) cfg.num_layers = u32;
    if (r.get_u32("mamba.embedding_length", u32)) cfg.hidden_size = u32;
    if (r.get_u32("mamba.feed_forward_length", u32)) cfg.intermediate_size = u32;
    if (r.get_u32("mamba.vocab_size", u32)) cfg.vocab_size = u32;
    if (r.get_u32("mamba.ssm.state_size", u32)) cfg.head_dim = u32;  // reuse for d_state
    if (r.get_u32("mamba.ssm.inner_size", u32)) {} // inner_size = hidden*2 typically
    if (r.get_u32("mamba.attention.head_count", u32)) cfg.num_attention_heads = u32;
    if (r.get_u32("mamba.expert_count", u32)) cfg.num_experts = u32;

    fprintf(stderr, "  Architecture: %s (arch_enum=%d)\n", cfg.architecture.c_str(), cfg.arch);
    fprintf(stderr, "  Hidden: %d, Layers: %d, Vocab: %d, d_state: %d\n",
            cfg.hidden_size, cfg.num_layers, cfg.vocab_size, cfg.head_dim);
    fprintf(stderr, "  Experts: %d\n", cfg.num_experts);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [tokens]\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    int n_tokens = argc > 2 ? atoi(argv[2]) : 128;  // default 128 for steady-state

    // ── Read model config from GGUF ──
    fprintf(stderr, "\n=== Reading model config ===\n");
    ModelConfig cfg;
    if (!read_mamba_config(model_path, cfg)) {
        fprintf(stderr, "FAILED: config read\n");
        return 1;
    }

    // ── Create Mamba1 backend directly ──
    fprintf(stderr, "\n=== Creating Mamba1 GPU backend ===\n");
    Backend* backend = create_mamba1_backend();
    if (!backend) {
        fprintf(stderr, "FAILED: create_mamba1_backend returned null\n");
        return 1;
    }
    fprintf(stderr, "  Backend: %s\n", backend->name.c_str());

    // ── Init ──
    fprintf(stderr, "\n=== Loading model weights ===\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!backend->init(cfg, model_path)) {
        fprintf(stderr, "FAILED: backend init\n");
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    float load_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    fprintf(stderr, "  Load time: %.1f ms\n", load_ms);

    // ── Warmup ──
    fprintf(stderr, "\n=== Warmup (3 tokens) ===\n");
    std::vector<float> hidden(cfg.hidden_size);
    std::vector<float> logits(cfg.vocab_size);
    int tok = 1;
    for (int i = 0; i < 3; i++) {
        if (!backend->forward(tok, hidden.data())) {
            fprintf(stderr, "  WARMUP FAIL at token %d (forward)\n", i);
            return 1;
        }
        if (!backend->lm_head(hidden.data(), logits.data(), &tok)) {
            fprintf(stderr, "  WARMUP FAIL at token %d (lm_head)\n", i);
            return 1;
        }
        fprintf(stderr, "  token %d → %d\n", i, tok);
    }

    // ── Benchmark (3 runs for statistical validity) ──
    const int kRuns = 3;
    float runs_ms[kRuns];
    fprintf(stderr, "\n=== Benchmark (%d tokens, %d runs) ===\n", n_tokens, kRuns);
    for (int r = 0; r < kRuns; r++) {
        backend->reset();
        tok = 1;
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_tokens; i++) {
            backend->forward(tok, hidden.data());
            backend->lm_head(hidden.data(), logits.data(), &tok);
        }
        t1 = std::chrono::high_resolution_clock::now();
        runs_ms[r] = std::chrono::duration<float, std::milli>(t1 - t0).count();
        fprintf(stderr, "  run %d: %d tokens in %.1f ms\n", r + 1, n_tokens, runs_ms[r]);
    }
    float sum_ms = 0, sum_sq = 0;
    for (int r = 0; r < kRuns; r++) { sum_ms += runs_ms[r]; sum_sq += runs_ms[r] * runs_ms[r]; }
    float mean_ms = sum_ms / kRuns;
    float std_ms = sqrtf((sum_sq - sum_ms * sum_ms / kRuns) / (kRuns - 1));
    float mean_tok_s = n_tokens / (mean_ms / 1000.0f);
    fprintf(stderr, "\n  %d tokens: %.1f ± %.1f ms = %.1f ± %.1f tok/s\n",
            n_tokens, mean_ms, std_ms, mean_tok_s, mean_tok_s * std_ms / mean_ms);
    fprintf(stderr, "  %.2f ms/tok\n", mean_ms / n_tokens);

    // ── Generate ──
    fprintf(stderr, "\n=== Generation (8 tokens) ===\n");
    backend->reset();
    tok = 1;
    for (int i = 0; i < 8; i++) {
        int next = backend->generate(tok);
        fprintf(stderr, "  [%d] %d → %d\n", i, tok, next);
        tok = next;
        if (tok < 0) break;
    }

    // ── Cleanup ──
    backend->destroy();
    delete backend;
    fprintf(stderr, "\nDone.\n");
    return 0;
}
