// test_cuda_backend.cpp — Direct CUDA GPU backend test
// Loads a model and runs inference through the CUDA backend.
// Follows the same pattern as test_mamba1_backend.cpp (PR #579).
#include "backend.h"
#include "gguf_reader.h"
#include <cstdio>
#include <chrono>
#include <vector>
#include <string>

extern "C" Backend* create_cuda_backend();

static bool read_model_config(const std::string& path, ModelConfig& cfg) {
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

    uint32_t u32;
    if (r.get_u32("llama.block_count", u32) || r.get_u32("bert.block_count", u32))
        cfg.num_layers = u32;
    if (r.get_u32("llama.embedding_length", u32) || r.get_u32("bert.embedding_length", u32))
        cfg.hidden_size = u32;
    if (r.get_u32("llama.feed_forward_length", u32))
        cfg.intermediate_size = u32;
    if (r.get_u32("llama.vocab_size", u32) || r.get_u32("bert.vocab_size", u32))
        cfg.vocab_size = u32;
    if (r.get_u32("llama.attention.head_count", u32))
        cfg.num_heads = u32;
    if (r.get_u32("llama.attention.head_count_kv", u32))
        cfg.num_kv_heads = u32;

    // Fallback: set sensible defaults for common models
    if (cfg.num_heads == 0) cfg.num_heads = 32;
    if (cfg.num_kv_heads == 0) cfg.num_kv_heads = 8;
    if (cfg.head_dim == 0) cfg.head_dim = 128;

    fprintf(stderr, "  Model: %s\n", cfg.model_name.c_str());
    fprintf(stderr, "  Arch: %s (enum=%d)\n", cfg.architecture.c_str(), cfg.arch);
    fprintf(stderr, "  H=%d L=%d NH=%d NKV=%d V=%d FF=%d\n",
            cfg.hidden_size, cfg.num_layers, cfg.num_heads,
            cfg.num_kv_heads, cfg.vocab_size, cfg.intermediate_size);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [tokens]\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    int n_tokens = argc > 2 ? atoi(argv[2]) : 32;

    // ── Read model config ──
    fprintf(stderr, "\n=== Reading model config ===\n");
    ModelConfig cfg;
    if (!read_model_config(model_path, cfg)) {
        fprintf(stderr, "FAILED: config read\n");
        return 1;
    }

    // ── Create CUDA backend ──
    fprintf(stderr, "\n=== Creating CUDA backend ===\n");
    Backend* backend = create_cuda_backend();
    if (!backend) {
        fprintf(stderr, "FAILED: create_cuda_backend() returned null\n"
                        "  CUDA not available or CUDA backend not built.\n"
                        "  Build with: cmake -DUSE_CUDA=ON && cmake --build . --target cuda_backend\n");
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
    int tok = 1;
    for (int i = 0; i < 3; i++) {
        tok = backend->generate(tok);
        fprintf(stderr, "  token %d → %d\n", i, tok);
        if (tok < 0) { fprintf(stderr, "  WARMUP FAIL at token %d\n", i); return 1; }
    }

    // ── Benchmark ──
    fprintf(stderr, "\n=== Benchmark (%d tokens) ===\n", n_tokens);
    backend->reset();
    tok = 1;
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_tokens; i++) {
        tok = backend->generate(tok);
        if (tok < 0) { fprintf(stderr, "  FAIL at token %d\n", i); break; }
    }
    t1 = std::chrono::high_resolution_clock::now();
    float total_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    float tok_s = n_tokens / (total_ms / 1000.0f);
    fprintf(stderr, "\n  %d tokens in %.1f ms = %.1f tok/s\n", n_tokens, total_ms, tok_s);
    fprintf(stderr, "  %.2f ms/tok\n", total_ms / n_tokens);

    // ── Generate sample ──
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
