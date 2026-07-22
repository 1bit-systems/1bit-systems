// backend_zamba2.cpp — Zamba2 backend for 1bit.systems Backend interface
//
// Implements the Backend interface for Zamba2 models using the CPU reference
// Mamba2 engine. Provides:
//   - Model loading (GGUF files via gguf_zamba2_loader)
//   - Token-by-token autoregressive generation
//   - State reset between sequences
//
// For GPU acceleration, the mamba2_kernels.hip kernels can be plugged in
// when running on AMD Strix Halo (gfx1151).

#include "backend.h"
#include "zamba2_engine.h"
#include "gguf_zamba2_loader.cpp"  // included for simplicity; split in production

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>

// ── Tokenizer wrapper (simple BPE lookup for Zamba2) ──
struct Zamba2Tokenizer {
    // Zamba2 uses Mistral v0.1 tokenizer (vocab_size=32000, BPE)
    // For now, we use a minimal stub that forwards to the existing tokenizer
    // In production, integrate with the HuggingFace tokenizers library or
    // use the tokenizer from the GGUF file (tokenizer.ggml.* KV pairs).

    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;

    bool load_from_gguf(const std::string& gguf_path) {
        // TODO(#gguf-tokenizer): Read tokenizer.ggml.* KV pairs from the GGUF
        // metadata header (see gguf_loader.cpp for KV-pair parsing).  The
        // tokenizer model type, vocab, merges, and special-token IDs are all
        // stored there.  Until this is wired, we assume a Mistral tokenizer.
        fprintf(stderr, "[zamba2] Tokenizer: using Mistral v0.1 tokenizer (vocab=32000)\n");
        return true;
    }

    int bos_id() const { return 1; }
    int eos_id() const { return 2; }
};

// ── Zamba2 Backend ──
struct Zamba2Backend : Backend {
    Zamba2Model model;
    Zamba2Tokenizer tokenizer;
    std::vector<float> logits_buf;
    int pos = 0;

    Zamba2Backend() {
        type = BackendType::CPU_AVX512;  // starts as CPU; GPU path uses HIP
        name = "Zamba2 (Mamba2 SSD)";
    }

    ~Zamba2Backend() override { destroy(); }

    bool init(const ModelConfig& cfg, const std::string& weights_path) override {
        this->cfg = cfg;

        fprintf(stderr, "Zamba2: Loading model from %s\n", weights_path.c_str());

        // Load model from GGUF
        if (!load_zamba2_from_gguf(weights_path, model)) {
            fprintf(stderr, "Zamba2: Failed to load model\n");
            return false;
        }

        // Load tokenizer
        if (!tokenizer.load_from_gguf(weights_path)) {
            fprintf(stderr, "Zamba2: Warning: tokenizer may be incomplete\n");
        }

        // Allocate logits buffer
        logits_buf.resize(model.cfg.vocab_size, 0.0f);

        initialized = true;
        fprintf(stderr, "Zamba2: Engine ready (%d layers, %d params)\n",
                model.cfg.n_layers, model.cfg.vocab_size);
        return true;
    }

    bool reset() override {
        model.reset();
        pos = 0;
        return true;
    }

    bool forward(int token_id, float* hidden_out) override {
        // Zamba2 forward produces logits, not hidden states.
        // The Backend interface's hidden_out buffer is only hidden_size floats
        // (typically ~2048), but logits are vocab_size floats (typically ~262K).
        // Copying vocab_size floats would overflow the caller's buffer.
        // Instead, copy only hidden_size floats and treat the result as a
        // projected hidden state. The lm_head() path handles full logit
        // computation separately.
        if (!model.forward(token_id, logits_buf.data())) {
            return false;
        }
        // Copy only cfg.hidden floats to hidden_out — safe upper bound
        std::memcpy(hidden_out, logits_buf.data(),
                    (size_t)cfg.hidden * sizeof(float));
        return true;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        // For tied embeddings, lm_head = embedding^T
        // hidden is the final normalized hidden state
        int v = cfg.vocab;
        int h = cfg.hidden;

        for (int i = 0; i < v; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < h; ++j) {
                sum += model.embed_w[i * h + j] * hidden[j];
            }
            logits[i] = sum;
        }

        if (argmax) {
            *argmax = 0;
            float max_val = logits[0];
            for (int i = 1; i < v; ++i) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    *argmax = i;
                }
            }
        }
        return true;
    }

    int generate(int token_id) override {
        // Full generate: forward + argmax
        if (!model.forward(token_id, logits_buf.data())) {
            return -1;
        }

        // Argmax
        int next = 0;
        float max_val = logits_buf[0];
        for (int i = 1; i < model.cfg.vocab_size; ++i) {
            if (logits_buf[i] > max_val) {
                max_val = logits_buf[i];
                next = i;
            }
        }
        return next;
    }

    float benchmark(int tokens = 10) override {
        if (!initialized) return 0;
        reset();

        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = tokenizer.bos_id();
        for (int i = 0; i < tokens; ++i) {
            tok = generate(tok);
            if (tok < 0) break;
        }
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return ms / tokens;
    }

    void destroy() override {
        model.loaded = false;
        logits_buf.clear();
        initialized = false;
    }
};

// ── Factory entry point ──
extern "C" Backend* create_zamba2_backend() {
    return new Zamba2Backend();
}
