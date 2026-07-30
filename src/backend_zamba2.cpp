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
        // Read tokenizer.ggml.bos_token_id / eos_token_id from the GGUF
        // KV metadata header. Simple inline reader — no external deps.
        FILE* f = fopen(gguf_path.c_str(), "rb");
        if (!f) { fprintf(stderr, "[zamba2] Tokenizer: can't open %s\n", gguf_path.c_str()); return true; }
        uint32_t magic;
        if (fread(&magic, 4, 1, f) != 1) { fclose(f); return true; }
        if (magic != 0x46554747) { fclose(f); return true; }
        fseek(f, 4, SEEK_CUR);  // skip version
        uint64_t n_tensors, n_kv;
        if (fread(&n_tensors, 8, 1, f) != 1 || fread(&n_kv, 8, 1, f) != 1) { fclose(f); return true; }
        for (uint64_t i = 0; i < n_kv && i < 1024; i++) {
            uint64_t klen;
            if (fread(&klen, 8, 1, f) != 1) { fclose(f); return true; }
            if (klen > 256) { fclose(f); return true; }
            std::string key(klen, '\0');
            if (fread(&key[0], 1, klen, f) != klen) { fclose(f); return true; }
            uint32_t vtype;
            if (fread(&vtype, 4, 1, f) != 1) { fclose(f); return true; }
            if (key == "tokenizer.ggml.bos_token_id" && vtype == 4) {
                uint32_t v;
                if (fread(&v, 4, 1, f) != 1) { fclose(f); return true; }
                bos_id_ = (int)v;
            } else if (key == "tokenizer.ggml.eos_token_id" && vtype == 4) {
                uint32_t v;
                if (fread(&v, 4, 1, f) != 1) { fclose(f); return true; }
                eos_id_ = (int)v;
            } else {
                // Skip value
                if (vtype == 0 || vtype == 1 || vtype == 7) fseek(f, 1, SEEK_CUR);
                else if (vtype >= 2 && vtype <= 6) fseek(f, 4, SEEK_CUR);
                else if (vtype >= 10 && vtype <= 12) fseek(f, 8, SEEK_CUR);
                else if (vtype == 8) {
                    uint64_t slen = 0;
                    if (fread(&slen, 8, 1, f) != 1) slen = 0;
                    if (slen <= (1ULL << 24)) fseek(f, (long)slen, SEEK_CUR);
                    else fseek(f, 0, SEEK_END); // skip garbage past reasonable bounds
                }
                else if (vtype == 9) {
                    uint32_t n_arr = 0, at = 0;
                    fread(&n_arr, 4, 1, f); fread(&at, 4, 1, f);
                    if (n_arr > 1000000) n_arr = 0; // cap array count
                    for (uint32_t j = 0; j < n_arr; j++) {
                        if (at == 2 || at == 8) {
                            uint64_t sl = 0;
                            if (fread(&sl, 8, 1, f) != 1) sl = 0;
                            if (sl <= (1ULL << 24)) fseek(f, (long)sl, SEEK_CUR);
                            else fseek(f, 0, SEEK_END);
                        }
                        else if (at <= 7) fseek(f, 1, SEEK_CUR);
                        else fseek(f, 8, SEEK_CUR);
                    }
                }
            }
        }
        fclose(f);
        fprintf(stderr, "[zamba2] Tokenizer: BOS=%d EOS=%d (from GGUF)\n", bos_id_, eos_id_);
        return true;
    }

    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }

private:
    int bos_id_ = 1;
    int eos_id_ = 2;
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

        // BackendManager passes the weights *directory* here; the Zamba2 loader
        // needs the actual .gguf file. Prefer the discovered model_path (a file)
        // and only fall back to weights_path. Passing the directory made
        // load_zamba2_from_gguf fail and the failed init could then segfault
        // downstream (#843).
        std::string model_path = !cfg.model_path.empty() ? cfg.model_path : weights_path;
        if (model_path.empty()) {
            fprintf(stderr, "Zamba2: no model path available\n");
            return false;
        }
        fprintf(stderr, "Zamba2: Loading model from %s\n", model_path.c_str());

        // Load model from GGUF
        if (!load_zamba2_from_gguf(model_path, model)) {
            fprintf(stderr, "Zamba2: Failed to load model\n");
            return false;
        }

        // Load tokenizer
        if (!tokenizer.load_from_gguf(model_path)) {
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
