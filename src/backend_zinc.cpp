// backend_zinc.cpp — general-purpose GGUF backend via ZINC (engine/gpu),
// the multi-architecture, multi-quant Vulkan compute engine. Unlike
// zaya_engine's hand-tuned HIP kernels (hardcoded to one architecture), ZINC
// derives its runtime config from GGUF metadata itself — no dims rejection
// needed here, matching that generality.
//
// ZINC is a genuinely separate open-source project (github.com/zolotukhin/zinc,
// forked to bong-water-water-bong/zinc — see /home/bcloud/zinc, NOT the stale
// vendored copy at engine/gpu inside this repo). This backend links against
// libzinc.so, built from that fork's src/c_abi.zig (a thin C ABI shim added
// there, mirroring zaya_engine.h's opaque-state init/generate/destroy
// convention — no changes to ZINC's own compute internals).
//
// The C ABI exposes a single fused generate call (prompt text in, response
// text out) rather than per-token stepping — ZINC's own compute.forward.generate()
// already does a real batched prefill + full decode loop in one call, and
// text is the only safe boundary since ZINC's native tokenizer vocabulary
// (derived per-model from each GGUF's own embedded vocab) won't generally
// match whatever tokenizer produced the token IDs Backend::generate() receives.
// So, like backend_flm.cpp, this bridges token IDs <-> text via g_tokenizer,
// batching queries at word/punctuation boundaries rather than one interior
// call per prompt token. Unlike FLM (an external subprocess with its own
// persistent conversational state), each zinc_generate_text() call is a
// fresh, stateless computation over exactly the tokens passed in — so the
// FLM backend's context-pollution problem doesn't apply here; redundant
// calls during prefill accumulation are pure wasted compute, not a
// correctness bug.

#include "backend.h"
#include "simple_tokenizer.h"
#include <cstdio>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>
#include <chrono>

extern "C" {
    void* zinc_init(const char* gguf_path);
    void zinc_destroy(void* state);
    long long zinc_generate_text(void* state, const char* prompt_text, unsigned int max_tokens,
                                  char* out_buf, size_t out_cap);
}

struct ZincBackend : Backend {
    void* zinc_state_ = nullptr;

    std::vector<int> prompt_tokens_;
    std::vector<int> response_tokens_;
    size_t response_pos_ = 0;
    int last_returned_ = -1;

    ZincBackend() { type = BackendType::ZINC_GPU; name = "ZINC GPU (Vulkan, multi-arch)"; }
    ~ZincBackend() override { destroy(); }

    bool init(const ModelConfig& model_cfg, const std::string& weights_dir) override {
        (void)weights_dir;
        cfg = model_cfg;
        destroy();

        if (cfg.model_path.empty() ||
            (cfg.format != ModelFormat::GGUF && cfg.format != ModelFormat::H1B)) {
            fprintf(stderr, "ZINC: no GGUF model path available\n");
            return false;
        }

        printf("ZINC: loading %s...\n", cfg.model_path.c_str());
        zinc_state_ = zinc_init(cfg.model_path.c_str());
        if (!zinc_state_) {
            fprintf(stderr, "ZINC: zinc_init failed (model load or Vulkan init error — see stderr above)\n");
            return false;
        }
        initialized = true;
        printf("ZINC: ready\n");
        return true;
    }

    bool reset() override {
        prompt_tokens_.clear();
        response_tokens_.clear();
        response_pos_ = 0;
        last_returned_ = -1;
        return true;
    }

    bool forward(int, float*) override { return false; } // no hidden states — use generate()
    bool lm_head(const float*, float*, int*) override { return false; } // use generate()

    int generate(int token_id) override {
        if (!initialized) return -1;

        if (token_id == last_returned_ && !response_tokens_.empty()) {
            if (response_pos_ < response_tokens_.size()) {
                last_returned_ = response_tokens_[response_pos_++];
                return last_returned_;
            }
            last_returned_ = g_tokenizer.eos_id;
            return last_returned_;
        }

        prompt_tokens_.push_back(token_id);
        std::string prompt = g_tokenizer.decode(prompt_tokens_);
        if (prompt.empty()) prompt = " ";

        // Batch queries at word/punctuation boundaries rather than every
        // token (see file header) — pure perf optimization here, not a
        // correctness requirement like it was for FlmBackend.
        char last_ch = prompt.back();
        bool at_boundary = last_ch == ' ' || last_ch == '\n' || ispunct((unsigned char)last_ch);
        if (!at_boundary && !response_tokens_.empty()) {
            last_returned_ = response_tokens_[0];
            return last_returned_;
        }

        static thread_local std::vector<char> out_buf(1 << 20); // 1MB
        long long n = zinc_generate_text(zinc_state_, prompt.c_str(), 64,
                                          out_buf.data(), out_buf.size());
        if (n < 0) {
            fprintf(stderr, "ZINC: generate failed (code %lld)\n", n);
            last_returned_ = g_tokenizer.eos_id;
            return last_returned_;
        }
        std::string text(out_buf.data(), (size_t)n);
        response_tokens_ = g_tokenizer.encode(text);
        response_pos_ = 0;
        if (response_tokens_.empty()) { last_returned_ = g_tokenizer.eos_id; return last_returned_; }
        last_returned_ = response_tokens_[response_pos_++];
        return last_returned_;
    }

    void destroy() override {
        if (zinc_state_) {
            zinc_destroy(zinc_state_);
            zinc_state_ = nullptr;
        }
        initialized = false;
    }

    float benchmark(int tokens) override {
        if (!initialized) return -1.0f;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = g_tokenizer.bos_id;
        for (int i = 0; i < tokens; i++) tok = generate(tok);
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return tokens > 0 ? ms / tokens : 0.0f;
    }

    bool can_infer() const override { return initialized; }
};

Backend* create_zinc_backend() { return new ZincBackend(); }
