#include <chrono>
// backend_adapter.h — Adapts InferenceBackend (tests/) to Backend (src/) interface
#pragma once
#include "backend.h"
#include "../../src/backend.h"

// Wraps an InferenceBackend pointer into the canonical Backend interface.
// This allows zaya_server's backends to be used with BackendManager.
class InferenceBackendAdapter : public Backend {
    InferenceBackend* wrapped;
    int pos = 0;
public:
    InferenceBackendAdapter(InferenceBackend* b) : wrapped(b) {
        type = static_cast<BackendType>(static_cast<int>(b->type()));
        name = b->name();
    }

    bool init(const ModelConfig& cfg, const std::string& weights_dir) override {
        // Convert ModelConfig: src uses short names, tests use long names
        // Copy all fields to match test ModelConfig layout
        ModelConfig tc;
        tc.hidden_size = cfg.hidden;
        tc.num_heads = cfg.n_heads;
        tc.num_kv_heads = cfg.n_kv_heads;
        tc.head_dim = cfg.head_dim;
        tc.num_layers = cfg.n_layers;
        tc.vocab_size = cfg.vocab;
        tc.intermediate_size = cfg.n_ff;
        tc.num_experts = cfg.n_experts;
        tc.router_hidden = cfg.router_hidden;
        tc.max_seq_len = cfg.max_seq_len;
        tc.rope_theta = cfg.rope_theta;
        tc.rms_norm_eps = cfg.rms_norm_eps;
        tc.model_name = cfg.model_name;
        tc.model_path = cfg.model_path;
        tc.weights_dir = cfg.weights_dir;
        return wrapped->load_model(tc);
    }

    bool reset() override { wrapped->reset_state(); pos = 0; return true; }

    bool forward(int token_id, float* hidden_out) override {
        // Test backends don't expose hidden state separately.
        // generate() is the correct path (fused forward+lm_head).
        (void)token_id; (void)hidden_out;
        return false;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        (void)hidden; (void)logits; (void)argmax;
        return false;
    }

    int generate(int token_id) override {
        int result = wrapped->forward(token_id, pos);
        pos++;
        return result;
    }

    void destroy() override { wrapped->unload_model(); }
    float benchmark(int tokens = 10) override {
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = 100;
        for (int i = 0; i < tokens; i++) {
            tok = wrapped->forward(tok, i);
            if (tok < 0) break;
        }
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return ms / tokens;
    }
    bool can_infer() const override { return true; }
};
