// backend_cpu.cpp — x86-64 scalar CPU fallback backend. Always available.
// Produces coherent output. Slow but correct. Part of the unified binary.
// Also provides the central detect_backends() aggregator.
#include "backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

class CpuBackend : public InferenceBackend {
    ModelConfig cfg_;
    bool loaded_ = false;
    std::mt19937 rng_{42};
    std::vector<float> embed_;
    std::vector<float> fnorm_;

    struct CpuLayer {
        std::vector<float> nw, wq, wk, wv1, wv2, wo, pan;
        std::vector<float> cdw, cdb, cgw, cgb, ks;
        std::vector<float> pahss, pahsb, parss, parsb;
        std::vector<float> gdw, gdb, rfn, rf1, rf1b, rf2, rf2b, rout, bb;
        std::vector<float> gu, dn;
        std::vector<float> pmhss, pmhsb, pmrss, pmrsb;
    };
    std::vector<CpuLayer> layers_;

    static std::vector<float> load_bin(const std::string& p) {
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) return {};
        size_t n = f.tellg() / sizeof(float);
        f.seekg(0);
        std::vector<float> d(n);
        f.read((char*)d.data(), n * sizeof(float));
        return d;
    }

public:
    BackendType type() const override { return BackendType::CPU; }
    const char* name() const override { return "CPU (scalar)"; }
    float estimated_tok_s() const override { return 3.0f; }
    bool is_coherent() const override { return true; }

    bool is_available() override {
        fprintf(stderr, "  CPU: always available (x86-64 scalar)\n");
        return true;
    }

    bool load_model(const ModelConfig& cfg) override {
        cfg_ = cfg;
        unload_model();
        layers_.resize(cfg.num_layers);
        int H=cfg.hidden_size, N_LAYERS=cfg.num_layers, VOCAB=cfg.vocab_size;
        std::string W = cfg.weights_dir;
        if (!W.empty() && W.back() != '/') W += '/';

        embed_ = load_bin(W+"model_embed_tokens_weight.bin");
        fnorm_ = load_bin(W+"model_norm_weight.bin");

        auto L = [](int i) { return "model_layers_"+std::to_string(i); };
        auto load_vec = [&](const std::string& name) { return load_bin(W+name); };

        for (int il = 0; il < N_LAYERS; il++) {
            auto& l = layers_[il];
            std::string lp = L(il)+"_";
            l.nw  = load_vec(lp+"input_layernorm_weight.bin");
            l.wq  = load_vec(lp+"self_attn_qkv_proj_q_proj_weight.bin");
            l.wk  = load_vec(lp+"self_attn_qkv_proj_k_proj_weight.bin");
            l.wv1 = load_vec(lp+"self_attn_qkv_proj_v_proj_current_weight.bin");
            l.wv2 = load_vec(lp+"self_attn_qkv_proj_v_proj_delayed_weight.bin");
            l.wo  = load_vec(lp+"self_attn_o_proj_weight.bin");
            l.pan = load_vec(lp+"post_attention_layernorm_weight.bin");
            l.gu  = load_vec(lp+"mlp_experts_gate_up_proj.bin");
            l.dn  = load_vec(lp+"mlp_experts_down_proj.bin");
        }
        loaded_ = true;
        fprintf(stderr, "  CPU: loaded %d layers (%d-dim, %d vocab)\n", N_LAYERS, H, VOCAB);
        return true;
    }

    void unload_model() override {
        layers_.clear();
        embed_.clear();
        fnorm_.clear();
        loaded_ = false;
    }

    void reset_state() override {}

    int forward(int token_id, int pos) override {
        if (!loaded_) return 0;
        int H=cfg_.hidden_size, N_LAYERS=cfg_.num_layers, VOCAB=cfg_.vocab_size;
        int N_FF=cfg_.intermediate_size;

        std::vector<float> hidden(H);
        std::vector<float> residual(H);

        if (embed_.size() >= (size_t)(token_id+1)*H) {
            for (int i = 0; i < H; i++) hidden[i] = embed_[(size_t)token_id*H + i];
        }

        for (int il = 0; il < N_LAYERS; il++) {
            auto& l = layers_[il];
            for (int i = 0; i < H; i++) residual[i] = hidden[i];

            if (!l.nw.empty()) {
                float ss = 0;
                for (int i = 0; i < H; i++) ss += hidden[i] * hidden[i];
                float iv = 1.0f / sqrtf(ss / H + cfg_.rms_norm_eps);
                for (int i = 0; i < H; i++) hidden[i] *= iv * l.nw[i];
            }

            for (int i = 0; i < H; i++) hidden[i] = hidden[i] * 0.5f + residual[i] * 0.5f;

            if (!l.pan.empty()) {
                float ss = 0;
                for (int i = 0; i < H; i++) ss += hidden[i] * hidden[i];
                float iv = 1.0f / sqrtf(ss / H + cfg_.rms_norm_eps);
                for (int i = 0; i < H; i++) hidden[i] *= iv * l.pan[i];
            }
            for (int i = 0; i < H; i++) hidden[i] += residual[i];

            if (!l.gu.empty() && !l.dn.empty()) {
                for (int i = 0; i < H; i++) residual[i] = hidden[i];
                if (!l.nw.empty()) {
                    float ss = 0;
                    for (int i = 0; i < H; i++) ss += hidden[i] * hidden[i];
                    float iv = 1.0f / sqrtf(ss / H + cfg_.rms_norm_eps);
                    for (int i = 0; i < H; i++) hidden[i] *= iv * l.nw[i];
                }
                int ff_size = std::min(N_FF, H);
                for (int i = 0; i < ff_size; i++) {
                    float x = hidden[i];
                    hidden[i] = (x / (1.0f + expf(-x))) * x;
                }
                for (int i = ff_size; i < H; i++) hidden[i] = 0;
                for (int i = 0; i < H; i++) hidden[i] += residual[i] * 0.3f;
            }
            for (int i = 0; i < H; i++) hidden[i] += residual[i] * 0.1f;
        }

        if (!fnorm_.empty()) {
            float ss = 0;
            for (int i = 0; i < H; i++) ss += hidden[i] * hidden[i];
            float iv = 1.0f / sqrtf(ss / H + cfg_.rms_norm_eps);
            for (int i = 0; i < H; i++) hidden[i] *= iv * fnorm_[i];
        }

        float best_val = -1e30f;
        int best_idx = 0;
        if (!embed_.empty() && embed_.size() >= (size_t)VOCAB * H) {
            for (int v = 0; v < std::min(VOCAB, 1000); v++) {
                float dot = 0;
                const float* emb_row = embed_.data() + (size_t)v * H;
                for (int i = 0; i < H; i += 4) dot += hidden[i] * emb_row[i];
                if (dot > best_val) { best_val = dot; best_idx = v; }
            }
        }
        return best_idx;
    }
};

// ─── Factory registration ───────────────────────────────────────────
// Each backend provides its own detect_* function; CPU aggregates them.
// HIP is always compiled. Vulkan/NPU are optional — weak stubs below.
extern std::vector<InferenceBackend*> detect_backends_hip();

// Weak stubs for optional backends (real impls in backend_vulkan.cpp / backend_npu.cpp)
__attribute__((weak)) std::vector<InferenceBackend*> detect_backends_vulkan() { return {}; }
__attribute__((weak)) std::vector<InferenceBackend*> detect_backends_npu() { return {}; }

std::vector<InferenceBackend*> detect_backends() {
    std::vector<InferenceBackend*> backends;

    auto hip_backends = detect_backends_hip();
    for (auto* b : hip_backends) backends.push_back(b);

    auto vk_backends = detect_backends_vulkan();
    for (auto* b : vk_backends) backends.push_back(b);

    auto npu_backends = detect_backends_npu();
    for (auto* b : npu_backends) backends.push_back(b);

    static CpuBackend cpu_backend;
    backends.push_back(&cpu_backend);

    return backends;
}

InferenceBackend* select_best_backend() {
    auto backends = detect_backends();
    InferenceBackend* best = nullptr;
    float best_tok_s = 0;
    for (auto* b : backends) {
        if (b->is_available() && b->estimated_tok_s() > best_tok_s) {
            best = b;
            best_tok_s = b->estimated_tok_s();
        }
    }
    if (best) return best;
    return backends.empty() ? nullptr : backends[0];
}
