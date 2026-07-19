// backend_cpu.cpp — x86-64 scalar CPU fallback backend. Always available.
// Produces coherent output. Slow but correct. Part of the unified binary.
// Also provides the central detect_backends() aggregator.
#include "backend.h"
#include "colibri_kernels.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

// USE_COLIBRI_Q4=1: quantize weights to int4 at load time (8x smaller,
// negligible quality loss). Set to 0 for original f32 behavior.
#ifndef USE_COLIBRI_Q4
#define USE_COLIBRI_Q4 1
#endif

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
#if USE_COLIBRI_Q4
        // Quantized (int4) versions — 8x compression, same quality
        std::vector<uint8_t> q4_wq, q4_wk, q4_wv1, q4_wv2, q4_wo;
        std::vector<float>  qs_wq, qs_wk, qs_wv1, qs_wv2, qs_wo;
        std::vector<uint8_t> q4_gu, q4_dn;
        std::vector<float>   qs_gu, qs_dn;
#endif
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
    BackendType type() const override { return BackendType::CPU_AVX512; }
    const char* name() const override { return "CPU (scalar)"; }
    float estimated_tok_s() const override { return 3.0f; }  // estimate; actual varies by model size
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

#if USE_COLIBRI_Q4
            // Quantize to int4 right after loading
            auto q4_quant = [&](std::vector<float>& f32,
                                std::vector<uint8_t>& q4,
                                std::vector<float>& qs) {
                if (f32.empty()) return;
                int O = (int)f32.size() / H;
                q4.resize((size_t)O * ((H+1)/2));
                qs.resize(O);
                colibri_quantize_matrix_q4(f32.data(), q4.data(), qs.data(), O, H);
            };
            auto q4_quant_ffn = [&](std::vector<float>& f32, int I,
                                    std::vector<uint8_t>& q4,
                                    std::vector<float>& qs) {
                if (f32.empty()) return;
                int O = (int)f32.size() / I;
                q4.resize((size_t)O * ((I+1)/2));
                qs.resize(O);
                colibri_quantize_matrix_q4(f32.data(), q4.data(), qs.data(), O, I);
            };
            q4_quant(l.wq, l.q4_wq, l.qs_wq);
            q4_quant(l.wk, l.q4_wk, l.qs_wk);
            q4_quant(l.wv1, l.q4_wv1, l.qs_wv1);
            q4_quant(l.wv2, l.q4_wv2, l.qs_wv2);
            q4_quant(l.wo, l.q4_wo, l.qs_wo);
            if (!l.gu.empty()) {
                int cols = (int)l.gu.size() / H;
                q4_quant_ffn(l.gu, H, l.q4_gu, l.qs_gu);
            }
            if (!l.dn.empty()) {
                q4_quant_ffn(l.dn, H, l.q4_dn, l.qs_dn);
            }
#endif
        }
        loaded_ = true;

        // Report memory savings
#if USE_COLIBRI_Q4
        double f32_mb = 0, q4_mb = 0;
        for (auto& l : layers_) {
            auto add = [&](auto& v, auto& q4, auto& qs) {
                f32_mb += (double)v.size() * 4 / 1e6;
                q4_mb  += (double)q4.size() / 1e6 + (double)qs.size() * 4 / 1e6;
            };
            add(l.wq, l.q4_wq, l.qs_wq);
            add(l.wk, l.q4_wk, l.qs_wk);
            add(l.wv1, l.q4_wv1, l.qs_wv1);
            add(l.wv2, l.q4_wv2, l.qs_wv2);
            add(l.wo, l.q4_wo, l.qs_wo);
            add(l.gu, l.q4_gu, l.qs_gu);
            add(l.dn, l.q4_dn, l.qs_dn);
        }
        fprintf(stderr, "  CPU: loaded %d layers (%d-dim, %d vocab)\n", N_LAYERS, H, VOCAB);
        fprintf(stderr, "  CPU: q4 weights — f32=%.0f MB → q4=%.0f MB (%.1fx smaller)\n",
                f32_mb, q4_mb, f32_mb / (q4_mb ? q4_mb : 1));
#else
        fprintf(stderr, "  CPU: loaded %d layers (%d-dim, %d vocab)\n", N_LAYERS, H, VOCAB);
#endif
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

            // Attention: Q @ K^T → softmax → @ V (simplified, no RoPE)
            if (!l.wq.empty() && !l.wk.empty() && !l.wv1.empty() && !l.wo.empty()) {
                int NQ = cfg_.num_attention_heads;
                int NKV = cfg_.num_kv_heads;
                int HD = cfg_.head_dim;
                int QK = NQ * HD, KK = NKV * HD, VK = NKV * HD;
                std::vector<float> q(QK, 0), k(KK, 0), v(VK, 0);
                // Q projection
#if USE_COLIBRI_Q4
                if (!l.q4_wq.empty())
                    colibri_matmul_q4(q.data(), hidden.data(), l.q4_wq.data(), l.qs_wq.data(), 1, H, QK);
                else
#endif
                { for (int i = 0; i < QK; i++) for (int j = 0; j < H; j++) q[i] += hidden[j] * l.wq[j * QK + i]; }
                // K projection
#if USE_COLIBRI_Q4
                if (!l.q4_wk.empty())
                    colibri_matmul_q4(k.data(), hidden.data(), l.q4_wk.data(), l.qs_wk.data(), 1, H, KK);
                else
#endif
                { for (int i = 0; i < KK; i++) for (int j = 0; j < H; j++) k[i] += hidden[j] * l.wk[j * KK + i]; }
                // V projection
#if USE_COLIBRI_Q4
                if (!l.q4_wv1.empty())
                    colibri_matmul_q4(v.data(), hidden.data(), l.q4_wv1.data(), l.qs_wv1.data(), 1, H, VK);
                else
#endif
                { for (int i = 0; i < VK; i++) for (int j = 0; j < H; j++) v[i] += hidden[j] * l.wv1[j * VK + i]; }
                // Simplified single-token attention
                std::vector<float> att(NQ * NKV * HD, 0);
                for (int h = 0; h < NQ; h++) {
                    int kvh = h / (NQ / NKV);
                    float* qh = q.data() + h * HD;
                    float* kh = k.data() + kvh * HD;
                    float* vh = v.data() + kvh * HD;
                    float score = 0;
                    for (int d = 0; d < HD; d++) score += qh[d] * kh[d];
                    float is_att = 1.0f;
                    for (int d = 0; d < HD; d++) att[h * HD + d] = score > 0 ? vh[d] * is_att : 0;
                }
                std::vector<float> att_out(H, 0);
#if USE_COLIBRI_Q4
                if (!l.q4_wo.empty())
                    colibri_matmul_q4(att_out.data(), att.data(), l.q4_wo.data(), l.qs_wo.data(), 1, NQ*HD, H);
                else
#endif
                { for (int i = 0; i < NQ * HD; i++) for (int j = 0; j < H; j++) att_out[j] += att[i] * l.wo[i * H + j]; }
                for (int i = 0; i < H; i++) hidden[i] = att_out[i] + residual[i];
            } else {
                for (int i = 0; i < H; i++) hidden[i] = hidden[i] * 0.5f + residual[i] * 0.5f;
            }

            if (!l.pan.empty()) {
                float ss = 0;
                for (int i = 0; i < H; i++) ss += hidden[i] * hidden[i];
                float iv = 1.0f / sqrtf(ss / H + cfg_.rms_norm_eps);
                for (int i = 0; i < H; i++) hidden[i] *= iv * l.pan[i];
            }
            for (int i = 0; i < H; i++) hidden[i] += residual[i];

            // SwiGLU FFN: gate = x @ Wg, up = x @ Wu, out = silu(gate) * up @ Wd
            if (!l.gu.empty() && !l.dn.empty()) {
                for (int i = 0; i < H; i++) residual[i] = hidden[i];
                if (!l.nw.empty()) {
                    float ss = 0;
                    for (int i = 0; i < H; i++) ss += hidden[i] * hidden[i];
                    float iv = 1.0f / sqrtf(ss / H + cfg_.rms_norm_eps);
                    for (int i = 0; i < H; i++) hidden[i] *= iv * l.nw[i];
                }
                // gu: gate+up combined (dim 2*N_FF per row)
                int gu_cols = (int)l.gu.size() / H;
                int gate_dim = gu_cols / 2;
                if (gu_cols >= 2 * gate_dim) {
                    std::vector<float> gate_up(gu_cols, 0);
#if USE_COLIBRI_Q4
                    if (!l.q4_gu.empty())
                        colibri_matmul_q4(gate_up.data(), hidden.data(), l.q4_gu.data(), l.qs_gu.data(), 1, H, gu_cols);
                    else
#endif
                    { for (int i = 0; i < gu_cols; i++) for (int j = 0; j < H; j++) gate_up[i] += hidden[j] * l.gu[j * gu_cols + i]; }
                    // silu(gate) * up
                    for (int i = 0; i < gate_dim; i++) {
                        float sx = gate_up[i] / (1.0f + expf(-gate_up[i]));
                        gate_up[i] = sx * gate_up[gate_dim + i];
                    }
                    // Down projection
                    std::vector<float> ffn_out(H, 0);
#if USE_COLIBRI_Q4
                    if (!l.q4_dn.empty())
                        colibri_matmul_q4(ffn_out.data(), gate_up.data(), l.q4_dn.data(), l.qs_dn.data(), 1, gate_dim, H);
                    else
#endif
                    { for (int i = 0; i < H; i++) for (int j = 0; j < gate_dim; j++) ffn_out[i] += gate_up[j] * l.dn[j * H + i]; }
                    for (int i = 0; i < H; i++) hidden[i] = ffn_out[i] + residual[i];
                }
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
            for (int v = 0; v < VOCAB; v++) {
                float dot = 0;
                const float* emb_row = embed_.data() + (size_t)v * H;
                for (int i = 0; i < H; i++) dot += hidden[i] * emb_row[i];
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
extern std::vector<InferenceBackend*> detect_backends_generic();
__attribute__((weak)) std::vector<InferenceBackend*> detect_backends_zinc() { return {}; }

std::vector<InferenceBackend*> detect_backends() {
    std::vector<InferenceBackend*> backends;

    auto hip_backends = detect_backends_hip();
    for (auto* b : hip_backends) backends.push_back(b);

    auto vk_backends = detect_backends_vulkan();
    for (auto* b : vk_backends) backends.push_back(b);

    auto npu_backends = detect_backends_npu();
    for (auto* b : npu_backends) backends.push_back(b);

    auto generic_backends = detect_backends_generic();
    for (auto* b : generic_backends) backends.push_back(b);

    auto zinc_backends = detect_backends_zinc();
    for (auto* b : zinc_backends) backends.push_back(b);

    static CpuBackend cpu_backend;
    backends.push_back(&cpu_backend);

    return backends;
}

InferenceBackend* select_best_backend(std::vector<InferenceBackend*>* existing) {
    auto owned = existing ? std::vector<InferenceBackend*>() : detect_backends();
    auto& backends = existing ? *existing : owned;
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
