#pragma once
// MTP Draft Model — Eagle3-style draft head for XDNA 2 NPU, matching DeepSpec's
// Qwen3Eagle3Model architecture exactly (deepspec/modeling/eagle3/qwen3/modeling.py)
// so trained checkpoints load and run correctly. hidden=1024, heads=16, KV=8,
// head_dim=128, 1 draft decoder layer.
//
// Forward pass (matches modeling.py precisely):
//   1. hidden = fc(concat(target_layer_hidden[0..4]))            // 5120 -> 1024
//   2. embed = embed_tokens(input_id)                            // 1024
//   3. per layer: residual=hidden
//        h_n = hidden_norm(hidden); e_n = input_layernorm(embed)
//        x = concat(e_n, h_n)                                    // 2048
//        attn = self_attn(x)  (q/k_norm + RoPE "rotate_half" + causal attn + o_proj)
//        hidden = residual + attn
//        residual = hidden
//        hidden = post_attention_layernorm(hidden)
//        hidden = residual + swiglu_mlp(hidden)
//   4. logits = lm_head(norm(hidden))

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

struct MTPDraftConfig {
    int32_t hidden_size = 1024;
    int32_t num_heads = 16;
    int32_t num_kv_heads = 8;
    int32_t head_dim = 128;
    int32_t vocab_size = 151936;
    int32_t block_size = 7;          // N speculative tokens per forward
    int32_t num_target_layers = 5;   // target layer features to fuse
    int32_t inter_dim = 3072;        // FFN intermediate (same as trunk)
    int32_t max_seq = 4096;          // Max sequence length
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
};

// Matches the exact tensor names/shapes in a trained DeepSpec Eagle3 checkpoint
// (model.safetensors). See scripts_local/export_draft_weights.py for the writer.
struct MTPDraftWeights {
    std::vector<float> embed_tokens;   // [vocab, hidden]
    std::vector<float> fc;             // [hidden, num_target_layers*hidden] (row-major, out x in)
    std::vector<float> hidden_norm;    // [hidden]
    std::vector<float> input_layernorm;// [hidden]
    std::vector<float> q_proj;         // [num_heads*head_dim, 2*hidden]
    std::vector<float> k_proj;         // [num_kv_heads*head_dim, 2*hidden]
    std::vector<float> v_proj;         // [num_kv_heads*head_dim, 2*hidden]
    std::vector<float> o_proj;         // [hidden, num_heads*head_dim]
    std::vector<float> q_norm;         // [head_dim]
    std::vector<float> k_norm;         // [head_dim]
    std::vector<float> post_attention_layernorm; // [hidden]
    std::vector<float> gate_proj;      // [inter_dim, hidden]
    std::vector<float> up_proj;        // [inter_dim, hidden]
    std::vector<float> down_proj;      // [hidden, inter_dim]
    std::vector<float> norm;           // [hidden]
    std::vector<float> lm_head;        // [vocab, hidden]

    bool empty() const { return fc.empty(); }

    // Loads the flat binary dump written by export_draft_weights.py — a fixed-order
    // sequential concatenation of float32 arrays matching the fields above exactly.
    bool load(const char* path, const MTPDraftConfig& cfg) {
        FILE* f = fopen(path, "rb");
        if (!f) return false;
        auto read_vec = [&](std::vector<float>& v, size_t n) {
            v.resize(n);
            size_t got = fread(v.data(), sizeof(float), n, f);
            return got == n;
        };
        int H = cfg.hidden_size, V = cfg.vocab_size, NH = cfg.num_heads, NKV = cfg.num_kv_heads;
        int D = cfg.head_dim, IM = cfg.inter_dim, NTL = cfg.num_target_layers;
        bool ok = true;
        ok &= read_vec(embed_tokens, (size_t)V * H);
        ok &= read_vec(fc, (size_t)H * NTL * H);
        ok &= read_vec(hidden_norm, H);
        ok &= read_vec(input_layernorm, H);
        ok &= read_vec(q_proj, (size_t)(NH * D) * (2 * H));
        ok &= read_vec(k_proj, (size_t)(NKV * D) * (2 * H));
        ok &= read_vec(v_proj, (size_t)(NKV * D) * (2 * H));
        ok &= read_vec(o_proj, (size_t)H * (NH * D));
        ok &= read_vec(q_norm, D);
        ok &= read_vec(k_norm, D);
        ok &= read_vec(post_attention_layernorm, H);
        ok &= read_vec(gate_proj, (size_t)IM * H);
        ok &= read_vec(up_proj, (size_t)IM * H);
        ok &= read_vec(down_proj, (size_t)H * IM);
        ok &= read_vec(norm, H);
        ok &= read_vec(lm_head, (size_t)V * H);
        fclose(f);
        return ok;
    }
};

// Runtime state: single KV cache entry for the draft block
struct MTPDraftState {
    std::vector<float> k_cache; // [max_seq, num_kv_heads, head_dim]
    std::vector<float> v_cache; // [max_seq, num_kv_heads, head_dim]
    int32_t seq_len = 0;
    int32_t max_seq = 4096;

    void resize(int32_t num_kv_heads, int32_t head_dim, int32_t max_len) {
        max_seq = max_len;
        k_cache.assign((size_t)num_kv_heads * max_len * head_dim, 0.0f);
        v_cache.assign((size_t)num_kv_heads * max_len * head_dim, 0.0f);
        seq_len = 0;
    }
};

class MTPDraftModel {
public:
    MTPDraftModel(const MTPDraftConfig& cfg) : cfg_(cfg) {
        int half = cfg_.head_dim / 2;
        inv_freq_.resize(half);
        for (int i = 0; i < half; i++)
            inv_freq_[i] = 1.0f / std::pow(cfg_.rope_theta, (2.0f * i) / cfg_.head_dim);
    }

    bool load_weights(const char* path) { return w_.load(path, cfg_); }
    bool weights_loaded() const { return !w_.empty(); }

    // trunk_hidden: [num_target_layers, hidden_size] fp32 features from the target model
    // (already fp32-converted by the caller). input_id: current token id (embedding is
    // looked up internally). draft_logits: [vocab_size] output. draft_hidden: [hidden_size]
    // output (fed back as trunk_hidden[0] substitute isn't needed — draft re-derives from
    // target features each call in this integration's usage pattern).
    void forward(
        const float* trunk_hidden,
        int32_t input_id,
        int32_t pos,
        MTPDraftState& state,
        float* draft_logits,
        float* draft_hidden
    ) {
        int H = cfg_.hidden_size;
        if (w_.empty()) {
            // Fast path: no trained weights yet — passthrough so callers/tests don't crash.
            for (int i = 0; i < H; i++) draft_hidden[i] = trunk_hidden[i];
            draft_logits[0] = draft_hidden[0];
            return;
        }

        std::vector<float> hidden(H);
        linear(trunk_hidden, w_.fc.data(), hidden.data(), cfg_.num_target_layers * H, H);

        std::vector<float> embed(H);
        const float* erow = &w_.embed_tokens[(size_t)input_id * H];
        std::copy(erow, erow + H, embed.begin());

        std::vector<float> residual(hidden);
        std::vector<float> h_n(H), e_n(H);
        rms_norm(hidden.data(), h_n.data(), w_.hidden_norm.data(), H);
        rms_norm(embed.data(), e_n.data(), w_.input_layernorm.data(), H);

        std::vector<float> x(2 * H);
        std::copy(e_n.begin(), e_n.end(), x.begin());
        std::copy(h_n.begin(), h_n.end(), x.begin() + H);

        std::vector<float> attn_out(H);
        self_attention(x.data(), pos, state, attn_out.data());
        for (int i = 0; i < H; i++) hidden[i] = residual[i] + attn_out[i];

        residual = hidden;
        std::vector<float> post_n(H);
        rms_norm(hidden.data(), post_n.data(), w_.post_attention_layernorm.data(), H);
        std::vector<float> mlp_out(H);
        swiglu_ffn(post_n.data(), mlp_out.data());
        for (int i = 0; i < H; i++) hidden[i] = residual[i] + mlp_out[i];

        std::copy(hidden.begin(), hidden.end(), draft_hidden);

        std::vector<float> final_n(H);
        rms_norm(hidden.data(), final_n.data(), w_.norm.data(), H);
        linear(final_n.data(), w_.lm_head.data(), draft_logits, H, cfg_.vocab_size);
    }

private:
    MTPDraftConfig cfg_;
    MTPDraftWeights w_;
    std::vector<float> inv_freq_;

    void rms_norm(const float* x, float* y, const float* weight, int n) {
        double ss = 0.0;
        for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
        float inv = 1.0f / std::sqrt((float)(ss / n) + cfg_.rms_norm_eps);
        for (int i = 0; i < n; i++) y[i] = weight[i] * (x[i] * inv);
    }

    // y = W @ x, W stored row-major [out_dim, in_dim] (PyTorch nn.Linear.weight layout)
    void linear(const float* x, const float* W, float* y, int in_dim, int out_dim) {
        for (int o = 0; o < out_dim; o++) {
            double s = 0.0;
            const float* row = W + (size_t)o * in_dim;
            for (int i = 0; i < in_dim; i++) s += (double)row[i] * x[i];
            y[o] = (float)s;
        }
    }

    // HuggingFace "rotate_half" RoPE convention: pairs (i, i+head_dim/2) rotate together,
    // NOT adjacent pairs (i, i+1) — different from the NPU engine's interleaved convention,
    // because these weights are trained straight from HF transformers with no repacking.
    void apply_rope(float* x, int head_dim, int pos) {
        int half = head_dim / 2;
        std::vector<float> orig(x, x + head_dim);
        for (int i = 0; i < half; i++) {
            float angle = pos * inv_freq_[i];
            float c = std::cos(angle), s = std::sin(angle);
            x[i] = orig[i] * c - orig[i + half] * s;
            x[i + half] = orig[i + half] * c + orig[i] * s;
        }
    }

    void self_attention(const float* x /* [2*hidden] */, int pos, MTPDraftState& state, float* out) {
        int H = cfg_.hidden_size, NH = cfg_.num_heads, NKV = cfg_.num_kv_heads, D = cfg_.head_dim;
        std::vector<float> q(NH * D), k(NKV * D), v(NKV * D);
        linear(x, w_.q_proj.data(), q.data(), 2 * H, NH * D);
        linear(x, w_.k_proj.data(), k.data(), 2 * H, NKV * D);
        linear(x, w_.v_proj.data(), v.data(), 2 * H, NKV * D);

        for (int h = 0; h < NH; h++) {
            rms_norm(&q[h * D], &q[h * D], w_.q_norm.data(), D);
            apply_rope(&q[h * D], D, pos);
        }
        for (int h = 0; h < NKV; h++) {
            rms_norm(&k[h * D], &k[h * D], w_.k_norm.data(), D);
            apply_rope(&k[h * D], D, pos);
        }

        for (int h = 0; h < NKV; h++) {
            std::copy(&k[h * D], &k[h * D] + D, &state.k_cache[((size_t)pos * NKV + h) * D]);
            std::copy(&v[h * D], &v[h * D] + D, &state.v_cache[((size_t)pos * NKV + h) * D]);
        }
        state.seq_len = pos + 1;
        int gqa = NH / NKV;
        int ctx = pos + 1;

        std::vector<float> attn_out(NH * D, 0.0f);
        std::vector<float> scores(ctx);
        for (int h = 0; h < NH; h++) {
            int kvh = h / gqa;
            for (int t = 0; t < ctx; t++) {
                double s = 0.0;
                const float* krow = &state.k_cache[((size_t)t * NKV + kvh) * D];
                for (int d = 0; d < D; d++) s += (double)q[h * D + d] * krow[d];
                scores[t] = (float)(s / std::sqrt((float)D));
            }
            float mx = *std::max_element(scores.begin(), scores.end());
            double sum = 0.0;
            for (int t = 0; t < ctx; t++) { scores[t] = std::exp(scores[t] - mx); sum += scores[t]; }
            float inv_sum = (float)(1.0 / sum);
            for (int t = 0; t < ctx; t++) {
                float p = scores[t] * inv_sum;
                const float* vrow = &state.v_cache[((size_t)t * NKV + kvh) * D];
                for (int d = 0; d < D; d++) attn_out[h * D + d] += p * vrow[d];
            }
        }
        linear(attn_out.data(), w_.o_proj.data(), out, NH * D, H);
    }

    void swiglu_ffn(const float* x, float* out) {
        int H = cfg_.hidden_size, IM = cfg_.inter_dim;
        std::vector<float> gate(IM), up(IM);
        linear(x, w_.gate_proj.data(), gate.data(), H, IM);
        linear(x, w_.up_proj.data(), up.data(), H, IM);
        std::vector<float> act(IM);
        for (int i = 0; i < IM; i++) act[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
        linear(act.data(), w_.down_proj.data(), out, IM, H);
    }
};
