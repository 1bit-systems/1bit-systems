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

// Runtime state: minimal — no KV cache needed (single-position cross-head attention).
// Kept for API compatibility with the spec decode engine's StateResizeTraits.
struct MTPDraftState {
    int32_t seq_len = 0;

    void resize(int32_t /*num_kv_heads*/, int32_t /*head_dim*/, int32_t /*max_len*/) {
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

    // Matches Python train_from_cache.py Eagle3Draft.forward() exactly:
    //   h = hidden_norm(fc(feat))          # Pos 0: project target features
    //   for t in range(T):
    //     x = cat([h, embed_normed])       # Concatenate hidden + token embedding
    //     q,k,v = proj(x); attn = cross-head(q,k,v)  # No KV cache, cross-head only
    //     h = h + attn; h = post_norm(h); h = h + swiglu(h)
    //     h = norm(h)                      # Final norm for next step
    //     logits = lm_head(h)
    // Key: at pos>0, h already has norm() applied; no hidden_norm() re-application.
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
            for (int i = 0; i < H; i++) draft_hidden[i] = trunk_hidden[i];
            draft_logits[0] = draft_hidden[0];
            return;
        }

        // h: the autoregressive hidden state. At pos 0, project from target features.
        // At pos > 0, trunk_hidden is draft_hidden from previous step (already norm'ed).
        std::vector<float> h(H);
        if (pos == 0) {
            linear(trunk_hidden, w_.fc.data(), h.data(), cfg_.num_target_layers * H, H);
            rms_norm(h.data(), h.data(), w_.hidden_norm.data(), H);
        } else {
            std::copy(trunk_hidden, trunk_hidden + H, h.begin());
        }

        std::vector<float> embed(H);
        const float* erow = &w_.embed_tokens[(size_t)input_id * H];
        std::copy(erow, erow + H, embed.begin());
        rms_norm(embed.data(), embed.data(), w_.input_layernorm.data(), H);

        // x = cat([h, embed_normed]) — matches Python: cat([h.unsqueeze(1), e], dim=-1)
        std::vector<float> x(2 * H);
        std::copy(h.begin(), h.end(), x.begin());
        std::copy(embed.begin(), embed.end(), x.begin() + H);

        // Attention + add: h = h + attn_out  (Python: h + o)
        std::vector<float> attn_out(H);
        self_attention(x.data(), pos, attn_out.data());
        for (int i = 0; i < H; i++) h[i] += attn_out[i];

        // Post-attention norm: h2 = post_attn_norm(h + o)
        std::vector<float> h2(H);
        rms_norm(h.data(), h2.data(), w_.post_attention_layernorm.data(), H);

        // FFN on normed value: ffn_out = swiglu(h2)
        std::vector<float> ffn_out(H);
        swiglu_ffn(h2.data(), ffn_out.data());

        // h3 = h2 + ffn_out (Python: h3 = h2 + swiglu(h2))
        std::vector<float> h3(H);
        for (int i = 0; i < H; i++) h3[i] = h2[i] + ffn_out[i];

        // Final norm: h = norm(h3). This norm'ed state feeds the next step.
        std::vector<float> final_n(H);
        rms_norm(h3.data(), final_n.data(), w_.norm.data(), H);
        std::copy(final_n.begin(), final_n.end(), draft_hidden);

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

    // Single-position cross-head attention. No KV cache — matches Python training:
    //   q = q_norm(rope(q_proj(x)))   # [NH, D]
    //   k = k_norm(rope(k_proj(x)))   # [NKV, D]
    //   v = v_proj(x)                 # [NKV, D]
    //   attn = softmax(q @ k^T * scale, dim=-1)  # [NH, NKV]
    //   out = o_proj(attn @ v)        # [H]
    void self_attention(const float* x /* [2*hidden] */, int pos, float* out) {
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

        float scale = 1.0f / std::sqrt((float)D);

        // q @ k^T: [NH, D] x [D, NKV] = [NH, NKV]
        // Each of NH query heads attends to all NKV key heads at THIS position only.
        std::vector<float> scores(NH * NKV);
        for (int h = 0; h < NH; h++) {
            for (int kvh = 0; kvh < NKV; kvh++) {
                double s = 0.0;
                const float* qrow = &q[(size_t)h * D];
                const float* krow = &k[(size_t)kvh * D];
                for (int d = 0; d < D; d++) s += (double)qrow[d] * krow[d];
                scores[(size_t)h * NKV + kvh] = (float)(s * scale);
            }
        }

        // Softmax per query head (over NKV dimension)
        std::vector<float> attn_weights(NH * NKV);
        for (int h = 0; h < NH; h++) {
            float* row = &scores[(size_t)h * NKV];
            float mx = row[0];
            for (int i = 1; i < NKV; i++) if (row[i] > mx) mx = row[i];
            double sum = 0.0;
            for (int i = 0; i < NKV; i++) { row[i] = std::exp(row[i] - mx); sum += row[i]; }
            float inv_sum = (float)(1.0 / sum);
            float* arow = &attn_weights[(size_t)h * NKV];
            for (int i = 0; i < NKV; i++) arow[i] = row[i] * inv_sum;
        }

        // attn_weights @ v: [NH, NKV] x [NKV, D] = [NH, D]
        std::vector<float> attn_out(NH * D, 0.0f);
        for (int h = 0; h < NH; h++) {
            const float* arow = &attn_weights[(size_t)h * NKV];
            for (int kvh = 0; kvh < NKV; kvh++) {
                float w = arow[kvh];
                const float* vrow = &v[(size_t)kvh * D];
                float* aout = &attn_out[(size_t)h * D];
                for (int d = 0; d < D; d++) aout[d] += w * vrow[d];
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
