#pragma once
// DSpark Draft Model — 5-layer speculative decoding draft head with Markov
// correction and confidence prediction, matching DeepSpec's Qwen3DSparkModel
// architecture exactly (deepspec/modeling/dspark/qwen3/modeling.py).
//
// Architecture differences from Eagle3 (mtp_draft.h):
//   1. 5 decoder layers (vs 1) — each with Q/K/V on hidden_size (not 2*hidden)
//   2. Cross-attention to projected target features (trunk hidden → fc → norm)
//   3. Vanilla Markov head: embed(prev_token) → W1[V,R] → W2[R,V] logit bias
//   4. Confidence head: sigmoid(linear(concat(hidden, W1(prev_token))))
//
// Supported configurations:
//   Qwen3-0.6B: hidden=1024, heads=16, KV=8, head_dim=128, inter_dim=3072
//   Qwen3-4B:   hidden=2560, heads=20, KV=20, head_dim=128, inter_dim=9216
//
// Weight binary layout (sequential float32, written by export_dspark_weights.py):
//   1.  embed_tokens:           [vocab, hidden]
//   2.  fc:                     [hidden, num_target_layers*hidden]
//   3.  hidden_norm:            [hidden]
//   4.  layers.0.input_layernorm:  [hidden]
//   5.  layers.0.q_proj:          [num_heads*head_dim, hidden]
//   6.  layers.0.k_proj:          [num_kv_heads*head_dim, hidden]
//   7.  layers.0.v_proj:          [num_kv_heads*head_dim, hidden]
//   8.  layers.0.o_proj:          [hidden, num_heads*head_dim]
//   9.  layers.0.q_norm:          [head_dim]
//   10. layers.0.k_norm:          [head_dim]
//   11. layers.0.post_attention_layernorm: [hidden]
//   12. layers.0.gate_proj:       [inter_dim, hidden]
//   13. layers.0.up_proj:         [inter_dim, hidden]
//   14. layers.0.down_proj:       [hidden, inter_dim]
//   ... repeats for layers 1..4 (same field order)
//   15. norm:                   [hidden]
//   16. lm_head:                [vocab, hidden]
//   17. markov_w1:              [vocab, markov_rank]
//   18. markov_w2:              [vocab, markov_rank]  (Linear weight: out=vocab, in=rank)
//   19. confidence_proj.weight: [1, hidden + markov_rank]
//   20. confidence_proj.bias:   [1]
//
// Forward pass (matches inference path of Qwen3DSparkModel.forward() exactly):
//   1. target_features = hidden_norm(fc(trunk_hidden))              // cached once per round
//   2. noise_embed = embed_tokens(input_id)
//   3. For each of 5 layers:
//        residual = hidden
//        h = input_layernorm(hidden)
//        q = q_proj(h);  k = concat(k_ctx, k_proj(h));  v = concat(v_ctx, v_proj(h))
//        rms_norm(q, q_norm);  rms_norm(k, k_norm);  apply_rope(q, k)
//        Cache k_noise, v_noise
//        Attention: softmax(Q @ K_cache^T / sqrt(d)) @ V_cache
//        hidden = residual + o_proj(attn)
//        residual = hidden
//        hidden = post_attention_layernorm(hidden)
//        hidden = residual + swiglu(hidden)
//   4. final_hidden = norm(hidden)
//   5. base_logits = lm_head(final_hidden)
//   6. Markov correction: draft_logits = base_logits + W2 @ W1(prev_token)
//   7. Confidence: sigmoid(proj(concat(final_hidden, W1(prev_token))))

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct DSparkDraftConfig {
    int32_t hidden_size = 1024;          // 1024 (0.6B) or 2560 (4B)
    int32_t num_heads = 16;              // 16 (0.6B) or 20 (4B)
    int32_t num_kv_heads = 8;            // 8 (0.6B) or 20 (4B, full GQA)
    int32_t head_dim = 128;
    int32_t vocab_size = 151936;
    int32_t block_size = 7;              // N speculative tokens per forward
    int32_t num_target_layers = 5;       // target layer features to fuse
    int32_t num_draft_layers = 5;        // DSpark draft backbone depth
    int32_t inter_dim = 3072;            // FFN intermediate (3072 for 0.6B, 9216 for 4B)
    int32_t max_seq = 4096;              // Max sequence length
    int32_t markov_rank = 128;           // Markov head rank
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
};

// ---------------------------------------------------------------------------
// Weights — all float32 tensors in flat binary order
// ---------------------------------------------------------------------------

struct DSparkDraftWeights {
    // Shared (target model sized)
    std::vector<float> embed_tokens;   // [vocab, hidden]
    std::vector<float> fc;             // [hidden, num_target_layers*hidden] (out x in)
    std::vector<float> hidden_norm;    // [hidden]

    // Per-layer weights (num_draft_layers copies)
    std::vector<float> input_layernorm;        // [num_layers, hidden]
    std::vector<float> q_proj;                 // [num_layers, num_heads*head_dim, hidden]
    std::vector<float> k_proj;                 // [num_layers, num_kv_heads*head_dim, hidden]
    std::vector<float> v_proj;                 // [num_layers, num_kv_heads*head_dim, hidden]
    std::vector<float> o_proj;                 // [num_layers, hidden, num_heads*head_dim]
    std::vector<float> q_norm;                 // [num_layers, head_dim]
    std::vector<float> k_norm;                 // [num_layers, head_dim]
    std::vector<float> post_attention_layernorm; // [num_layers, hidden]
    std::vector<float> gate_proj;              // [num_layers, inter_dim, hidden]
    std::vector<float> up_proj;                // [num_layers, inter_dim, hidden]
    std::vector<float> down_proj;              // [num_layers, hidden, inter_dim]

    // Final norm + output
    std::vector<float> norm;           // [hidden]
    std::vector<float> lm_head;        // [vocab, hidden]

    // Markov head
    std::vector<float> markov_w1;      // [vocab, markov_rank] — embedding table
    std::vector<float> markov_w2;      // [vocab, markov_rank] — Linear weight (out=V, in=R)

    // Confidence head
    std::vector<float> confidence_weight; // [1, hidden + markov_rank]
    std::vector<float> confidence_bias;   // [1]

    bool empty() const { return fc.empty(); }

    // Layout accessor: returns pointer to start of layer l's weight block
    const float* layer_weight(const std::vector<float>& all, int l, int per_layer) const {
        return all.data() + (size_t)l * per_layer;
    }
    float* layer_weight(std::vector<float>& all, int l, int per_layer) {
        return all.data() + (size_t)l * per_layer;
    }

    // Load the flat binary dump — a fixed-order sequential concatenation of
    // float32 arrays matching the fields declared above, in exactly this order.
    bool load(const char* path, const DSparkDraftConfig& cfg) {
        FILE* f = fopen(path, "rb");
        if (!f) return false;

        auto read_vec = [&](std::vector<float>& v, size_t n) {
            v.resize(n);
            size_t got = fread(v.data(), sizeof(float), n, f);
            return got == n;
        };

        int H = cfg.hidden_size, V = cfg.vocab_size;
        int NH = cfg.num_heads, NKV = cfg.num_kv_heads;
        int D = cfg.head_dim, IM = cfg.inter_dim;
        int NTL = cfg.num_target_layers, NL = cfg.num_draft_layers;
        int R = cfg.markov_rank;

        // Per-layer element counts
        size_t per_in_ln  = (size_t)H;
        size_t per_q      = (size_t)(NH * D) * H;
        size_t per_k      = (size_t)(NKV * D) * H;
        size_t per_v      = (size_t)(NKV * D) * H;
        size_t per_o      = (size_t)H * (NH * D);
        size_t per_qn     = (size_t)D;
        size_t per_kn     = (size_t)D;
        size_t per_pa_ln  = (size_t)H;
        size_t per_gate   = (size_t)IM * H;
        size_t per_up     = (size_t)IM * H;
        size_t per_down   = (size_t)H * IM;

        bool ok = true;
        // 1. Shared
        ok &= read_vec(embed_tokens,  (size_t)V * H);
        ok &= read_vec(fc,            (size_t)H * NTL * H);
        ok &= read_vec(hidden_norm,   H);
        // 2. Layers (interleaved: all copies of each field stored together for cleaner layout)
        //    Within each field: layer 0's data, then layer 1's, ..., layer NL-1's
        ok &= read_vec(input_layernorm,          (size_t)NL * per_in_ln);
        ok &= read_vec(q_proj,                  (size_t)NL * per_q);
        ok &= read_vec(k_proj,                  (size_t)NL * per_k);
        ok &= read_vec(v_proj,                  (size_t)NL * per_v);
        ok &= read_vec(o_proj,                  (size_t)NL * per_o);
        ok &= read_vec(q_norm,                  (size_t)NL * per_qn);
        ok &= read_vec(k_norm,                  (size_t)NL * per_kn);
        ok &= read_vec(post_attention_layernorm, (size_t)NL * per_pa_ln);
        ok &= read_vec(gate_proj,               (size_t)NL * per_gate);
        ok &= read_vec(up_proj,                 (size_t)NL * per_up);
        ok &= read_vec(down_proj,               (size_t)NL * per_down);
        // 3. Final norm + output
        ok &= read_vec(norm,           H);
        ok &= read_vec(lm_head,        (size_t)V * H);
        // 4. Markov head
        ok &= read_vec(markov_w1,      (size_t)V * R);
        ok &= read_vec(markov_w2,      (size_t)V * R);
        // 5. Confidence head
        ok &= read_vec(confidence_weight, (size_t)(H + R));
        ok &= read_vec(confidence_bias,   1);

        fclose(f);
        return ok;
    }

    // Validate weight shapes against config
    bool validate(const DSparkDraftConfig& cfg) const {
        int H = cfg.hidden_size, V = cfg.vocab_size;
        int NH = cfg.num_heads, NKV = cfg.num_kv_heads;
        int D = cfg.head_dim, IM = cfg.inter_dim;
        int NTL = cfg.num_target_layers, NL = cfg.num_draft_layers;
        int R = cfg.markov_rank;

        auto check = [&](const std::vector<float>& v, size_t expected, const char* name) {
            if (v.size() != expected) {
                fprintf(stderr, "DSpark weights validate: %s expected %zu elements, got %zu\n",
                        name, expected, v.size());
                return false;
            }
            return true;
        };

        return
            check(embed_tokens,  (size_t)V * H, "embed_tokens") &&
            check(fc,            (size_t)H * NTL * H, "fc") &&
            check(hidden_norm,   H, "hidden_norm") &&
            check(input_layernorm,          (size_t)NL * H, "input_layernorm") &&
            check(q_proj,                  (size_t)NL * NH * D * H, "q_proj") &&
            check(k_proj,                  (size_t)NL * NKV * D * H, "k_proj") &&
            check(v_proj,                  (size_t)NL * NKV * D * H, "v_proj") &&
            check(o_proj,                  (size_t)NL * H * NH * D, "o_proj") &&
            check(q_norm,                  (size_t)NL * D, "q_norm") &&
            check(k_norm,                  (size_t)NL * D, "k_norm") &&
            check(post_attention_layernorm, (size_t)NL * H, "post_attention_layernorm") &&
            check(gate_proj,               (size_t)NL * IM * H, "gate_proj") &&
            check(up_proj,                 (size_t)NL * IM * H, "up_proj") &&
            check(down_proj,               (size_t)NL * H * IM, "down_proj") &&
            check(norm,           H, "norm") &&
            check(lm_head,        (size_t)V * H, "lm_head") &&
            check(markov_w1,      (size_t)V * R, "markov_w1") &&
            check(markov_w2,      (size_t)V * R, "markov_w2") &&
            check(confidence_weight, (size_t)(H + R), "confidence_weight") &&
            check(confidence_bias, 1, "confidence_bias");
    }
};

// ---------------------------------------------------------------------------
// Runtime state: per-layer KV caches for the draft backbone
// ---------------------------------------------------------------------------

struct DSparkDraftState {
    // Per-layer KV cache entries for draft positions (noise tokens only).
    // Target features' K/V are re-projected each round (stored in the model).
    // Each cache grows as draft positions accumulate.
    std::vector<std::vector<float>> k_cache;  // [num_layers][max_draft * num_kv_heads * head_dim]
    std::vector<std::vector<float>> v_cache;  // [num_layers][max_draft * num_kv_heads * head_dim]
    int32_t num_layers = 0;
    int32_t seq_len = 0;         // number of draft positions cached so far
    int32_t max_draft = 7;       // max draft positions to cache (block_size)

    void resize(int32_t num_draft_layers, int32_t num_kv_heads, int32_t head_dim, int32_t max_draft_len) {
        num_layers = num_draft_layers;
        max_draft = max_draft_len;
        size_t per_layer = (size_t)num_kv_heads * max_draft_len * head_dim;
        k_cache.resize(num_draft_layers);
        v_cache.resize(num_draft_layers);
        for (int l = 0; l < num_draft_layers; l++) {
            k_cache[l].assign(per_layer, 0.0f);
            v_cache[l].assign(per_layer, 0.0f);
        }
        seq_len = 0;
    }

    // Write a K,V pair for the given layer at the current seq_len position
    void write_kv(int layer, int num_kv_heads, int head_dim,
                  const float* k, const float* v) {
        size_t offset = (size_t)seq_len * num_kv_heads * head_dim;
        std::copy(k, k + (size_t)num_kv_heads * head_dim, k_cache[layer].begin() + offset);
        std::copy(v, v + (size_t)num_kv_heads * head_dim, v_cache[layer].begin() + offset);
    }

    // Read K,V for all positions up to seq_len+1 (for attention)
    const float* k_ptr(int layer) const { return k_cache[layer].data(); }
    const float* v_ptr(int layer) const { return v_cache[layer].data(); }

    void reset() { seq_len = 0; }

    // Target-side KV precomputation context (recomputed per round)
    std::vector<float> ctx_k;  // [num_layers, num_kv_heads * ctx_seq_len * head_dim]
    std::vector<float> ctx_v;  // [num_layers, num_kv_heads * ctx_seq_len * head_dim]
    int32_t ctx_seq_len = 0;   // 1 unless using full context (always 1 in current integration)
};

// ---------------------------------------------------------------------------
// DSpark Draft Model — main class
// ---------------------------------------------------------------------------

class DSparkDraftModel {
public:
    DSparkDraftModel(const DSparkDraftConfig& cfg)
        : cfg_(cfg) {
        int half = cfg_.head_dim / 2;
        inv_freq_.resize(half);
        for (int i = 0; i < half; i++)
            inv_freq_[i] = 1.0f / std::pow(cfg_.rope_theta, (2.0f * i) / cfg_.head_dim);
    }

    bool load_weights(const char* path) {
        bool ok = w_.load(path, cfg_);
        if (ok) {
            ok = w_.validate(cfg_);
            if (ok) {
                // Pre-compute target feature projections to avoid re-projection
                // on every draft step. The fc projection is cached per round.
                cached_target_features_.resize(cfg_.hidden_size);
                cached_round_id_ = -1;
            }
        }
        return ok;
    }

    bool weights_loaded() const { return !w_.empty(); }
    const DSparkDraftConfig& config() const { return cfg_; }

    // ---- Forward pass (drop-in compatible with MTPDraftModel interface) ----
    //
    // trunk_hidden: [num_target_layers * hidden_size] fp32 features from the target
    //   model's selected layers, concatenated along the last dimension.
    // input_id: current token ID (embedded internally).
    // pos: position within the current speculative round (0 = first draft token).
    // state: per-layer KV cache for the draft backbone.
    // draft_logits: [vocab_size] output logits (with Markov correction applied).
    // draft_hidden: [hidden_size] final backbone hidden state (pre-norm).
    // confidence_out: [1] confidence score (sigmoid), may be nullptr.
    //
    void forward(
        const float* trunk_hidden,
        int32_t input_id,
        int32_t pos,
        DSparkDraftState& state,
        float* draft_logits,
        float* draft_hidden,
        float* confidence_out = nullptr
    ) {
        if (w_.empty()) {
            // No weights loaded — passthrough for testing
            int H = cfg_.hidden_size;
            for (int i = 0; i < H && i < cfg_.num_target_layers * cfg_.hidden_size; i++)
                draft_hidden[i] = trunk_hidden[i];
            draft_logits[0] = draft_hidden[0];
            if (confidence_out) *confidence_out = 0.5f;
            return;
        }

        int H = cfg_.hidden_size;
        int NTL = cfg_.num_target_layers;
        int NL = cfg_.num_draft_layers;
        int V = cfg_.vocab_size;
        int R = cfg_.markov_rank;

        // ---- 1. On new round: project and cache target features ----
        if (pos == 0) {
            // Project trunk features: fc projection + hidden_norm
            linear(trunk_hidden, w_.fc.data(), cached_target_features_.data(), NTL * H, H);
            rms_norm(cached_target_features_.data(), cached_target_features_.data(),
                     w_.hidden_norm.data(), H);

            // Pre-compute target side K/V for all layers (ctx_len=1 — single position)
            state.ctx_seq_len = 1;
            size_t ctx_kv_per = (size_t)cfg_.num_kv_heads * cfg_.head_dim;
            state.ctx_k.resize((size_t)NL * ctx_kv_per);
            state.ctx_v.resize((size_t)NL * ctx_kv_per);

            for (int l = 0; l < NL; l++) {
                // k_ctx = k_proj(target_features)
                linear(cached_target_features_.data(),
                       layer_w(w_.k_proj, l, (size_t)cfg_.num_kv_heads * H * cfg_.head_dim, H),
                       &state.ctx_k[(size_t)l * ctx_kv_per],
                       H, cfg_.num_kv_heads * cfg_.head_dim);
                // v_ctx = v_proj(target_features)
                linear(cached_target_features_.data(),
                       layer_w(w_.v_proj, l, (size_t)cfg_.num_kv_heads * H * cfg_.head_dim, H),
                       &state.ctx_v[(size_t)l * ctx_kv_per],
                       H, cfg_.num_kv_heads * cfg_.head_dim);
                // Apply k_norm + RoPE to K_ctx at position 0
                rms_norm(&state.ctx_k[(size_t)l * ctx_kv_per],
                         &state.ctx_k[(size_t)l * ctx_kv_per],
                         layer_w(w_.k_norm, l, cfg_.head_dim), cfg_.num_kv_heads * cfg_.head_dim);
                for (int h = 0; h < cfg_.num_kv_heads; h++)
                    apply_rope(&state.ctx_k[(size_t)l * ctx_kv_per + (size_t)h * cfg_.head_dim],
                               cfg_.head_dim, 0);
            }

            // Reset draft KV cache
            state.reset();
        }

        // ---- 2. Embed current input token ----
        std::vector<float> hidden(H);
        {
            const float* erow = &w_.embed_tokens[(size_t)input_id * H];
            std::copy(erow, erow + H, hidden.begin());
        }

        // ---- 3. Run through all draft decoder layers ----
        int NH = cfg_.num_heads, NKV = cfg_.num_kv_heads, D = cfg_.head_dim;
        int GQA = NH / NKV;
        int IM = cfg_.inter_dim;

        for (int l = 0; l < NL; l++) {
            int per_layer_q = NH * D * H;
            int per_layer_kv = NKV * D * H;
            int per_layer_o = H * NH * D;
            int per_layer_gate = IM * H;
            int per_layer_up = IM * H;
            int per_layer_down = H * IM;
            int per_layer_h = H;

            std::vector<float> residual = hidden;

            // input_layernorm
            rms_norm(hidden.data(), hidden.data(),
                     layer_w(w_.input_layernorm, l, per_layer_h), H);

            // Q = q_proj(hidden)
            std::vector<float> q(NH * D);
            linear(hidden.data(), layer_w(w_.q_proj, l, per_layer_q, H),
                   q.data(), H, NH * D);

            // K_noise = k_proj(hidden), V_noise = v_proj(hidden)
            std::vector<float> k_noise(NKV * D), v_noise(NKV * D);
            linear(hidden.data(), layer_w(w_.k_proj, l, per_layer_kv, H),
                   k_noise.data(), H, NKV * D);
            linear(hidden.data(), layer_w(w_.v_proj, l, per_layer_kv, H),
                   v_noise.data(), H, NKV * D);

            // QK RMSNorm + RoPE
            for (int h = 0; h < NH; h++) {
                rms_norm(&q[h * D], &q[h * D],
                         layer_w(w_.q_norm, l, cfg_.head_dim), D);
                apply_rope(&q[h * D], D, state.ctx_seq_len + pos);
            }
            for (int h = 0; h < NKV; h++) {
                rms_norm(&k_noise[h * D], &k_noise[h * D],
                         layer_w(w_.k_norm, l, cfg_.head_dim), D);
                apply_rope(&k_noise[h * D], D, state.ctx_seq_len + pos);
            }

            // Cache K_noise, V_noise
            state.write_kv(l, NKV, D, k_noise.data(), v_noise.data());

            // Attention: softmax(Q @ [K_ctx; K_cache]^T / sqrt(d)) @ [V_ctx; V_cache]
            int total_kv = state.ctx_seq_len + state.seq_len + 1; // ctx + cached + current
            int cur_ctx = state.ctx_seq_len;
            int cur_cache = state.seq_len + 1; // includes current position just cached

            std::vector<float> attn_out(NH * D, 0.0f);
            std::vector<float> scores(total_kv);

            for (int h = 0; h < NH; h++) {
                int kvh = h / GQA;
                const float* q_h = &q[h * D];

                // Score against context (target features)
                const float* k_ctx_layer = &state.ctx_k[(size_t)l * NKV * D];
                for (int t = 0; t < cur_ctx; t++) {
                    double s = 0.0;
                    const float* krow = &k_ctx_layer[(size_t)kvh * D];
                    for (int d = 0; d < D; d++)
                        s += (double)q_h[d] * krow[d];
                    scores[t] = (float)(s / std::sqrt((float)D));
                }

                // Score against cached draft positions
                const float* k_cache_layer = state.k_ptr(l);
                for (int t = 0; t < cur_cache; t++) {
                    double s = 0.0;
                    const float* krow = &k_cache_layer[((size_t)t * NKV + kvh) * D];
                    for (int d = 0; d < D; d++)
                        s += (double)q_h[d] * krow[d];
                    scores[cur_ctx + t] = (float)(s / std::sqrt((float)D));
                }

                // Softmax
                float mx = scores[0];
                for (int i = 1; i < total_kv; i++)
                    if (scores[i] > mx) mx = scores[i];
                double sum = 0.0;
                for (int i = 0; i < total_kv; i++) {
                    scores[i] = std::exp(scores[i] - mx);
                    sum += scores[i];
                }
                float inv_sum = (float)(1.0 / sum);

                // Weighted sum of V
                // V_ctx part
                const float* v_ctx_layer = &state.ctx_v[(size_t)l * NKV * D];
                for (int t = 0; t < cur_ctx; t++) {
                    float p = scores[t] * inv_sum;
                    const float* vrow = &v_ctx_layer[(size_t)kvh * D];
                    for (int d = 0; d < D; d++)
                        attn_out[h * D + d] += p * vrow[d];
                }
                // V_cache part
                const float* v_cache_layer = state.v_ptr(l);
                for (int t = 0; t < cur_cache; t++) {
                    float p = scores[cur_ctx + t] * inv_sum;
                    const float* vrow = &v_cache_layer[((size_t)t * NKV + kvh) * D];
                    for (int d = 0; d < D; d++)
                        attn_out[h * D + d] += p * vrow[d];
                }
            }

            // O projection
            std::vector<float> o_buf(H);
            linear(attn_out.data(), layer_w(w_.o_proj, l, per_layer_o, NH * D),
                   o_buf.data(), NH * D, H);

            // Residual add
            for (int i = 0; i < H; i++)
                hidden[i] = residual[i] + o_buf[i];

            // Post-attention norm + SwiGLU MLP
            std::vector<float> post_residual = hidden;
            rms_norm(hidden.data(), hidden.data(),
                     layer_w(w_.post_attention_layernorm, l, per_layer_h), H);

            std::vector<float> gate(IM), up(IM);
            linear(hidden.data(), layer_w(w_.gate_proj, l, per_layer_gate, H),
                   gate.data(), H, IM);
            linear(hidden.data(), layer_w(w_.up_proj, l, per_layer_up, H),
                   up.data(), H, IM);
            for (int i = 0; i < IM; i++)
                up[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];

            std::vector<float> mlp_out(H);
            linear(up.data(), layer_w(w_.down_proj, l, per_layer_down, IM),
                   mlp_out.data(), IM, H);

            for (int i = 0; i < H; i++)
                hidden[i] = post_residual[i] + mlp_out[i];
        }

        // ---- 4. Final norm ----
        std::vector<float> final_normed(H);
        rms_norm(hidden.data(), final_normed.data(), w_.norm.data(), H);
        std::copy(hidden.begin(), hidden.end(), draft_hidden);

        // ---- 5. LM head ----
        linear(final_normed.data(), w_.lm_head.data(), draft_logits, H, V);

        // ---- 6. Markov correction: draft_logits += W2 @ W1(prev_token) ----
        // prev_token is input_id (the token fed to this step = token at pos-1 in the round)
        {
            float* w1_row = &w_.markov_w1[(size_t)input_id * R];
            std::vector<float> markov_embed(R);
            for (int r = 0; r < R; r++) markov_embed[r] = w1_row[r];

            // W2 is [V, R] stored row-major (out=V, in=R)
            const float* w2 = w_.markov_w2.data();
            for (int v = 0; v < V; v++) {
                double bias = 0.0;
                const float* w2_row = &w2[(size_t)v * R];
                for (int r = 0; r < R; r++)
                    bias += (double)w2_row[r] * markov_embed[r];
                draft_logits[v] += (float)bias;
            }

            // ---- 7. Confidence head: sigmoid(proj(concat(hidden, W1(prev_token)))) ----
            if (confidence_out) {
                double conf = (double)w_.confidence_bias[0];
                const float* cw = w_.confidence_weight.data();
                for (int i = 0; i < H; i++)
                    conf += (double)cw[i] * final_normed[i];
                for (int r = 0; r < R; r++)
                    conf += (double)cw[H + r] * markov_embed[r];
                *confidence_out = 1.0f / (1.0f + (float)std::exp(-conf));
            }
        }

        // Advance sequence length in state (current position stored above)
        state.seq_len = pos + 1;
    }

    // Compute confidence for the current step without running the full forward.
    // Can be called after forward() to retrieve the confidence from a previous step,
    // or standalone with the hidden state and prev_token.
    float compute_confidence(const float* hidden, int32_t prev_token) const {
        if (w_.empty()) return 0.5f;
        int H = cfg_.hidden_size;
        int R = cfg_.markov_rank;
        const float* w1_row = &w_.markov_w1[(size_t)prev_token * R];
        double conf = (double)w_.confidence_bias[0];
        const float* cw = w_.confidence_weight.data();
        for (int i = 0; i < H; i++)
            conf += (double)cw[i] * hidden[i];
        for (int r = 0; r < R; r++)
            conf += (double)cw[H + r] * w1_row[r];
        return 1.0f / (1.0f + (float)std::exp(-conf));
    }

    // ---- Batch forward: generate block_size draft tokens with Markov correction ----
    // trunk_hidden: [num_target_layers * hidden_size]
    // last_token: token at position before draft block
    // draft_tokens: [block_size] output draft token IDs
    // draft_confidences: [block_size] output confidence scores (or nullptr)
    // Returns: number of tokens actually generated (always block_size unless EOS hit)
    int generate_block(
        const float* trunk_hidden,
        int32_t last_token,
        int32_t* draft_tokens,
        float* draft_confidences = nullptr
    ) {
        DSparkDraftState state;
        state.resize(cfg_.num_draft_layers, cfg_.num_kv_heads, cfg_.head_dim, cfg_.block_size);

        int V = cfg_.vocab_size;
        std::vector<float> logits(V);
        std::vector<float> hidden(cfg_.hidden_size);
        float conf;

        int32_t prev = last_token;
        for (int i = 0; i < cfg_.block_size; i++) {
            forward(trunk_hidden, prev, i, state, logits.data(), hidden.data(),
                    draft_confidences ? &conf : nullptr);
            prev = argmax(logits.data(), V);
            draft_tokens[i] = prev;
            if (draft_confidences) draft_confidences[i] = conf;
        }
        return cfg_.block_size;
    }

private:
    DSparkDraftConfig cfg_;
    DSparkDraftWeights w_;
    std::vector<float> inv_freq_;
    std::vector<float> cached_target_features_;  // [hidden_size], cached per round
    int cached_round_id_ = -1;

    // Helper: get pointer to layer l's weight within a flattened multi-layer array
    const float* layer_w(const std::vector<float>& all, int l, size_t per_layer_size) const {
        return all.data() + (size_t)l * per_layer_size;
    }
    float* layer_w(std::vector<float>& all, int l, size_t per_layer_size) {
        return all.data() + (size_t)l * per_layer_size;
    }

    // Overload for Linear weights that have in_dim x out_dim layout
    const float* layer_w(const std::vector<float>& all, int l, size_t per_layer_size, int /*in_dim*/) const {
        return all.data() + (size_t)l * per_layer_size;
    }

    // ---- Low-level ops (same conventions as MTPDraftModel) ----

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

    // HuggingFace "rotate_half" RoPE convention
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

    int32_t argmax(const float* logits, int32_t n) {
        return (int32_t)std::distance(logits, std::max_element(logits, logits + n));
    }
};

// ---------------------------------------------------------------------------
// Python export script (documentation — embedded as comment for reference)
// ---------------------------------------------------------------------------
// To export a trained DeepSpec DSpark checkpoint to the flat binary format:
//
//   # export_dspark_weights.py
//   """Export DSpark checkpoint to flat binary for DSparkDraftWeights::load()."""
//   import argparse, struct, numpy as np
//   from safetensors import safe_open
//
//   PER_LAYER_FIELDS = [
//       "input_layernorm.weight",
//       "self_attn.q_proj.weight",
//       "self_attn.k_proj.weight",
//       "self_attn.v_proj.weight",
//       "self_attn.o_proj.weight",
//       "self_attn.q_norm.weight",
//       "self_attn.k_norm.weight",
//       "post_attention_layernorm.weight",
//       "mlp.gate_proj.weight",
//       "mlp.up_proj.weight",
//       "mlp.down_proj.weight",
//   ]
//
//   def main():
//       ap = argparse.ArgumentParser()
//       ap.add_argument("--checkpoint", required=True)
//       ap.add_argument("--output", required=True)
//       ap.add_argument("--num-layers", type=int, default=5)
//       args = ap.parse_args()
//
//       with safe_open(args.checkpoint, framework="pt") as f:
//           keys = set(f.keys())
//           with open(args.output, "wb") as out:
//               def write(name):
//                   if name not in keys:
//                       print(f"WARNING: {name} not found, skipping", flush=True)
//                       return
//                   t = f.get_tensor(name).to(dtype=torch.float32).numpy().astype(np.float32)
//                   out.write(t.tobytes())
//                   print(f"  {name:40s} {str(list(t.shape)):20s} {t.nbytes/1e6:6.1f} MB")
//
//               # Shared
//               write("model.embed_tokens.weight")
//               write("model.fc.weight")
//               write("model.hidden_norm.weight")
//
//               # Layers (all copies of each field together)
//               for field in PER_LAYER_FIELDS:
//                   for l in range(args.num_layers):
//                       write(f"model.layers.{l}.{field}")
//
//               # Final norm + head
//               write("model.norm.weight")
//               write("model.lm_head.weight")
//
//               # Markov head
//               write("model.markov_head.markov_w1.weight")    # Embedding
//               write("model.markov_head.markov_w2.weight")    # Linear
//
//               # Confidence head
//               write("model.confidence_head.proj.weight")     # [1, H+R]
//               write("model.confidence_head.proj.bias")       # [1]
//
//       print(f"\nWrote {args.output}")
//
//   if __name__ == "__main__":
//       main()


