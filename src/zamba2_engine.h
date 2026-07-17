// zamba2_engine.h — Zamba2 inference engine for 1bit.systems
//
// Zamba2 is a hybrid Mamba2 + shared-attention architecture:
//   - 45 Mamba2 SSM layers (Mamba2 blocks)
//   - 9 hybrid layers (Mamba2 decoder + shared attention + shared MLP with LoRA)
//   - 2 shared attention blocks reused in ABAB pattern
//   - 9 LoRA adapters (one per hybrid position) on shared MLP
//
// Architecture config for Zamba2-2.7B:
//   d_model = 2560, d_state = 64, d_conv = 4, d_inner = 5120
//   n_head = 80, n_group = 1, head_dim = 64
//   n_hybrid_layers = 9, n_shared_blocks = 2
//   hybrid_layer_ids = [6, 12, 18, 24, 30, 36, 42, 47, 51]
//
// This engine provides:
//   1. Model weight loading (from GGUF or raw tensors)
//   2. Per-layer Mamba2 forward (via mamba2_kernels)
//   3. Hybrid layer: Mamba2 decoder + shared attn + shared MLP + LoRA
//   4. Full model: embedding → 54 layers → final norm → lm_head
//
#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <unordered_map>

#include "mamba2_kernels.h"

// ── Zamba2 Architecture Config (from HF config.json) ──
struct Zamba2Config {
    // Core dimensions
    int d_model = 2560;          // hidden_size
    int d_state = 64;            // mamba_d_state
    int d_conv = 4;              // mamba_d_conv
    int d_inner = 5120;          // hidden_size * mamba_expand
    int n_head = 80;             // d_inner / head_dim = attention_head_dim * num_attention_heads / head_dim
                                 // actually: head_dim * num_attention_heads / ... let's calculate
    int n_group = 1;             // mamba_ngroups
    int head_dim = 64;           // mamba_headdim

    // Attention
    int n_attn_heads = 32;       // num_attention_heads
    int n_kv_heads = 32;         // num_key_value_heads
    int attn_head_dim = 80;      // kv_channels
    int attn_hidden_size = 5120; // attention_hidden_size

    // Architecture layout
    int n_layers = 54;           // num_hidden_layers
    int n_hybrid = 9;            // number of hybrid layers
    int n_shared_blocks = 2;     // num_mem_blocks
    int vocab_size = 32000;      // padded_vocab_size or vocab_size
    int max_seq_len = 4096;      // max_position_embeddings

    // LoRA
    int lora_rank = 128;         // adapter_rank
    bool use_shared_mlp_adapter = true;
    bool use_shared_attention_adapter = false;

    // Norm
    float rms_norm_eps = 1e-5f;

    // RoPE
    float rope_theta = 10000.0f;

    // Derived
    int hyb_layer_ids[9] = {6, 12, 18, 24, 30, 36, 42, 47, 51};

    // Validate and derive
    bool validate() {
        // d_inner = head_dim * n_head
        if (d_inner <= 0) d_inner = head_dim * n_head;
        if (n_head <= 0) n_head = d_inner / head_dim;
        return true;
    }
};

// ── Per-layer Mamba2 weights ──
struct Mamba2LayerWeights {
    // in_proj: [d_in_proj, d_model]  where d_in_proj = d_inner + conv_dim + n_head
    std::vector<float> in_proj_w;
    // conv1d: [d_conv, conv_dim]
    std::vector<float> conv1d_w;
    std::vector<float> conv1d_b;   // [conv_dim]
    // dt_bias: [n_head]
    std::vector<float> dt_bias;
    // A_log: [n_head]
    std::vector<float> A_log;
    // D: [n_head]
    std::vector<float> D;
    // norm: [d_inner / n_group, n_group]
    std::vector<float> norm_w;
    // out_proj: [d_model, d_inner]
    std::vector<float> out_proj_w;
    // input_layernorm: [d_model]
    std::vector<float> input_norm_w;

    bool loaded = false;
};

// ── Per-shared-block attention + MLP weights ──
struct SharedBlockWeights {
    // input_layernorm: [d_model]
    std::vector<float> input_norm_w;
    // pre_ff_layernorm: [d_model]
    std::vector<float> pre_ff_norm_w;
    // QKV projections: [d_model, d_model] each
    std::vector<float> q_proj_w;
    std::vector<float> k_proj_w;
    std::vector<float> v_proj_w;
    std::vector<float> o_proj_w;
    // Gate/Up/Down projections: [d_model, d_model]
    std::vector<float> gate_up_proj_w;  // fused gate+up: [2 * d_model, d_model]
    std::vector<float> down_proj_w;     // [d_model, d_model]

    bool loaded = false;
};

// ── Hybrid layer weights (mamba decoder + shared block reference + projection + LoRA) ──
struct HybridLayerWeights {
    Mamba2LayerWeights mamba;        // mamba_decoder.mamba.*
    std::vector<float> mamba_input_norm_w;  // mamba_decoder.input_layernorm
    std::vector<float> linear_w;     // linear projection [d_model, d_model]

    // Reference to shared block (index into shared_blocks[])
    int shared_block_idx = 0;

    // LoRA adapters for this hybrid layer position
    // gate_up_proj_adapter_list.{pos}.0 = A matrix (LoRA down)
    // gate_up_proj_adapter_list.{pos}.1 = B matrix (LoRA up)
    std::vector<float> lora_a_w;  // [lora_rank, d_model]
    std::vector<float> lora_b_w;  // [2 * d_model, lora_rank]

    // input_layernorm for the full hybrid layer
    std::vector<float> input_norm_w;  // layers.N.input_layernorm

    // Shared transformer weights (duplicated per layer by GGUF converter)
    std::vector<float> shared_transformer_q;     // self_attn.q_proj
    std::vector<float> shared_transformer_k;     // self_attn.k_proj
    std::vector<float> shared_transformer_v;     // self_attn.v_proj
    std::vector<float> shared_transformer_o;     // self_attn.o_proj
    std::vector<float> shared_transformer_pre_ff_norm;  // post_attention_norm
    std::vector<float> shared_transformer_ffn_norm;     // ffn_norm (input norm for FFN)
    std::vector<float> shared_transformer_gate;  // ffn_gate (SiLU gate)
    std::vector<float> shared_transformer_up;    // ffn_up
    std::vector<float> shared_transformer_down;  // ffn_down

    bool loaded = false;
};

// ── Complete Zamba2 model ──
struct Zamba2Model {
    Zamba2Config cfg;

    // Embedding
    std::vector<float> embed_w;         // [vocab_size, d_model] or [d_model, vocab_size]

    // Final norm
    std::vector<float> final_norm_w;    // [d_model]

    // Pure Mamba2 layers (indices not in hyb_layer_ids)
    std::unordered_map<int, Mamba2LayerWeights> mamba_layers;

    // Hybrid layers (indices in hyb_layer_ids)
    std::unordered_map<int, HybridLayerWeights> hybrid_layers;

    // Shared attention/MLP blocks
    std::vector<SharedBlockWeights> shared_blocks;

    // State (persistent across tokens)
    std::vector<float> conv_states;     // [n_layers, d_conv-1, conv_dim]
    std::vector<float> ssm_states;      // [n_layers, d_state, d_inner]

    // KV cache for attention layers (one per hybrid layer, since weights are duplicated)
    std::vector<float> kv_cache;        // [n_hybrid, 2, max_seq, n_kv_heads * attn_head_dim]

    int pos = 0;  // current position in sequence

    bool loaded = false;

    // Initialize state buffers
    bool init_state() {
        int64_t conv_dim = cfg.d_inner + 2 * cfg.n_group * cfg.d_state;
        conv_states.resize(cfg.n_layers * (cfg.d_conv - 1) * conv_dim, 0.0f);
        ssm_states.resize(cfg.n_layers * cfg.d_state * cfg.d_inner, 0.0f);
        // Allocate KV cache: one slot per hybrid layer
        int n_hybrid_layers = num_hybrid_layers();
        kv_cache.resize(n_hybrid_layers * 2 * cfg.max_seq_len * cfg.n_kv_heads * cfg.attn_head_dim, 0.0f);
        pos = 0;
        return true;
    }

    // Reset state for new sequence
    void reset() {
        std::fill(conv_states.begin(), conv_states.end(), 0.0f);
        std::fill(ssm_states.begin(), ssm_states.end(), 0.0f);
        std::fill(kv_cache.begin(), kv_cache.end(), 0.0f);
        pos = 0;
    }

    // Get number of hybrid layers
    int num_hybrid_layers() const {
        return (int)hybrid_layers.size();
    }

    // ── Forward pass for one token ──
    // token_id: input token
    // logits: [vocab_size] output
    bool forward(int token_id, float* logits);
};

// ── Utility: RMS Norm ──
inline void rms_norm(const float* x, float* y, const float* w, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    float rms = std::sqrt(ss / n + eps);
    float inv_rms = 1.0f / rms;
    for (int i = 0; i < n; ++i) y[i] = x[i] * inv_rms * w[i];
}

// ── Utility: RoPE ──
inline void apply_rope(float* q, float* k, int pos, int head_dim, int n_heads, int n_kv_heads, float theta) {
    for (int h = 0; h < n_heads; ++h) {
        for (int d = 0; d < head_dim; d += 2) {
            float freq = pos / std::pow(theta, (float)d / head_dim);
            float cos_val = std::cos(freq);
            float sin_val = std::sin(freq);
            int idx = h * head_dim + d;
            float q0 = q[idx], q1 = q[idx + 1];
            q[idx]     = q0 * cos_val - q1 * sin_val;
            q[idx + 1] = q0 * sin_val + q1 * cos_val;
            if (h < n_kv_heads) {
                float k0 = k[idx], k1 = k[idx + 1];
                k[idx]     = k0 * cos_val - k1 * sin_val;
                k[idx + 1] = k0 * sin_val + k1 * cos_val;
            }
        }
    }
}

// ── Utility: scaled dot-product attention ──
inline void attention_forward(
    const float* q,           // [n_heads * head_dim]
    const float* k_cache,     // [max_seq, n_kv_heads * head_dim]
    const float* v_cache,     // [max_seq, n_kv_heads * head_dim]
    float* output,            // [d_model]
    int pos, int max_seq, int n_heads, int n_kv_heads, int head_dim, int d_model
) {
    int n_groups = n_heads / n_kv_heads;
    std::vector<float> scores(max_seq, 0.0f);

    for (int g = 0; g < n_groups; ++g) {
        for (int t = 0; t <= pos; ++t) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                int hi = g * n_kv_heads;  // which KV head this group uses
                score += q[(g * n_kv_heads) * head_dim + d] * k_cache[t * n_kv_heads * head_dim + hi * head_dim + d];
            }
            scores[t] = score / std::sqrt((float)head_dim);
        }

        // Softmax
        float max_s = *std::max_element(scores.begin(), scores.begin() + pos + 1);
        float sum_exp = 0.0f;
        for (int t = 0; t <= pos; ++t) scores[t] = std::exp(scores[t] - max_s);
        for (int t = 0; t <= pos; ++t) sum_exp += scores[t];
        for (int t = 0; t <= pos; ++t) scores[t] /= sum_exp;

        // Weighted sum of values
        for (int d = 0; d < head_dim; ++d) {
            float val = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                val += scores[t] * v_cache[t * n_kv_heads * head_dim + g * head_dim + d];
            }
            output[g * head_dim + d] = val;
        }
    }
}
