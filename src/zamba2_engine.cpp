// zamba2_engine.cpp — Zamba2 model forward pass implementation
//
// Implements the full Zamba2 architecture:
//   1. Token embedding lookup
//   2. 54 layers (45 Mamba2 + 9 hybrid)
//   3. Final RMS norm
//   4. LM head (tied embeddings)
//
// Hybrid layer structure:
//   hidden → input_norm → mamba_decoder → linear → shared_transformer → hidden
//   Where mamba_decoder is a full Mamba2 block
//   And shared_transformer = self_attn + RoPE + pre_ff_norm + gate_up/down MLP + LoRA

#include "zamba2_engine.h"
#include <algorithm>
#include <cmath>
#include <cstring>

// Forward helper: apply one pure Mamba2 layer
static void forward_mamba_layer(
    const float* input,
    float* output,
    const Mamba2LayerWeights& w,
    float* conv_state,
    float* ssm_state,
    const Mamba2Config& cfg,
    int conv_dim
) {
    int d_model = cfg.d_model;
    int n = d_model;

    // Allocate temps
    std::vector<float> normed(n);

    // RMS norm
    rms_norm(input, normed.data(), w.input_norm_w.data(), n, cfg.rms_norm_eps);

    // Mamba2 forward
    mamba2_cpu_forward(
        normed.data(),
        w.in_proj_w.data(),
        w.conv1d_w.data(),
        w.conv1d_b.data(),
        w.dt_bias.data(),
        w.A_log.data(),
        w.D.data(),
        w.norm_w.data(),
        w.out_proj_w.data(),
        conv_state,
        ssm_state,
        output,
        cfg
    );

    // Residual
    for (int i = 0; i < n; ++i) {
        output[i] += input[i];
    }
}

// Forward helper: apply one hybrid layer (Mamba2 decoder + shared attention + MLP)
// In the GGUF format, shared block weights are duplicated per hybrid layer,
// so we use the per-layer weights from HybridLayerWeights directly.
static void forward_hybrid_layer(
    const float* input,
    float* output,
    const HybridLayerWeights& hw,
    const Zamba2Config& cfg,
    float* conv_state,
    float* ssm_state,
    float* kv_k_cache,
    float* kv_v_cache,
    int pos,
    int max_seq
) {
    int d_model = cfg.d_model;
    int n = d_model;

    std::vector<float> normed(n);
    std::vector<float> mamba_out(n);
    std::vector<float> projected(n);
    std::vector<float> attn_out(n);
    std::vector<float> ff_in(n);
    std::vector<float> ff_out(n);

    // ── Step 1: Input norm ──
    rms_norm(input, normed.data(), hw.input_norm_w.data(), n, cfg.rms_norm_eps);

    // ── Step 2: Mamba2 decoder ──
    {
        std::vector<float> mamba_normed(n);
        rms_norm(normed.data(), mamba_normed.data(), hw.mamba_input_norm_w.data(), n, cfg.rms_norm_eps);

        mamba2_cpu_forward(
            mamba_normed.data(),
            hw.mamba.in_proj_w.data(),
            hw.mamba.conv1d_w.data(),
            hw.mamba.conv1d_b.data(),
            hw.mamba.dt_bias.data(),
            hw.mamba.A_log.data(),
            hw.mamba.D.data(),
            hw.mamba.norm_w.data(),
            hw.mamba.out_proj_w.data(),
            conv_state,
            ssm_state,
            mamba_out.data(),
            [&]() -> Mamba2Config {
                Mamba2Config mc;
                mc.d_model = cfg.d_model;
                mc.d_state = cfg.d_state;
                mc.d_conv = cfg.d_conv;
                mc.d_inner = cfg.d_inner;
                mc.n_head = cfg.n_head;
                mc.n_group = cfg.n_group;
                mc.head_dim = cfg.head_dim;
                mc.rms_norm_eps = cfg.rms_norm_eps;
                return mc;
            }()
        );
    }

    // Mamba decoder residual
    for (int i = 0; i < n; ++i) mamba_out[i] += normed[i];

    // ── Step 3: Linear projection ──
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < n; ++j) {
            sum += hw.linear_w[i * n + j] * mamba_out[j];
        }
        projected[i] = sum;
    }

    // ── Step 4: Transformer (attention + FFN) ──
    // Input norm for attention
    rms_norm(projected.data(), ff_in.data(), hw.shared_transformer_ffn_norm.data(), n, cfg.rms_norm_eps);

    // Self-attention with per-layer weights (duplicated shared blocks)
    {
        int n_heads = cfg.n_attn_heads;
        int n_kv = cfg.n_kv_heads;
        int hd = cfg.attn_head_dim;

        // QKV projections using per-hybrid-layer weights
        std::vector<float> q(n_heads * hd, 0.0f);
        std::vector<float> k(n_kv * hd, 0.0f);
        std::vector<float> v(n_kv * hd, 0.0f);

        for (int h = 0; h < n_heads; ++h) {
            for (int d = 0; d < hd; ++d) {
                float sum = 0.0f;
                for (int j = 0; j < n; ++j) {
                    sum += hw.shared_transformer_q[(h * hd + d) * n + j] * ff_in[j];
                }
                q[h * hd + d] = sum;
            }
        }
        for (int h = 0; h < n_kv; ++h) {
            for (int d = 0; d < hd; ++d) {
                float sum_k = 0.0f, sum_v = 0.0f;
                for (int j = 0; j < n; ++j) {
                    sum_k += hw.shared_transformer_k[(h * hd + d) * n + j] * ff_in[j];
                    sum_v += hw.shared_transformer_v[(h * hd + d) * n + j] * ff_in[j];
                }
                k[h * hd + d] = sum_k;
                v[h * hd + d] = sum_v;
            }
        }

        // RoPE
        apply_rope(q.data(), k.data(), pos, hd, n_heads, n_kv, cfg.rope_theta);

        // Store in KV cache
        for (int h = 0; h < n_kv; ++h) {
            for (int d = 0; d < hd; ++d) {
                kv_k_cache[pos * n_kv * hd + h * hd + d] = k[h * hd + d];
                kv_v_cache[pos * n_kv * hd + h * hd + d] = v[h * hd + d];
            }
        }

        // Attention
        attention_forward(q.data(), kv_k_cache, kv_v_cache, attn_out.data(),
                         pos, max_seq, n_heads, n_kv, hd, n);

        // Output projection
        std::vector<float> attn_proj(n, 0.0f);
        for (int i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                sum += hw.shared_transformer_o[i * n + j] * attn_out[j];
            }
            attn_proj[i] = sum;
        }

        // Attention residual — add to projected (the original pre-norm input),
        // not ff_in (the normalized version). Pre-LN: x = x + attn(x_normed)
        for (int i = 0; i < n; ++i) {
            projected[i] = projected[i] + attn_proj[i];
        }
    }

    // ── Step 5: Shared MLP (with separate gate/up/down weights) ──
    {
        std::vector<float> ff_normed(n);
        rms_norm(projected.data(), ff_normed.data(), hw.shared_transformer_pre_ff_norm.data(), n, cfg.rms_norm_eps);

        int d_ff = (int)hw.shared_transformer_up.size() / n;  // hidden size from up_proj
        if (d_ff <= 0) d_ff = n;

        // Gate projection: gate = gate_w @ x
        std::vector<float> gate_act(d_ff, 0.0f);
        for (int i = 0; i < d_ff; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                sum += hw.shared_transformer_gate[i * n + j] * ff_normed[j];
            }
            // SiLU activation
            gate_act[i] = sum / (1.0f + std::exp(-sum));
        }

        // Up projection: up = up_w @ x
        std::vector<float> up_act(d_ff, 0.0f);
        for (int i = 0; i < d_ff; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                sum += hw.shared_transformer_up[i * n + j] * ff_normed[j];
            }
            up_act[i] = sum;
        }

        // Element-wise multiply: act = gate * up
        std::vector<float> act(d_ff, 0.0f);
        for (int i = 0; i < d_ff; ++i) {
            act[i] = gate_act[i] * up_act[i];
        }

        // Down projection
        for (int i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < d_ff; ++j) {
                sum += hw.shared_transformer_down[i * d_ff + j] * act[j];
            }
            ff_out[i] = sum;
        }
    }

    // MLP residual + projection residual
    for (int i = 0; i < n; ++i) {
        output[i] = projected[i] + ff_out[i];
    }
}

// ── Full model forward pass ──
bool Zamba2Model::forward(int token_id, float* logits) {
    if (!loaded) return false;
    if (token_id < 0 || token_id >= cfg.vocab_size) return false;

    int d_model = cfg.d_model;
    int n_layers = cfg.n_layers;
    int conv_dim = cfg.d_inner + 2 * cfg.n_group * cfg.d_state;

    // ── Embedding ──
    std::vector<float> hidden(d_model, 0.0f);
    for (int i = 0; i < d_model; ++i) {
        hidden[i] = embed_w[token_id * d_model + i];
    }

    // ── Layer loop ──
    for (int layer = 0; layer < n_layers; ++layer) {
        // Check if this is a hybrid layer — use actual loaded dictionaries
        bool is_hybrid = hybrid_layers.find(layer) != hybrid_layers.end();

        if (is_hybrid) {
            // Hybrid layer (weights stored per-layer by GGUF converter)
            auto& hl = hybrid_layers[layer];

            // KV cache — sequential index into hybrid layers
            int hyb_idx = 0;
            for (int ll = 0; ll <= layer; ++ll) {
                if (hybrid_layers.find(ll) != hybrid_layers.end()) hyb_idx++;
            }
            hyb_idx--;  // 0-based

            int max_seq = cfg.max_seq_len;
            int n_kv = cfg.n_kv_heads;
            int hd = cfg.attn_head_dim;
            size_t kv_offset = (size_t)hyb_idx * 2 * max_seq * n_kv * hd;
            float* k_cache = kv_cache.data() + kv_offset;
            float* v_cache = kv_cache.data() + kv_offset + (size_t)max_seq * n_kv * hd;

            std::vector<float> layer_out(d_model);
            forward_hybrid_layer(
                hidden.data(), layer_out.data(),
                hl, cfg,
                conv_states.data() + layer * (cfg.d_conv - 1) * conv_dim,
                ssm_states.data() + layer * cfg.d_state * cfg.d_inner,
                k_cache, v_cache,
                pos, max_seq
            );
            hidden = layer_out;
        } else {
            // Pure Mamba2 layer
            auto& ml = mamba_layers[layer];
            std::vector<float> layer_out(d_model);
            Mamba2Config mc2;
            mc2.d_model = cfg.d_model;
            mc2.d_state = cfg.d_state;
            mc2.d_conv = cfg.d_conv;
            mc2.d_inner = cfg.d_inner;
            mc2.n_head = cfg.n_head;
            mc2.n_group = cfg.n_group;
            mc2.head_dim = cfg.head_dim;
            mc2.rms_norm_eps = cfg.rms_norm_eps;

            forward_mamba_layer(
                hidden.data(), layer_out.data(),
                ml,
                conv_states.data() + layer * (cfg.d_conv - 1) * conv_dim,
                ssm_states.data() + layer * cfg.d_state * cfg.d_inner,
                mc2, conv_dim
            );
            hidden = layer_out;
        }
    }

    // ── Final RMS Norm ──
    rms_norm(hidden.data(), hidden.data(), final_norm_w.data(), d_model, cfg.rms_norm_eps);

    // ── LM Head (tied embeddings) ──
    // embed_w layout: [vocab_size, d_model], so lm_head is embed_w^T
    for (int v = 0; v < cfg.vocab_size; ++v) {
        float sum = 0.0f;
        for (int i = 0; i < d_model; ++i) {
            sum += embed_w[v * d_model + i] * hidden[i];
        }
        logits[v] = sum;
    }

    // Advance position
    pos++;

    return true;
}
