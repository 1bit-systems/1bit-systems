// deepseek.h — DeepSeek model family with Multi-Head Latent Attention (MLA)
//
// Models: DeepSeek (V1), DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, DeepSeek-Coder, DeepSeek-VL
//
// MLA (Multi-Head Latent Attention) is DeepSeek's core innovation:
//   Standard: K = x @ W_k [d, d_kv], V = x @ W_v [d, d_kv]  — large KV cache
//   MLA:      c = x @ W_down_kv [d, d_c]     ← compressed latent (d_c << d_kv)
//             K = c @ W_up_k   [d_c, d_kr]    ← decompressed for attention
//             V = c @ W_up_v   [d_c, d_kr]    ← decompressed for attention
//             KV cache stores c (small), not K,V (large)
//
// DeepSeek-V2 config: d=5120, d_c=512, d_kr=128, n_heads=64, n_kv_heads=16
// DeepSeek-V3 config: d=7168, d_c=716, d_kr=128, n_heads=128, n_kv_heads=16
//                    + MoE: 256 experts, top-8, shared expert

#pragma once

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>

// ─── DeepSeek model config ────────────────────────────────────────
struct DeepSeekConfig {
    int hidden_size      = 2048;    // model dimension
    int num_layers       = 24;      // transformer layers
    int num_heads        = 16;      // query heads
    int num_kv_heads     = 16;      // key/value heads (standard = all heads)
    int head_dim         = 128;     // head dimension
    int vocab_size       = 102400;  // DeepSeek vocabulary
    int max_seq_len      = 4096;

    // MLA-specific
    int qk_nope_dim      = 128;     // per-head dim without RoPE (nope)
    int qk_rope_dim      = 64;      // per-head dim with RoPE (rope)
    int v_dim            = 128;     // per-head value dim
    int qk_compressed    = 512;     // compressed KV latent dimension (d_c)
    int q_compressed     = 1536;    // compressed Q latent dimension
    int kv_lora_rank     = 512;     // KV LoRA rank
    int q_lora_rank      = 1536;    // Q LoRA rank
    int shared_expert_n  = 1;       // shared experts
    int routed_expert_n  = 64;      // routed experts (V2: 160, V3: 256)
    int top_k            = 6;       // experts per token (V2: 6, V3: 8)
    int moe_intermediate = 1408;    // expert FFN intermediate (V3: 2048)
    float moe_scale      = 1.0f;    // expert scale

    int n_routed_experts = 64;
    int n_shared_experts = 1;
    int n_expert_groups  = 1;
    int n_limited_groups = 1;
    int score_func       = 0;       // 0=softmax, 1=sigmoid
    float routed_scaling = 1.0f;

    // Factory helpers
    static DeepSeekConfig deepseek_v2() {
        DeepSeekConfig c;
        c.hidden_size = 5120; c.num_layers = 60; c.num_heads = 64; c.num_kv_heads = 16;
        c.head_dim = 128; c.vocab_size = 102400; c.max_seq_len = 4096;
        c.qk_nope_dim = 128; c.qk_rope_dim = 64; c.v_dim = 128;
        c.qk_compressed = 512; c.q_compressed = 1536;
        c.kv_lora_rank = 512; c.q_lora_rank = 1536;
        c.n_routed_experts = 160; c.n_shared_experts = 2; c.top_k = 6;
        c.moe_intermediate = 1408;
        return c;
    }
    static DeepSeekConfig deepseek_v3() {
        DeepSeekConfig c;
        c.hidden_size = 7168; c.num_layers = 61; c.num_heads = 128; c.num_kv_heads = 16;
        c.head_dim = 128; c.vocab_size = 129280; c.max_seq_len = 8192;
        c.qk_nope_dim = 128; c.qk_rope_dim = 64; c.v_dim = 128;
        c.qk_compressed = 716; c.q_compressed = 2048;
        c.kv_lora_rank = 512; c.q_lora_rank = 1536;
        c.n_routed_experts = 256; c.n_shared_experts = 1; c.top_k = 8;
        c.moe_intermediate = 2048;
        c.n_expert_groups = 8; c.n_limited_groups = 4;
        return c;
    }
    static DeepSeekConfig deepseek_r1() {
        auto c = deepseek_v3();
        c.max_seq_len = 32768; // R1 supports extended context
        return c;
    }
};

// ─── Per-layer weights ───────────────────────────────────────────
struct DeepSeekLayerWeights {
    // RMSNorm
    std::vector<float> rms_attn_w;   // pre-attention norm
    std::vector<float> rms_ffn_w;    // pre-FFN norm

    // MLA compressed projections
    std::vector<float> w_kv_a;       // [hidden, kv_lora_rank + qk_rope_dim]
    std::vector<float> w_kv_a_bias;
    std::vector<float> w_kv_b;       // [kv_lora_rank, num_heads * (qk_nope_dim + v_dim)]
    std::vector<float> w_q_a;        // [hidden, q_lora_rank]
    std::vector<float> w_q_a_bias;
    std::vector<float> w_q_b;        // [q_lora_rank, num_heads * qk_nope_dim]
    std::vector<float> w_rope;       // RoPE frequency (learned or computed)
    
    // Output projection
    std::vector<float> w_o;          // [num_heads * v_dim, hidden]
    
    // MoE FFN
    std::vector<float> w_gate;       // router weights [hidden, n_routed_experts]
    std::vector<float> w_gate_bias;
    std::vector<float> w_shared_gate; // shared expert gate [hidden, moe_intermediate]
    std::vector<float> w_shared_up;   // shared expert up [hidden, moe_intermediate]
    std::vector<float> w_shared_down; // shared expert down [moe_intermediate, hidden]
    // Per-expert weights (stored flat: [n_routed_experts * hidden * moe_intermediate])
    std::vector<float> exp_gate;     // expert gate [n_routed_experts, hidden, moe_intermediate]
    std::vector<float> exp_up;       // expert up   [n_routed_experts, hidden, moe_intermediate]
    std::vector<float> exp_down;     // expert down [n_routed_experts, moe_intermediate, hidden]
};

// ─── Full DeepSeek model weights ──────────────────────────────────
struct DeepSeekModel {
    DeepSeekConfig cfg;
    
    // Embeddings
    std::vector<float> token_emb;    // [vocab_size, hidden]
    std::vector<float> final_norm_w; // [hidden]
    std::vector<float> output_w;     // [vocab_size, hidden] (may be tied)
    
    // Layers
    std::vector<DeepSeekLayerWeights> layers;
    
    bool load_from_gguf(const std::string& path, const DeepSeekConfig* override_cfg = nullptr);
    void clear();
};

// ─── Forward declarations ─────────────────────────────────────────
// Run one token through all layers (MLA + MoE FFN)
// token_id: input token
// mla_kv_cache: persistent KV cache for MLA latents [num_layers, kv_lora_rank]
// Returns logits over vocabulary
std::vector<float> deepseek_forward(
    const DeepSeekModel& model, int token_id,
    std::vector<std::vector<float>>& mla_kv_cache,
    int& pos);

// ─── Math helpers (MLA-specific) ──────────────────────────────────
namespace deepseek_math {
    // RMSNorm
    static inline void rmsnorm(float* out, const float* x, const float* w, int n, float eps) {
        double ss = 0; for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
        float inv = 1.0f / sqrtf((float)(ss / n) + eps);
        for (int i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
    }

    // GELU
    static inline float gelu(float x) {
        const float c = 0.7978845608f;
        return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
    }

    // SiLU
    static inline float silu(float x) { return x / (1.0f + expf(-x)); }

    // GEMV: out[M] = in[K] @ w[M, K]
    static inline void matmul(float* out, const float* in, const float* w, int M, int K) {
        for (int i = 0; i < M; i++) {
            float s = 0; for (int j = 0; j < K; j++) s += in[j] * w[(size_t)i * K + j];
            out[i] = s;
        }
    }

    // Softmax in-place
    static inline void softmax_inplace(float* x, int n) {
        float mx = x[0]; for (int i = 1; i < n; i++) mx = std::max(mx, x[i]);
        float sum = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
        float inv = 1.0f / sum; for (int i = 0; i < n; i++) x[i] *= inv;
    }

    // RoPE (rotary position embedding)
    static inline void rope(float* x, int dim, int pos, float theta = 10000.0f) {
        for (int i = 0; i < dim / 2; i++) {
            float freq = pos / powf(theta, 2.0f * i / dim);
            float c = cosf(freq), s = sinf(freq);
            float a = x[i], b = x[i + dim / 2];
            x[i] = a * c - b * s;
            x[i + dim / 2] = b * c + a * s;
        }
    }
};
