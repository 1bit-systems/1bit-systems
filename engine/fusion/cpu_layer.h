#pragma once
// cpu_layer.h — C ABI for CPU ternary inference backend
// Exposes all ops needed for the fused engine's CPU path.
// Compiled as plain C++17, no GPU deps.
//
// Weight format: 2-bit packed ternary, 16 values per uint32
//   bits[2i:2i+1] = 00→0(skip), 01→+1(add), 10→-1(sub)

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── GEMV: y[M] = scales[M] .* (packed[M, K/16] @ x[K]) ──────────────
// Per-row scale GEMV: one scale per output row
void cpu_ternary_gemv(
    const uint32_t* packed,   // [M, K/16] packed ternary weights
    const float*    x,        // [K] activation
    const float*    scales,   // [M] per-row scale
    float*          y,        // [M] output
    int M, int K);

// Per-block scale GEMV: one scale per 256-weight block per row
//   scales: [M, n_blocks] where n_blocks = K / 256
//   Block = 256 ternary values = 16 uint32s = 64 packed bytes
void cpu_ternary_gemv_block(
    const uint32_t* packed,   // [M, K/16] packed ternary weights
    const float*    x,        // [K] activation
    const float*    scales,   // [M, n_blocks] per-block scales
    float*          y,        // [M] output
    int M, int K, int n_blocks);

// ── RMSNorm: y[N] = x[N] * rms_inv * w[N] ──────────────────────────
//   rms = sqrt(mean(x^2) + eps)
//   y = x / rms * w
void cpu_rmsnorm(
    const float* x,       // [N] input
    const float* weight,  // [N] norm weight
    float*       y,       // [N] output
    int N, float eps);

// ── RoPE: in-place rotary position embedding ────────────────────────
void cpu_rope(
    float*       x,      // [n_heads * head_dim] in/out
    int          pos,    // absolute position
    int          n_heads,
    int          head_dim,
    const float* sin_table,  // [max_pos * head_dim]
    const float* cos_table); // [max_pos * head_dim]

// ── SiLU GLU (Qwen3): out[N] = silu(gate[N]) * up[N] ──────────────
void cpu_silu_glu(
    const float* gate,  // [N]
    const float* up,    // [N]
    float*       out,   // [N]
    int N);

// ── ReLU² GLU (BitNet): out[N] = relu2(gate[N]) * up[N] ───────────
void cpu_relu2_glu(
    const float* gate,
    const float* up,
    float*       out,
    int N);

// ── Causal attention (scaled dot-product, CPU) ─────────────────────
//   For each head h (GQA: kvh = h/gqa):
//     scores[p] = q[h*HD..(h+1)*HD] · k_cache[p*NKV*HD + kvh*HD..] * scale
//     softmax → weighted sum of v_cache
void cpu_attention(
    const float* q,           // [n_heads * head_dim]
    const float* k_cache,     // [seq_len * n_kv_heads * head_dim]
    const float* v_cache,     // [seq_len * n_kv_heads * head_dim]
    float*       output,      // [n_heads * head_dim]
    int n_heads, int n_kv_heads, int head_dim,
    int seq_len, int gqa_ratio);

// ── Full transformer layer (Qwen3) ─────────────────────────────────
//   Processes one token through one transformer layer.
//   All weights are fp32.
//
// Returns: 0 on success, non-zero on error.
int cpu_layer_forward_qwen3(
    // In/out: hidden state [hidden_dim]
    float*       hidden,
    // Output buffers (scratch, overwritten each call)
    float*       scratch_qkv,     // [n_heads*hd + 2*n_kv_heads*hd]
    float*       scratch_attn,    // [n_heads*hd]
    float*       scratch_ffn,     // [2 * inter_size]
    float*       scratch_act,     // [inter_size]
    // Layer weights
    const float* in_norm_weight,    // [hidden_dim]
    const float* attn_q_norm,       // [head_dim] or NULL
    const float* attn_k_norm,       // [head_dim] or NULL
    const float* pa_norm_weight,    // [hidden_dim]
    const float* final_norm_weight, // [hidden_dim] or NULL (only for last layer)
    // Packed ternary weights + scales
    const uint32_t* q_packed,  const float* q_scales,
    const uint32_t* k_packed,  const float* k_scales,
    const uint32_t* v_packed,  const float* v_scales,
    const uint32_t* o_packed,  const float* o_scales,
    const uint32_t* gate_packed, const float* gate_scales,
    const uint32_t* up_packed,   const float* up_scales,
    const uint32_t* down_packed, const float* down_scales,
    // KV cache (in/out)
    float* k_cache,   // [max_seq * n_kv_heads * head_dim]
    float* v_cache,   // [max_seq * n_kv_heads * head_dim]
    // Model config
    int hidden_dim, int inter_size,
    int n_heads, int n_kv_heads, int head_dim, int gqa_ratio,
    // Position
    int pos,
    // RoPE tables
    const float* sin_table,
    const float* cos_table,
    // Norm epsilon
    float rms_norm_eps);

// ── LM head: compute logits from final hidden state ────────────────
//   logits[n] = hidden · embedding[n*H .. (n+1)*H]
void cpu_lm_head(
    const float* hidden,      // [hidden_dim]
    const float* embedding,   // [vocab_size * hidden_dim]
    float*       logits,      // [vocab_size] output
    int vocab_size, int hidden_dim);

// ── Embedding lookup: out[H] = table[tok * H .. (tok+1) * H] ──────
void cpu_embed(
    const float* table,   // [vocab_size * hidden_dim]
    int token,
    float*       out,     // [hidden_dim]
    int hidden_dim);

// ── Argmax: find index of maximum value ────────────────────────────
int cpu_argmax(const float* values, int N);

// ── Softmax (in-place) ─────────────────────────────────────────────
void cpu_softmax(float* values, int N);

#ifdef __cplusplus
}
#endif
