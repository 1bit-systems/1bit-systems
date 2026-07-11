// Zaya MoE Ternary — C API for expert-sparse ternary GEMV kernels.
//
// Zaya is a hybrid CCA-attention + MoE architecture with EDA router.
// These kernels handle the ternary-weight MoE forward pass:
//
//   1. Accept EDA router output: expert_indices[tokens][top_k] int32,
//      expert_weights[tokens][top_k] float.
//   2. Load ONLY the selected experts' TQ1-packed ternary weights.
//   3. Compute dot product against INT8 input activations.
//   4. Apply per-expert scale + router combination weight.
//   5. Output fp16 (only top_k slots populated per token).
//
// Packing: TQ1 base-3 (1.6 bpw), lossless for ternary {-1, 0, +1}.

#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ── Single-token decode ──────────────────────────────────────────────────
// Computes expert outputs for ONE token. The EDA router selects top_k
// experts; only those are computed.
//
// packed_experts  [num_experts][K_padded/5 bytes viewed as u32s] — TQ1 weights
// x_i8            [K_padded] — INT8 input activations
// x_scale         — activation quantization scale
// expert_scales   [num_experts] — per-expert output scales
// expert_indices  [tokens × top_k] int32 — which experts are active
// expert_weights  [tokens × top_k] float — router combination weights
// y_f16           [tokens][num_experts] — fp16 output (only top_k populated)
// tokens          — batch size
// num_experts     — total experts in the MoE layer
// top_k           — experts selected per token (must be ≥ 1)
// K_padded        — hidden dimension, padded to multiple of 20
// stream          — HIP stream (nullptr for default)
void zaya_moe_ternary_gemv_launch(
    const void* packed_experts, const void* x_i8,
    float x_scale, const void* expert_scales,
    const void* expert_indices, const void* expert_weights,
    void* y_f16,
    int tokens, int num_experts, int top_k, int K_padded,
    void* stream);

// ── Batched prefill ──────────────────────────────────────────────────────
// Same as above but reads activations from a batched buffer.
//
// x_i8_batch  [B × K_padded] — INT8 activation batch (one row per token)
// B           — batch size (number of tokens in prefill)
//
// All other parameters are identical to the single-token variant.
void zaya_moe_ternary_gemv_batched_launch(
    const void* packed_experts, const void* x_i8_batch,
    float x_scale, const void* expert_scales,
    const void* expert_indices, const void* expert_weights,
    void* y_f16,
    int B, int num_experts, int top_k, int K_padded,
    void* stream);

#ifdef __cplusplus
} // extern "C"
#endif
