// Zaya MoE Ternary GEMV — host-side launcher wrappers.
// Follows the same pattern as ternary_gemv_launchers.hip.
// CK-independent — always compiled.

#include "rocm_cpp/zaya_moe.h"
#include "rocm_cpp/ck_gemm.h"

#include <hip/hip_runtime.h>
#include <cstdint>

// ── Forward declarations (kernel defined in kernels/zaya_moe_ternary_gemv.hip) ──
extern "C" void zaya_moe_ternary_gemv_launch(
    const void*, const void*, float, const void*,
    const void*, const void*, void*,
    int, int, int, int, void*);

extern "C" void zaya_moe_ternary_gemv_batched_launch(
    const void*, const void*, float, const void*,
    const void*, const void*, void*,
    int, int, int, int, void*);

// ── C API: single-token decode ───────────────────────────────────────────
extern "C" rcpp_status_t
rcpp_zaya_moe_ternary_gemv(
    const void* packed_experts, const void* x_i8,
    float x_scale, const void* expert_scales,
    const void* expert_indices, const void* expert_weights,
    void* y_f16,
    int tokens, int num_experts, int top_k, int K_padded,
    void* stream)
{
    if (!packed_experts || !x_i8 || !expert_scales ||
        !expert_indices || !expert_weights || !y_f16)
        return RCPP_INVALID_ARG;
    if (tokens <= 0 || num_experts <= 0 || top_k <= 0 || K_padded <= 0)
        return RCPP_INVALID_ARG;
    if (K_padded % 20 != 0)
        return RCPP_INVALID_ARG;
    if (top_k > num_experts)
        return RCPP_INVALID_ARG;

    zaya_moe_ternary_gemv_launch(
        packed_experts, x_i8, x_scale, expert_scales,
        expert_indices, expert_weights, y_f16,
        tokens, num_experts, top_k, K_padded, stream);

    return RCPP_OK;
}

// ── C API: batched prefill ───────────────────────────────────────────────
extern "C" rcpp_status_t
rcpp_zaya_moe_ternary_gemv_batched(
    const void* packed_experts, const void* x_i8_batch,
    float x_scale, const void* expert_scales,
    const void* expert_indices, const void* expert_weights,
    void* y_f16,
    int B, int num_experts, int top_k, int K_padded,
    void* stream)
{
    if (!packed_experts || !x_i8_batch || !expert_scales ||
        !expert_indices || !expert_weights || !y_f16)
        return RCPP_INVALID_ARG;
    if (B <= 0 || num_experts <= 0 || top_k <= 0 || K_padded <= 0)
        return RCPP_INVALID_ARG;
    if (K_padded % 20 != 0)
        return RCPP_INVALID_ARG;
    if (top_k > num_experts)
        return RCPP_INVALID_ARG;

    zaya_moe_ternary_gemv_batched_launch(
        packed_experts, x_i8_batch, x_scale, expert_scales,
        expert_indices, expert_weights, y_f16,
        B, num_experts, top_k, K_padded, stream);

    return RCPP_OK;
}
