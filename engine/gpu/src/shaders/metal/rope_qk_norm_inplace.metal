#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Fused single-token Q/K-norm + Q/K-RoPE for the Q8 KV-cache decode path.
// Adapted from `rope_kv_cache_write.metal`'s Q/K-norm-into-RoPE fusion (which
// mirrors the reference implementation's `ggml_metal_op_rms_norm` op-fusion at
// ggml-metal-ops.cpp:3384 — fusing rms_norm into the immediately-following
// producer). The existing fused kernel writes K rotated straight into the
// f32 KV cache; that path is gated on `!kv_cache_q8`, leaving Qwen3.6's hot
// decode path (kv_cache_q8 = true) running 4 separate dispatches with a
// norm→rope barrier between them.
//
// This kernel keeps the same op-fusion (Q-norm + Q-rope, K-norm + K-rope) but
// writes K back to `k_inout` in place, so the follow-up
// `dispatchKvCacheWriteQ8OnCmd` can quantize from k_buf as before. Q lives
// in q_buf either way. Saves 2 separate Q/K-norm dispatches + the
// norm→rope barrier + 1 of the 2 separate Q/K-rope dispatches per dense
// full-attn layer (≈10 attn layers / decode token on Qwen3.6-35B), i.e.
// 3 dispatches + 1 barrier dropped from the hot Q8 KV decode path.
//
// Each threadgroup handles one head slot:
//   - head ∈ [0, n_q_heads):                 Q-norm + Q-rope in q_inout[head]
//   - head ∈ [n_q_heads, n_q_heads + n_kv_h): K-norm + K-rope in k_inout[kv_head]

struct RopeQkNormInplacePush {
    uint stride;     // head_dim
    uint rope_dim;
    uint n_q_heads;
    uint position;
    float eps;
};

kernel void main0(
    constant RopeQkNormInplacePush& p [[buffer(0)]],
    device float* q_inout      [[buffer(1)]],
    device float* k_inout      [[buffer(2)]],
    device const float* freqs  [[buffer(3)]],
    device const float* q_norm_w [[buffer(4)]],
    device const float* k_norm_w [[buffer(5)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint stride = p.stride;
    const uint half_rot = p.rope_dim / 2;

    if (head < p.n_q_heads) {
        const uint base = head * stride;

        // Per-simdgroup-redundant reduction (matches rope_kv_cache_write.metal's
        // Q branch): each simdgroup independently sums the full head row and
        // computes its own rms_inv. Both simdgroups in a 64-thread TG read the
        // same L1-resident head row (≤1 KiB for head_dim=256), so the
        // duplicated work is negligible compared to saving the threadgroup
        // memory + barrier roundtrip the standalone RMS norm kernel needs.
        float sum_sq = 0.0f;
        for (uint i = simd_lane; i < stride; i += 32u) {
            const float v = q_inout[base + i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        const float q_rms_inv = fast::rsqrt(fast::divide(sum_sq, float(stride)) + p.eps);

        for (uint i = tid; i < half_rot; i += 64) {
            const float theta = float(p.position) * freqs[i];
            const float cos_t = fast::cos(theta);
            const float sin_t = fast::sin(theta);
            const float x0 = q_inout[base + i] * q_rms_inv * q_norm_w[i];
            const float x1 = q_inout[base + i + half_rot] * q_rms_inv * q_norm_w[i + half_rot];
            q_inout[base + i] = x0 * cos_t - x1 * sin_t;
            q_inout[base + i + half_rot] = x0 * sin_t + x1 * cos_t;
        }
        // Pass-through dims (rope_dim..stride): apply norm in place.
        for (uint i = p.rope_dim + tid; i < stride; i += 64) {
            q_inout[base + i] = q_inout[base + i] * q_rms_inv * q_norm_w[i];
        }
        return;
    }

    const uint kv_head = head - p.n_q_heads;
    const uint base = kv_head * stride;

    float sum_sq = 0.0f;
    for (uint i = simd_lane; i < stride; i += 32u) {
        const float v = k_inout[base + i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);
    const float k_rms_inv = fast::rsqrt(fast::divide(sum_sq, float(stride)) + p.eps);

    for (uint i = tid; i < half_rot; i += 64) {
        const float theta = float(p.position) * freqs[i];
        const float cos_t = fast::cos(theta);
        const float sin_t = fast::sin(theta);
        const float x0 = k_inout[base + i] * k_rms_inv * k_norm_w[i];
        const float x1 = k_inout[base + i + half_rot] * k_rms_inv * k_norm_w[i + half_rot];
        k_inout[base + i] = x0 * cos_t - x1 * sin_t;
        k_inout[base + i + half_rot] = x0 * sin_t + x1 * cos_t;
    }
    // K pass-through with norm applied in place.
    for (uint i = p.rope_dim + tid; i < stride; i += 64) {
        k_inout[base + i] = k_inout[base + i] * k_rms_inv * k_norm_w[i];
    }
}
