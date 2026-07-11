#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Fused single-token RoPE-Q + RoPE-K + KV cache write. Adapted from
// the reference implementation `ggml_metal_op_rope_set_rows` which folds rope output into the
// destination KV slot in one kernel; extended here to also rotate the Q
// vector in the same dispatch so the dense Gemma decode path collapses
// (Q-rope, K-rope+kv-write, V-norm+kv-write) into a single kernel,
// removing one dispatch per dense full-attn layer (≈60/token on Gemma 31B).
//
// Each threadgroup handles one head slot:
//   - if head < n_q_heads: rotates q_inout[head] in place (Q stream).
//   - else (kv_head = head - n_q_heads): rotates K[kv_head] into the cache
//     slot, copies V[kv_head] into the cache (RMS-normalized when
//     apply_v_norm != 0).
//
// When apply_v_norm != 0, V is RMS-normalized per head with unit weights
// before being written to v_cache (Gemma SWA path). This subsumes the
// separate `dispatchRmsNormOnCmd` over v_buf that previously preceded this
// kernel.
//
// When apply_qk_norm != 0, Q and K are per-head RMS-normalized with the
// supplied head_dim weight vectors (q_norm_w, k_norm_w) before RoPE. This
// subsumes the two standalone `dispatchRmsNormOnCmd` calls over q_buf and
// k_buf that previously preceded the fused rope+kv-write dispatch — one
// fewer dispatch per dense full-attn layer for Q-norm and one for K-norm
// (≈60+60/token on Gemma 31B), extending the reference implementation `ggml_metal_op_rms_norm`
// op-fusion to the attention prep path.

struct RopeKvCacheWritePush {
    uint stride;        // elements per head (head_dim)
    uint rope_dim;      // number of rotary dimensions per head (<= stride)
    uint n_q_heads;     // grid slots [0, n_q_heads) handle Q-rope; later slots handle K/V
    uint position;      // token position for this step
    uint dst_offset;    // element offset into kv_k_cache / kv_v_cache (= position * kv_dim)
    uint apply_v_norm;  // 0 = copy V verbatim; nonzero = RMS-normalize V (unit weights)
    uint apply_qk_norm; // 0 = no Q/K norm; nonzero = RMS-normalize Q and K with head_dim weights
    float eps;          // RMS norm epsilon (used by apply_v_norm and apply_qk_norm)
};

kernel void main0(
    constant RopeKvCacheWritePush& p [[buffer(0)]],
    device float* q_inout      [[buffer(1)]],
    device const float* k_in   [[buffer(2)]],
    device const float* v_in   [[buffer(3)]],
    device const float* freqs  [[buffer(4)]],
    device float* k_cache      [[buffer(5)]],
    device float* v_cache      [[buffer(6)]],
    device const float* q_norm_w [[buffer(7)]],
    device const float* k_norm_w [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint subgroup_size [[thread_execution_width]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint stride = p.stride;
    const uint half_rot = p.rope_dim / 2;

    if (head < p.n_q_heads) {
        // Q rotary (with optional Q-norm): rotate in place. Pass-through
        // dims (rope_dim..stride) are normalized when apply_qk_norm != 0,
        // otherwise left untouched (matches the standalone rope_native
        // kernel's behavior).
        const uint base = head * stride;

        float q_rms_inv = 1.0f;

        if (p.apply_qk_norm != 0u) {
            // Each simdgroup independently sums the FULL stride (head_dim=128
            // for Qwen3-8B → 4 elements/lane), then simd_sum gives the complete
            // sum-of-squares per simdgroup. No threadgroup memory, no barrier.
            // Work is duplicated 2× across simdgroups but each only reads ~512 B
            // (L1-resident) so cost is negligible; eliminates 1 threadgroup
            // barrier per Q-head dispatch (32 heads × 36 layers = 1152
            // dispatches/token on Qwen3-8B). Extends the cycle 17 broadcast-
            // barrier elimination to also eliminate the partial-reduction one.
            float sum_sq = 0.0f;
            for (uint i = simd_lane; i < stride; i += 32u) {
                const float v = q_inout[base + i];
                sum_sq += v * v;
            }
            sum_sq = simd_sum(sum_sq);
            q_rms_inv = fast::rsqrt(fast::divide(sum_sq, float(stride)) + p.eps);
        }

        for (uint i = tid; i < half_rot; i += 64) {
            const float theta = float(p.position) * freqs[i];
            const float cos_t = fast::cos(theta);
            const float sin_t = fast::sin(theta);
            float x0 = q_inout[base + i];
            float x1 = q_inout[base + i + half_rot];
            if (p.apply_qk_norm != 0u) {
                x0 = x0 * q_rms_inv * q_norm_w[i];
                x1 = x1 * q_rms_inv * q_norm_w[i + half_rot];
            }
            q_inout[base + i] = x0 * cos_t - x1 * sin_t;
            q_inout[base + i + half_rot] = x0 * sin_t + x1 * cos_t;
        }

        if (p.apply_qk_norm != 0u) {
            // Pass-through dims (rope_dim..stride): apply norm in place.
            for (uint i = p.rope_dim + tid; i < stride; i += 64) {
                q_inout[base + i] = q_inout[base + i] * q_rms_inv * q_norm_w[i];
            }
        }
        return;
    }

    const uint kv_head = head - p.n_q_heads;
    const uint base = kv_head * stride;
    const uint dst_base = p.dst_offset + base;

    float k_rms_inv = 1.0f;

    if (p.apply_qk_norm != 0u) {
        // Same per-simdgroup-redundant pattern as the Q-norm branch above:
        // each simdgroup independently sums the full stride and computes its
        // own rms_inv, eliminating shmem + threadgroup barrier entirely.
        float sum_sq = 0.0f;
        for (uint i = simd_lane; i < stride; i += 32u) {
            const float v = k_in[base + i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        k_rms_inv = fast::rsqrt(fast::divide(sum_sq, float(stride)) + p.eps);
    }

    // K rotary: apply RoPE pair (i, i+half_rot) and write to cache.
    for (uint i = tid; i < half_rot; i += 64) {
        const float theta = float(p.position) * freqs[i];
        const float cos_t = fast::cos(theta);
        const float sin_t = fast::sin(theta);
        float x0 = k_in[base + i];
        float x1 = k_in[base + i + half_rot];
        if (p.apply_qk_norm != 0u) {
            x0 = x0 * k_rms_inv * k_norm_w[i];
            x1 = x1 * k_rms_inv * k_norm_w[i + half_rot];
        }
        k_cache[dst_base + i] = x0 * cos_t - x1 * sin_t;
        k_cache[dst_base + i + half_rot] = x0 * sin_t + x1 * cos_t;
    }

    // K pass-through: dimensions beyond rope_dim are copied verbatim
    // (or normalized when apply_qk_norm != 0).
    for (uint i = p.rope_dim + tid; i < stride; i += 64) {
        float v = k_in[base + i];
        if (p.apply_qk_norm != 0u) v = v * k_rms_inv * k_norm_w[i];
        k_cache[dst_base + i] = v;
    }

    if (p.apply_v_norm != 0u) {
        // Same per-simdgroup full-stride reduction used by the Q/K norm
        // branches above: both simdgroups compute the complete V RMS scale,
        // avoiding threadgroup memory and two barriers per KV head. The extra
        // 2x read is only ~1 KiB/head on Gemma 31B and stays L1-resident.
        float sum_sq = 0.0f;
        for (uint i = simd_lane; i < stride; i += 32u) {
            const float v = v_in[base + i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        const float rms_inv = rsqrt((sum_sq / float(stride)) + p.eps);

        for (uint i = tid; i < stride; i += 64) {
            v_cache[dst_base + i] = v_in[base + i] * rms_inv;
        }
    } else {
        // V copy.
        for (uint i = tid; i < stride; i += 64) {
            v_cache[dst_base + i] = v_in[base + i];
        }
    }
}
