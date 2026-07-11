#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Fused single-token Q/K-norm + Q/K-RoPE + Q8 KV-cache write for the kv_cache_q8
// decode path. Extends cycle-22's `rope_qk_norm_inplace.metal` by folding the
// follow-up `kv_cache_write_q8` dispatch (and its barrier) into this kernel:
//   - Q-heads (head < n_q_heads): Q-norm + Q-rope, in place in q_inout.
//   - K-heads (head < n_q_heads + n_kv_heads): K-norm + K-rope kept entirely in
//     thread-private registers, then Q8-quantized directly into kv_k_cache at
//     `dst_offset_bytes` (current token slot) via an alternating-block layout
//     where each simdgroup owns blocks {simd_id, simd_id+2, ...} and each
//     thread quantizes only registers it itself produced — eliminating both
//     the cycle-23 mem_device threadgroup_barrier and the round-trip device
//     writes/reads through k_inout.
//   - V-heads (head < n_q_heads + 2*n_kv_heads): Q8-quantize v_inout directly
//     into kv_v_cache at `dst_offset_bytes`.
//
// Adapts the reference implementation `ggml_metal_op_concurrency_check/reset` single-consumer
// fusion (ggml-metal-ops.cpp:159, 175) to the QK-norm → rope → kv-write chain
// that the Q8 KV decode path of Qwen3.6 walks every dense full-attn layer.
// The rotated K in k_inout has no remaining consumer (flash_attn_q8 reads
// kv_k_cache, not k_buf), so the K writes are skipped entirely on the Q8 path.
//
// Layout invariant for the no-barrier K branch: thread `tid` owns indices
// {tid, tid+64, tid+128, ..., tid + 64*(blocks_per_simd-1)} in the head's
// rotated K row. With `simd_id ∈ {0,1}` and `simd_lane ∈ [0,32)`, this matches
// the alternating quantize block assignment exactly (block bi for simdgroup
// `simd_id` = block_in_head `simd_id + 2*bi`, elem_offset = tid + 64*bi). The
// rope loop (pair (i, i+half_rot)) and post-rope loop (rope_dim + j*64 + tid)
// preserve the {tid, tid+64, ...} layout when both `half_rot` and `rope_dim`
// are multiples of 64 — i.e. `rope_dim % 128 == 0`. The dispatch site gates
// on that condition; otherwise the cycle-22 unfused path runs instead.

struct RopeQkNormKvQ8Push {
    uint stride;            // head_dim
    uint rope_dim;
    uint n_q_heads;
    uint n_kv_heads;
    uint position;
    uint dst_offset_bytes;  // per-token byte offset within the layer's KV cache
    float eps;
};

kernel void main0(
    constant RopeQkNormKvQ8Push& p [[buffer(0)]],
    device float* q_inout       [[buffer(1)]],
    device float* k_inout       [[buffer(2)]],
    device const float* v_inout [[buffer(3)]],
    device const float* freqs   [[buffer(4)]],
    device const float* q_norm_w [[buffer(5)]],
    device const float* k_norm_w [[buffer(6)]],
    device uchar* kv_k_cache    [[buffer(7)]],
    device uchar* kv_v_cache    [[buffer(8)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    const uint stride = p.stride;
    const uint half_rot = p.rope_dim / 2;
    const uint blocks_per_head = stride / 32u;
    const uint blocks_per_simd = blocks_per_head / 2u;

    if (head < p.n_q_heads) {
        // Q branch — identical to rope_qk_norm_inplace.metal Q path.
        const uint base = head * stride;

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
        for (uint i = p.rope_dim + tid; i < stride; i += 64) {
            q_inout[base + i] = q_inout[base + i] * q_rms_inv * q_norm_w[i];
        }
        return;
    }

    if (head < p.n_q_heads + p.n_kv_heads) {
        // K branch — registers + alternating quantize, no barrier, no device
        // writes to k_inout. Each thread holds its blocks_per_simd K rope/post-
        // rope outputs in `k_out[]` and feeds them straight into the alternating
        // quantize loop, where the per-thread→per-block mapping `block_in_head
        // = simd_id + 2*bi`, `elem_offset = tid + 64*bi` matches the rope+post-
        // rope write layout exactly. Drops the cycle-23 threadgroup_barrier
        // (mem_device) and the round-trip through k_inout.
        const uint kv_head = head - p.n_q_heads;
        const uint base = kv_head * stride;

        float sum_sq = 0.0f;
        for (uint i = simd_lane; i < stride; i += 32u) {
            const float v = k_inout[base + i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        const float k_rms_inv = fast::rsqrt(fast::divide(sum_sq, float(stride)) + p.eps);

        // Up to head_dim=512 (blocks_per_simd=8). Apple9 register file (24 KiB
        // per simdgroup at peak occupancy) easily covers this 8 × float per
        // lane allocation.
        float k_out[8];
        const uint half_rot_blocks = half_rot >> 6;   // # of 64-stride steps fitting in half_rot
        const uint rope_dim_blocks = p.rope_dim >> 6; // # of 64-stride steps fitting in rope_dim

        // Rope writes: pair (i, i+half_rot) with i = tid + j*64, j ∈ [0, half_rot/64).
        // Output index `i` lands in k_out[j]; output index `i+half_rot` lands in
        // k_out[j + half_rot/64] (since half_rot % 64 == 0 under our gate).
        for (uint j = 0u, i = tid; i < half_rot; ++j, i += 64u) {
            const float theta = float(p.position) * freqs[i];
            const float cos_t = fast::cos(theta);
            const float sin_t = fast::sin(theta);
            const float x0 = k_inout[base + i] * k_rms_inv * k_norm_w[i];
            const float x1 = k_inout[base + i + half_rot] * k_rms_inv * k_norm_w[i + half_rot];
            k_out[j] = x0 * cos_t - x1 * sin_t;
            k_out[j + half_rot_blocks] = x0 * sin_t + x1 * cos_t;
        }
        // Post-rope writes: index rope_dim + j*64 + tid lands in
        // k_out[rope_dim/64 + j] (rope_dim % 64 == 0 under our gate).
        for (uint j = 0u, i = p.rope_dim + tid; i < stride; ++j, i += 64u) {
            k_out[rope_dim_blocks + j] = k_inout[base + i] * k_rms_inv * k_norm_w[i];
        }

        // Quantize: alternating blocks per simdgroup. SG simd_id owns blocks
        // {simd_id, simd_id+2, simd_id+4, ...}; thread (simd_id, simd_lane)
        // quantizes register k_out[bi] into block_in_head = simd_id + 2*bi.
        const uint kv_block_base = kv_head * blocks_per_head;
        for (uint bi = 0u; bi < blocks_per_simd; ++bi) {
            const uint block_in_head = simd_id + 2u * bi;
            const float k_val = k_out[bi];
            const float k_abs_max = simd_max(fast::abs(k_val));
            const float k_scale = k_abs_max > 0.0f ? k_abs_max * (1.0f / 127.0f) : 0.0f;
            const float k_inv_scale = k_scale > 0.0f ? 1.0f / k_scale : 0.0f;

            device uchar* k_dst = kv_k_cache + p.dst_offset_bytes + (kv_block_base + block_in_head) * 34u;
            if (simd_lane == 0u) {
                *(device ushort*)(k_dst) = as_type<ushort>(half(k_scale));
            }
            const int q = clamp(int(rint(k_val * k_inv_scale)), -127, 127);
            k_dst[2u + simd_lane] = as_type<uchar>(char(q));
        }
        return;
    }

    // V branch — Q8 quantize+write only (V needs neither rope nor norm on the
    // Qwen3.6 Q8 KV path). V has no in-kernel intermediate so the original
    // sequential per-simdgroup block layout is kept: SG 0 writes blocks
    // 0..blocks_per_simd-1 and SG 1 writes the remaining blocks contiguously,
    // which keeps the kv_v_cache writes from interleaving cache lines across
    // simdgroups.
    const uint v_head = head - p.n_q_heads - p.n_kv_heads;
    const uint base = v_head * stride;
    const uint kv_block_base = v_head * blocks_per_head;
    for (uint bi = 0u; bi < blocks_per_simd; ++bi) {
        const uint block_in_head = simd_id * blocks_per_simd + bi;
        const uint elem_offset = block_in_head * 32u + simd_lane;
        const float v_val = v_inout[base + elem_offset];
        const float v_abs_max = simd_max(fast::abs(v_val));
        const float v_scale = v_abs_max > 0.0f ? v_abs_max * (1.0f / 127.0f) : 0.0f;
        const float v_inv_scale = v_scale > 0.0f ? 1.0f / v_scale : 0.0f;

        device uchar* v_dst = kv_v_cache + p.dst_offset_bytes + (kv_block_base + block_in_head) * 34u;
        if (simd_lane == 0u) {
            *(device ushort*)(v_dst) = as_type<ushort>(half(v_scale));
        }
        const int q = clamp(int(rint(v_val * v_inv_scale)), -127, 127);
        v_dst[2u + simd_lane] = as_type<uchar>(char(q));
    }
}
