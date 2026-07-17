#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n;
    float eps;
    float hidden_scale;
};

// Triple-fused: residual_norm + residual_add + output_norm.
// Replaces (post_ffn_norm in-place) + barrier + (residual_rms_norm) with one
// dispatch + one barrier on the dense Gemma FFN/next-attn transition.
//
//   res_n[i] = residual_w[i] * residual[i] * rsqrt(mean(residual^2) + eps)
//   hidden[i] += res_n[i]
//   norm_out[i] = output_w[i] * hidden[i] * rsqrt(mean(hidden^2) + eps)
//
// Adapted from residual_rms_norm.metal (existing in-tree fusion) and
// the reference implementation `ggml-metal-ops.cpp::ggml_metal_op_rms_norm` op-fusion idea
// (residual+norm in one pass), extended to two reductions.
//
// 256 threads / 8 simdgroups per threadgroup. One token per threadgroup
// (group_id selects the row); register-cache hidden between the two
// reductions to avoid a third pass over the buffer.
#define N_SIMDGROUPS 8
#define SIMD_WIDTH 32
#define TG_SIZE (N_SIMDGROUPS * SIMD_WIDTH)
#define MAX_PER_THREAD 64

kernel void main0(
    constant Params& p [[buffer(0)]],
    device float* hidden [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device const float* residual_w [[buffer(3)]],
    device float* norm_out [[buffer(4)]],
    device const float* output_w [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial_sums_r[N_SIMDGROUPS];
    threadgroup float partial_sums_h[N_SIMDGROUPS];

    const uint base = group_id * p.n;

    // Pass 1: read residual, accumulate sum of squares for residual norm.
    float sum_sq_r = 0.0f;
    for (uint i = tid; i < p.n; i += TG_SIZE) {
        const float r = residual[base + i];
        sum_sq_r += r * r;
    }

    float sg_sum = simd_sum(sum_sq_r);
    if (lane == 0) partial_sums_r[sg_idx] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Mirror residual_rms_norm.metal: every simdgroup redundantly performs the
    // final 8-way reduction, avoiding the broadcast barrier through
    // partial_sums_r[0].
    const float part_r = (lane < N_SIMDGROUPS) ? partial_sums_r[lane] : 0.0f;
    const float total_sq_r = simd_sum(part_r);
    const float rms_inv_r = fast::rsqrt(fast::divide(total_sq_r, float(p.n)) + p.eps);

    // Pass 2: hidden += residual_w[i] * residual[i] * rms_inv_r;
    // accumulate sum of squares for hidden norm; cache new hidden in registers.
    // hidden write is multiplied by p.hidden_scale (folds the per-layer
    // layer_output_scale that was previously a separate scale_in_place
    // dispatch + barrier — saves ≈60 dispatches/60 barriers per token on
    // Gemma 31B). The cached h_vals stay unscaled so the second-pass norm
    // remains computed on the un-scaled residual stream, exactly matching
    // the original (post_norm_residual_rms_norm → barrier → scale_in_place)
    // ordering.
    float h_vals[MAX_PER_THREAD];
    float sum_sq_h = 0.0f;
    uint count = 0;
    const float hidden_scale = p.hidden_scale;
    const bool apply_hidden_scale = hidden_scale != 1.0f;
    for (uint i = tid; i < p.n; i += TG_SIZE) {
        const float r = residual[base + i];
        const float r_normed = residual_w[i] * (r * rms_inv_r);
        const float h = hidden[base + i] + r_normed;
        hidden[base + i] = apply_hidden_scale ? (h * hidden_scale) : h;
        h_vals[count++] = h;
        sum_sq_h += h * h;
    }

    // Use a separate scratch row for the second reduction so faster simdgroups
    // can start the hidden pass without waiting for every lane to finish reading
    // partial_sums_r above.
    sg_sum = simd_sum(sum_sq_h);
    if (lane == 0) partial_sums_h[sg_idx] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float part_h = (lane < N_SIMDGROUPS) ? partial_sums_h[lane] : 0.0f;
    const float total_sq_h = simd_sum(part_h);
    const float rms_inv_h = fast::rsqrt(fast::divide(total_sq_h, float(p.n)) + p.eps);

    // Pass 3: norm_out = output_w[i] * h * rms_inv_h.
    count = 0;
    for (uint i = tid; i < p.n; i += TG_SIZE) {
        norm_out[base + i] = output_w[i] * (h_vals[count++] * rms_inv_h);
    }
}
