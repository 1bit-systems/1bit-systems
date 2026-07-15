#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n;
    float eps;
    float hidden_scale;
};

// Shape-specific sibling of post_norm_residual_rms_norm.metal for dense
// Gemma 31B's hidden_dim=5376 decode tail. Same math, 1024 threads to expose
// the maximum Apple9 per-threadgroup parallelism on the two reductions.
#define N_SIMDGROUPS 32
#define SIMD_WIDTH 32
#define TG_SIZE (N_SIMDGROUPS * SIMD_WIDTH)
// This variant is only dispatched for n=5376. Mirror the reference implementation's aligned
// `kernel_rms_norm_fuse_impl<float4, ...>` path: operate on four contiguous
// elements per lane so the two reductions and the final write use fewer loop
// iterations and memory instructions. For n=5376, each thread owns at most two
// float4 lanes; cache the post-residual values so final norm still uses the
// unscaled hidden stream even when p.hidden_scale is folded into hidden writes.
#define MAX_VEC_PER_THREAD 2

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
    const uint n_vec4 = p.n >> 2;
    const uint vec4_base = group_id * n_vec4;
    const uint scalar_tail = n_vec4 << 2;

    device float4* hidden4 = (device float4*)hidden;
    device const float4* residual4 = (device const float4*)residual;
    device const float4* residual_w4 = (device const float4*)residual_w;
    device float4* norm_out4 = (device float4*)norm_out;
    device const float4* output_w4 = (device const float4*)output_w;

    float sum_sq_r = 0.0f;
    for (uint i = tid; i < n_vec4; i += TG_SIZE) {
        const float4 r = residual4[vec4_base + i];
        sum_sq_r += dot(r, r);
    }
    for (uint i = scalar_tail + tid; i < p.n; i += TG_SIZE) {
        const float r = residual[base + i];
        sum_sq_r = fma(r, r, sum_sq_r);
    }

    float sg_sum = simd_sum(sum_sq_r);
    if (lane == 0) partial_sums_r[sg_idx] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float part_r = (lane < N_SIMDGROUPS) ? partial_sums_r[lane] : 0.0f;
    const float total_sq_r = simd_sum(part_r);
    const float rms_inv_r = fast::rsqrt(fast::divide(total_sq_r, float(p.n)) + p.eps);

    float sum_sq_h = 0.0f;
    float4 h_vals[MAX_VEC_PER_THREAD];
    float h_tail = 0.0f;
    bool has_h_tail = false;
    uint count = 0;
    const float hidden_scale = p.hidden_scale;
    const bool apply_hidden_scale = hidden_scale != 1.0f;
    for (uint i = tid; i < n_vec4; i += TG_SIZE) {
        const float4 r = residual4[vec4_base + i];
        const float4 r_normed = residual_w4[i] * (r * rms_inv_r);
        const float4 h = hidden4[vec4_base + i] + r_normed;
        hidden4[vec4_base + i] = apply_hidden_scale ? (h * hidden_scale) : h;
        if (count < MAX_VEC_PER_THREAD) {
            h_vals[count] = h;
        }
        count++;
        sum_sq_h += dot(h, h);
    }
    for (uint i = scalar_tail + tid; i < p.n; i += TG_SIZE) {
        const float r = residual[base + i];
        const float r_normed = residual_w[i] * (r * rms_inv_r);
        const float h = hidden[base + i] + r_normed;
        hidden[base + i] = apply_hidden_scale ? (h * hidden_scale) : h;
        h_tail = h;
        has_h_tail = true;
        sum_sq_h = fma(h, h, sum_sq_h);
    }

    sg_sum = simd_sum(sum_sq_h);
    if (lane == 0) partial_sums_h[sg_idx] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float part_h = (lane < N_SIMDGROUPS) ? partial_sums_h[lane] : 0.0f;
    const float total_sq_h = simd_sum(part_h);
    const float rms_inv_h = fast::rsqrt(fast::divide(total_sq_h, float(p.n)) + p.eps);

    count = 0;
    for (uint i = tid; i < n_vec4; i += TG_SIZE) {
        const float4 h = (count < MAX_VEC_PER_THREAD) ? h_vals[count] : hidden4[vec4_base + i];
        norm_out4[vec4_base + i] = output_w4[i] * (h * rms_inv_h);
        count++;
    }
    if (has_h_tail) {
        for (uint i = scalar_tail + tid; i < p.n; i += TG_SIZE) {
            norm_out[base + i] = output_w[i] * (h_tail * rms_inv_h);
        }
    }
}
