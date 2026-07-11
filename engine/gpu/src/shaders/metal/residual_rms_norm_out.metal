#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n;
    float eps;
    float scale;
    uint input_offset;
    uint residual_offset;
    uint output_offset;
    uint norm_offset;
};

// Separate-output residual-add + RMS norm:
//   out = input + scale * residual
//   norm_out = weights * rms_norm(out)
//
// This keeps immutable prompt embeddings intact while avoiding the old
// copy-then-in-place residual_rms_norm pair in Qwen layer-0 prefill.
#define N_SIMDGROUPS 8
#define SIMD_WIDTH 32
#define TG_SIZE (N_SIMDGROUPS * SIMD_WIDTH)
#define MAX_PER_THREAD 128

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* norm_out [[buffer(4)]],
    device const float* weights [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial_sums[N_SIMDGROUPS];

    const uint base = group_id * p.n;
    const uint input_base = p.input_offset + base;
    const uint residual_base = p.residual_offset + base;
    const uint output_base = p.output_offset + base;
    const uint norm_base = p.norm_offset + base;

    float vals[MAX_PER_THREAD];
    float sum_sq = 0.0f;
    uint count = 0;
    for (uint i = tid; i < p.n; i += TG_SIZE) {
        const float h = fma(p.scale, residual[residual_base + i], input[input_base + i]);
        output[output_base + i] = h;
        vals[count++] = h;
        sum_sq += h * h;
    }

    const float sg_sum = simd_sum(sum_sq);
    if (lane == 0u) {
        partial_sums[sg_idx] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float part = (lane < N_SIMDGROUPS) ? partial_sums[lane] : 0.0f;
    const float total_sq = simd_sum(part);
    const float rms_inv = fast::rsqrt(fast::divide(total_sq, float(p.n)) + p.eps);

    count = 0;
    for (uint i = tid; i < p.n; i += TG_SIZE) {
        norm_out[norm_base + i] = weights[i] * (vals[count++] * rms_inv);
    }
}
