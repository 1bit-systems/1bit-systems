#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint n_used;
    uint src_stride;
    uint gate_weight_offset;
    uint norm_offset;
    float hidden_scale;
};

// Token-major Qwen MoE finalize with one-row F32 shared gate.
//
// This is the single-token analogue of the route-packed F32 shared-gate
// combine kernel: compute sigmoid(dot(norm, gate_weight)) once per output tile,
// then fold the weighted top-k expert sum and gated shared expert into hidden.
// The loop uses the actual dispatch width so the host can use 512-thread tiles
// for Qwen3.6 hidden_dim=2048, halving the redundant gate-dot work versus the
// original fixed 256-thread tile.
kernel void main0(
    device float* accum [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const uint* routing [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    device const float* shared_src [[buffer(4)]],
    device const float* norm_src [[buffer(5)]],
    device const char* gate_weight_bytes [[buffer(6)]],
    uint tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    if (p.n == 0u) {
        return;
    }

    threadgroup float partials[16];
    threadgroup float gate_value;

    device const float* gate_weight = (device const float*)(gate_weight_bytes + p.gate_weight_offset);
    device const float* norm = norm_src + (p.norm_offset >> 2);
    const uint threads_per_tg = uint(simdgroups_per_tg) * 32u;

    float dot = 0.0f;
    for (uint dim = tid; dim < p.n; dim += threads_per_tg) {
        dot = fma(gate_weight[dim], norm[dim], dot);
    }

    const float simd_sum_dot = simd_sum(dot);
    if (lane == 0) {
        partials[simdgroup] = simd_sum_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        const float part = (lane < simdgroups_per_tg) ? partials[lane] : 0.0f;
        const float group_sum = simd_sum(part);
        if (lane == 0) {
            gate_value = 1.0f / (1.0f + exp(-group_sum));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint id = tile * threads_per_tg + tid;
    if (id >= p.n) {
        return;
    }

    float sum = 0.0f;
    for (uint expert = 0u; expert < p.n_used; expert++) {
        const float weight = as_type<float>(routing[p.n_used + expert]);
        sum += weight * src[expert * p.src_stride + id];
    }

    accum[id] = (accum[id] + sum + gate_value * shared_src[id]) * p.hidden_scale;
}
