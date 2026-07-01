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

// Exact Qwen3.6 token-major MoE finalize for hidden_dim=2048.
//
// The generic kernel splits the hidden row into 512-wide tiles, so it
// recomputes sigmoid(dot(norm, shared_gate_weight)) once per tile. This variant
// runs one 1024-thread group, reduces the gate dot once through simdgroup
// partials, and lets each thread update two hidden dimensions.
kernel void main0(
    device float* accum [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const uint* routing [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    device const float* shared_src [[buffer(4)]],
    device const float* norm_src [[buffer(5)]],
    device const char* gate_weight_bytes [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    if (p.n != 2048u || p.n_used != 8u || p.src_stride != 2048u) {
        return;
    }

    threadgroup float partials[32];
    threadgroup float route_weights[8];
    device const float* gate_weight = (device const float*)(gate_weight_bytes + p.gate_weight_offset);
    device const float* norm = norm_src + (p.norm_offset >> 2);
    const uint threads_per_tg = simdgroups_per_tg * 32u;
    if (tid < 8u) {
        route_weights[tid] = as_type<float>(routing[8u + tid]);
    }

    float dot = 0.0f;
    for (uint dim = tid; dim < p.n; dim += threads_per_tg) {
        dot = fma(gate_weight[dim], norm[dim], dot);
    }

    const float simd_sum_dot = simd_sum(dot);
    if (lane == 0u) {
        partials[simdgroup] = simd_sum_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float part = (lane < simdgroups_per_tg) ? partials[lane] : 0.0f;
    const float group_sum = simd_sum(part);
    const float gate = fast::divide(1.0f, 1.0f + fast::exp(-group_sum));

    for (uint id = tid; id < p.n; id += threads_per_tg) {
        float sum = route_weights[0] * src[id];
        sum = fma(route_weights[1], src[2048u + id], sum);
        sum = fma(route_weights[2], src[4096u + id], sum);
        sum = fma(route_weights[3], src[6144u + id], sum);
        sum = fma(route_weights[4], src[8192u + id], sum);
        sum = fma(route_weights[5], src[10240u + id], sum);
        sum = fma(route_weights[6], src[12288u + id], sum);
        sum = fma(route_weights[7], src[14336u + id], sum);
        accum[id] = (accum[id] + sum + gate * shared_src[id]) * p.hidden_scale;
    }
}
