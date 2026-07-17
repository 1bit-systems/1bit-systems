#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint n_used;
    uint src_stride;
    uint gate_scalar_offset;
    uint norm_offset;
    float hidden_scale;
};

// Exact Qwen3.6 token-major MoE finalize for hidden_dim=2048 when the
// shared-expert gate scalar was already materialized by the router.
kernel void main0(
    device float* accum [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const uint* routing [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    device const float* shared_src [[buffer(4)]],
    device const float* gate_scalar_buf [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    if (p.n != 2048u || p.n_used != 8u || p.src_stride != 2048u) {
        return;
    }

    device const float* gate_scalar = gate_scalar_buf + (p.gate_scalar_offset >> 2);
    const uint threads_per_tg = simdgroups_per_tg * 32u;

    const float w0_lane = (lane == 0u) ? as_type<float>(routing[8u]) : 0.0f;
    const float w1_lane = (lane == 0u) ? as_type<float>(routing[9u]) : 0.0f;
    const float w2_lane = (lane == 0u) ? as_type<float>(routing[10u]) : 0.0f;
    const float w3_lane = (lane == 0u) ? as_type<float>(routing[11u]) : 0.0f;
    const float w4_lane = (lane == 0u) ? as_type<float>(routing[12u]) : 0.0f;
    const float w5_lane = (lane == 0u) ? as_type<float>(routing[13u]) : 0.0f;
    const float w6_lane = (lane == 0u) ? as_type<float>(routing[14u]) : 0.0f;
    const float w7_lane = (lane == 0u) ? as_type<float>(routing[15u]) : 0.0f;
    const float gate_lane = (lane == 0u) ? fast::divide(1.0f, 1.0f + fast::exp(-gate_scalar[0])) : 0.0f;

    const float w0 = simd_broadcast(w0_lane, 0u);
    const float w1 = simd_broadcast(w1_lane, 0u);
    const float w2 = simd_broadcast(w2_lane, 0u);
    const float w3 = simd_broadcast(w3_lane, 0u);
    const float w4 = simd_broadcast(w4_lane, 0u);
    const float w5 = simd_broadcast(w5_lane, 0u);
    const float w6 = simd_broadcast(w6_lane, 0u);
    const float w7 = simd_broadcast(w7_lane, 0u);
    const float gate = simd_broadcast(gate_lane, 0u);
    for (uint id = tid; id < p.n; id += threads_per_tg) {
        float sum = w0 * src[id];
        sum = fma(w1, src[2048u + id], sum);
        sum = fma(w2, src[4096u + id], sum);
        sum = fma(w3, src[6144u + id], sum);
        sum = fma(w4, src[8192u + id], sum);
        sum = fma(w5, src[10240u + id], sum);
        sum = fma(w6, src[12288u + id], sum);
        sum = fma(w7, src[14336u + id], sum);
        accum[id] = (accum[id] + sum + gate * shared_src[id]) * p.hidden_scale;
    }
}
