#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint n_used;
    uint src_stride;
    uint gate_scalar_offset;
    uint norm_offset;
    float eps;
    float hidden_scale;
    uint base_hidden_offset;
    uint shared_src_offset;
};

// Exact Qwen3.6 token-major MoE finalize plus next-layer RMSNorm, seeded from
// a separate hidden row and a precomputed shared-expert gate scalar.
kernel void main0(
    device float* hidden [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const uint* routing [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    device const float* shared_src [[buffer(4)]],
    device const float* gate_scalar_buf [[buffer(5)]],
    device float* next_norm [[buffer(6)]],
    device const float* next_norm_weight [[buffer(7)]],
    device const float* base_hidden [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    if (p.n != 2048u || p.n_used != 8u || p.src_stride != 2048u) {
        return;
    }

    threadgroup float norm_partials[32];
    device const float* base = base_hidden + (p.base_hidden_offset >> 2);
    device const float* shared = shared_src + (p.shared_src_offset >> 2);
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

    float h_vals[2];
    uint h_idxs[2];
    uint h_count = 0u;
    float hidden_sq = 0.0f;
    for (uint id = tid; id < p.n; id += threads_per_tg) {
        float sum = w0 * src[id];
        sum = fma(w1, src[2048u + id], sum);
        sum = fma(w2, src[4096u + id], sum);
        sum = fma(w3, src[6144u + id], sum);
        sum = fma(w4, src[8192u + id], sum);
        sum = fma(w5, src[10240u + id], sum);
        sum = fma(w6, src[12288u + id], sum);
        sum = fma(w7, src[14336u + id], sum);

        const float h = (base[id] + sum + gate * shared[id]) * p.hidden_scale;
        hidden[id] = h;
        h_vals[h_count] = h;
        h_idxs[h_count] = id;
        h_count++;
        hidden_sq = fma(h, h, hidden_sq);
    }

    float sg_sum = simd_sum(hidden_sq);
    if (lane == 0u) {
        norm_partials[tid >> 5] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float part = (lane < simdgroups_per_tg) ? norm_partials[lane] : 0.0f;
    const float rms_inv = fast::rsqrt(fast::divide(simd_sum(part), float(p.n)) + p.eps);

    for (uint i = 0u; i < h_count; i++) {
        const uint id = h_idxs[i];
        next_norm[id] = next_norm_weight[id] * (h_vals[i] * rms_inv);
    }
}
