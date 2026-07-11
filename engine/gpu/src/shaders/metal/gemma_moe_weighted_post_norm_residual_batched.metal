#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Push {
    uint n_tokens;
    uint hidden_dim;
    uint k;
    uint routing_stride;
    uint scale_offset;
    uint expert_weight_offset;
    uint shared_weight_offset;
    uint has_gate;
    uint has_final_norm;
    float eps;
};

#define MAX_PER_THREAD 16
#define MAX_K_USED 16

kernel void main0(
    constant Push& p [[buffer(0)]],
    device const float* expert_down [[buffer(1)]],
    device const uint* routing [[buffer(2)]],
    device const float* expert_scales_base [[buffer(3)]],
    device const float* shared_in [[buffer(4)]],
    device float* hidden [[buffer(5)]],
    device const float* expert_weights_base [[buffer(6)]],
    device const float* shared_weights_base [[buffer(7)]],
    device const float* final_weights [[buffer(8)]],
    device const float* gate_buf [[buffer(9)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint subgroup_size [[thread_execution_width]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float expert_sums[32];
    threadgroup float shared_sums[32];
    threadgroup float combined_sums[32];
    threadgroup float route_weights[MAX_K_USED];

    if (token >= p.n_tokens || p.hidden_dim == 0u || p.k == 0u || p.k > MAX_K_USED) {
        return;
    }

    device const uint* route_row = routing + token * p.routing_stride;
    device const float* expert_scales = expert_scales_base + p.scale_offset;
    device const float* expert_weights = expert_weights_base + p.expert_weight_offset;
    device const float* shared_weights = shared_weights_base + p.shared_weight_offset;
    device const float* shared_row = shared_in + token * p.hidden_dim;
    device float* hidden_row = hidden + token * p.hidden_dim;
    const float gate = p.has_gate != 0u ? 1.0f / (1.0f + exp(-gate_buf[token])) : 1.0f;

    if (tid < p.k) {
        const uint expert_id = route_row[tid];
        route_weights[tid] = as_type<float>(route_row[p.k + tid]) * expert_scales[expert_id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float expert_vals[MAX_PER_THREAD];
    float shared_vals[MAX_PER_THREAD];
    float combined_vals[MAX_PER_THREAD];
    uint cached = 0u;

    float expert_sum_sq = 0.0f;
    float shared_sum_sq = 0.0f;
    for (uint i = tid; i < p.hidden_dim; i += tg_size) {
        float e = 0.0f;
        if (p.k == 8u) {
            e = fma(route_weights[0], expert_down[(token * p.k + 0u) * p.hidden_dim + i], e);
            e = fma(route_weights[1], expert_down[(token * p.k + 1u) * p.hidden_dim + i], e);
            e = fma(route_weights[2], expert_down[(token * p.k + 2u) * p.hidden_dim + i], e);
            e = fma(route_weights[3], expert_down[(token * p.k + 3u) * p.hidden_dim + i], e);
            e = fma(route_weights[4], expert_down[(token * p.k + 4u) * p.hidden_dim + i], e);
            e = fma(route_weights[5], expert_down[(token * p.k + 5u) * p.hidden_dim + i], e);
            e = fma(route_weights[6], expert_down[(token * p.k + 6u) * p.hidden_dim + i], e);
            e = fma(route_weights[7], expert_down[(token * p.k + 7u) * p.hidden_dim + i], e);
        } else {
            for (uint slot = 0u; slot < p.k; slot++) {
                e = fma(route_weights[slot], expert_down[(token * p.k + slot) * p.hidden_dim + i], e);
            }
        }
        const float s = shared_row[i];
        expert_sum_sq = fma(e, e, expert_sum_sq);
        shared_sum_sq = fma(s, s, shared_sum_sq);
        if (cached < MAX_PER_THREAD) {
            expert_vals[cached] = e;
            shared_vals[cached] = s;
            cached++;
        }
    }

    expert_sum_sq = simd_sum(expert_sum_sq);
    shared_sum_sq = simd_sum(shared_sum_sq);
    if (simd_lane == 0u) {
        expert_sums[simd_group] = expert_sum_sq;
        shared_sums[simd_group] = shared_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_groups = (tg_size + subgroup_size - 1u) / subgroup_size;
    float expert_total = (simd_lane < n_groups) ? expert_sums[simd_lane] : 0.0f;
    float shared_total = (simd_lane < n_groups) ? shared_sums[simd_lane] : 0.0f;
    expert_total = simd_sum(expert_total);
    shared_total = simd_sum(shared_total);
    const float expert_rms = fast::rsqrt(fast::divide(expert_total, float(p.hidden_dim)) + p.eps);
    const float shared_rms = fast::rsqrt(fast::divide(shared_total, float(p.hidden_dim)) + p.eps);

    float combined_sum_sq = 0.0f;
    uint c = 0u;
    for (uint i = tid; i < p.hidden_dim; i += tg_size) {
        const float e = (c < MAX_PER_THREAD) ? expert_vals[c] : 0.0f;
        const float s = (c < MAX_PER_THREAD) ? shared_vals[c] : shared_row[i];
        const float combined = expert_weights[i] * (e * expert_rms) +
            gate * shared_weights[i] * (s * shared_rms);
        if (c < MAX_PER_THREAD) {
            combined_vals[c] = combined;
        }
        combined_sum_sq = fma(combined, combined, combined_sum_sq);
        c++;
    }

    combined_sum_sq = simd_sum(combined_sum_sq);
    if (simd_lane == 0u) {
        combined_sums[simd_group] = combined_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float combined_total = (simd_lane < n_groups) ? combined_sums[simd_lane] : 0.0f;
    combined_total = simd_sum(combined_total);
    const float combined_rms = p.has_final_norm != 0u
        ? fast::rsqrt(fast::divide(combined_total, float(p.hidden_dim)) + p.eps)
        : 1.0f;

    c = 0u;
    for (uint i = tid; i < p.hidden_dim; i += tg_size) {
        const float combined = (c < MAX_PER_THREAD) ? combined_vals[c] : 0.0f;
        const float out = p.has_final_norm != 0u
            ? final_weights[i] * (combined * combined_rms)
            : combined;
        hidden_row[i] += out;
        c++;
    }
}
