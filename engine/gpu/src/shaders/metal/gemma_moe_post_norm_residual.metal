#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Push {
    uint n;
    float eps;
    uint expert_weight_offset;
    uint shared_weight_offset;
    uint has_gate;
    uint has_final_norm;
};

#define MAX_PER_THREAD 16

kernel void main0(
    constant Push& p [[buffer(0)]],
    device const float* expert_in [[buffer(1)]],
    device const float* shared_in [[buffer(2)]],
    device float* hidden [[buffer(3)]],
    device const float* expert_weights_base [[buffer(4)]],
    device const float* shared_weights_base [[buffer(5)]],
    device const float* final_weights [[buffer(6)]],
    device const float* gate_buf [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint subgroup_size [[thread_execution_width]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float expert_sums[32];
    threadgroup float shared_sums[32];
    threadgroup float combined_sums[32];

    device const float* expert_weights = expert_weights_base + p.expert_weight_offset;
    device const float* shared_weights = shared_weights_base + p.shared_weight_offset;
    const float gate = p.has_gate != 0 ? 1.0f / (1.0f + exp(-gate_buf[0])) : 1.0f;

    float expert_vals[MAX_PER_THREAD];
    float shared_vals[MAX_PER_THREAD];
    float combined_vals[MAX_PER_THREAD];
    uint cached = 0;

    float expert_sum_sq = 0.0f;
    float shared_sum_sq = 0.0f;
    for (uint i = tid; i < p.n; i += tg_size) {
        const float e = expert_in[i];
        const float s = shared_in[i];
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
    if (simd_lane == 0) {
        expert_sums[simd_group] = expert_sum_sq;
        shared_sums[simd_group] = shared_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_groups = (tg_size + subgroup_size - 1) / subgroup_size;
    float expert_total = (simd_lane < n_groups) ? expert_sums[simd_lane] : 0.0f;
    float shared_total = (simd_lane < n_groups) ? shared_sums[simd_lane] : 0.0f;
    expert_total = simd_sum(expert_total);
    shared_total = simd_sum(shared_total);
    const float expert_rms = fast::rsqrt(fast::divide(expert_total, float(p.n)) + p.eps);
    const float shared_rms = fast::rsqrt(fast::divide(shared_total, float(p.n)) + p.eps);

    float combined_sum_sq = 0.0f;
    uint c = 0;
    for (uint i = tid; i < p.n; i += tg_size) {
        const float e = (c < MAX_PER_THREAD) ? expert_vals[c] : expert_in[i];
        const float s = (c < MAX_PER_THREAD) ? shared_vals[c] : shared_in[i];
        const float combined = expert_weights[i] * (e * expert_rms) +
            gate * shared_weights[i] * (s * shared_rms);
        if (c < MAX_PER_THREAD) {
            combined_vals[c] = combined;
        }
        combined_sum_sq = fma(combined, combined, combined_sum_sq);
        c++;
    }

    combined_sum_sq = simd_sum(combined_sum_sq);
    if (simd_lane == 0) {
        combined_sums[simd_group] = combined_sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float combined_total = (simd_lane < n_groups) ? combined_sums[simd_lane] : 0.0f;
    combined_total = simd_sum(combined_total);
    const float combined_rms = p.has_final_norm != 0
        ? fast::rsqrt(fast::divide(combined_total, float(p.n)) + p.eps)
        : 1.0f;

    c = 0;
    for (uint i = tid; i < p.n; i += tg_size) {
        const float combined = (c < MAX_PER_THREAD) ? combined_vals[c] : 0.0f;
        const float out = p.has_final_norm != 0
            ? final_weights[i] * (combined * combined_rms)
            : combined;
        hidden[i] += out;
        c++;
    }
}
