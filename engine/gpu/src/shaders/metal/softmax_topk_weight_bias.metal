#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n_experts;
    uint k;
    uint bias_offset;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* logits [[buffer(1)]],
    device uint* output_data [[buffer(2)]],
    device const float* bias [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint subgroup_size [[thread_execution_width]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float values[256];
    threadgroup float reduce_val[64];
    threadgroup float selected_val[16];
    threadgroup float local_val[64];
    threadgroup uint local_idx[64];

    const uint bias_base = p.bias_offset / 4u;
    for (uint i = tid; i < p.n_experts; i += 64u) {
        values[i] = logits[i] + bias[bias_base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint ki = 0u; ki < p.k; ki++) {
        float best_val = -INFINITY;
        uint best_idx = 0u;
        for (uint i = tid; i < p.n_experts; i += 64u) {
            const float v = values[i];
            if (v > best_val) {
                best_val = v;
                best_idx = i;
            }
        }

        const float wave_best = simd_max(best_val);
        if (subgroup_size < 64u) {
            if (simd_lane == 0u) {
                reduce_val[simd_group] = wave_best;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                const uint n_groups = (64u + subgroup_size - 1u) / subgroup_size;
                float merged = -INFINITY;
                for (uint sg = 0u; sg < n_groups; sg++) {
                    merged = max(merged, reduce_val[sg]);
                }
                local_val[0] = merged;
            }
        } else if (tid == 0u) {
            local_val[0] = wave_best;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        local_val[tid] = (best_val == local_val[0]) ? best_val : -INFINITY;
        local_idx[tid] = best_idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float global_best = -INFINITY;
            uint global_idx = 0u;
            for (uint lane = 0u; lane < 64u; lane++) {
                if (local_val[lane] > global_best) {
                    global_best = local_val[lane];
                    global_idx = local_idx[lane];
                }
            }
            output_data[ki] = global_idx;
            selected_val[ki] = global_best;
            values[global_idx] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float max_sel = -INFINITY;
        for (uint i = 0u; i < p.k; i++) {
            max_sel = max(max_sel, selected_val[i]);
        }

        float sum = 0.0f;
        for (uint i = 0u; i < p.k; i++) {
            const float e = exp(selected_val[i] - max_sel);
            selected_val[i] = e;
            sum += e;
        }

        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (uint i = 0u; i < p.k; i++) {
            output_data[p.k + i] = as_type<uint>(selected_val[i] * inv_sum);
        }
    }
}
