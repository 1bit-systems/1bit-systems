#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Fused GPT-OSS router: F32 matvec over <=32 experts, add f32 router bias,
// select top-k by biased logit, then softmax only over the selected experts.

struct RouterF32TopkBiasPush {
    uint n_experts;
    uint K;
    uint k;
    uint a_offset;
    uint x_offset;
    uint bias_offset;
};

#define TG_SIZE 512
#define ROWS_PER_TG ((TG_SIZE / 32) * 2)
#define MAX_K_VEC4 1024
#define MAX_EXPERTS 64
#define MAX_K_USED 16

kernel void main0(
    device const float* W [[buffer(0)]],
    constant RouterF32TopkBiasPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device uint* output_data [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float4 x_cache4[MAX_K_VEC4];
    threadgroup float values[MAX_EXPERTS];
    threadgroup float selected_val[MAX_K_USED];

    if (local_id < MAX_EXPERTS) {
        values[local_id] = -INFINITY;
    }

    device const float* input = X + (p.x_offset >> 2);
    const uint k_vec4 = p.K >> 2;
    for (uint i = local_id; i < k_vec4; i += TG_SIZE) {
        x_cache4[i] = *(device const float4*)(input + (i << 2));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base_row = sg_idx * 2u;
    const uint row_stride = p.K;
    const uint weight_base = p.a_offset >> 2;
    const uint bias_base = p.bias_offset >> 2;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    if (base_row < p.n_experts) {
        device const float* row0 = W + weight_base + base_row * row_stride;
        device const float* row1 = row0 + row_stride;

        for (uint vi = lane; vi < k_vec4; vi += 32u) {
            const float4 x = x_cache4[vi];
            acc0 += dot(*(device const float4*)(row0 + (vi << 2)), x);
            if (base_row + 1u < p.n_experts) {
                acc1 += dot(*(device const float4*)(row1 + (vi << 2)), x);
            }
        }

        const float sum0 = simd_sum(acc0);
        const float sum1 = simd_sum(acc1);
        if (lane == 0u) {
            values[base_row] = sum0 + bias[bias_base + base_row];
            if (base_row + 1u < p.n_experts) {
                values[base_row + 1u] = sum1 + bias[bias_base + base_row + 1u];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0u) {
        for (uint ki = 0u; ki < p.k; ki++) {
            float best_val = -INFINITY;
            uint best_idx = 0u;
            for (uint i = 0u; i < p.n_experts; i++) {
                const float v = values[i];
                if (v > best_val) {
                    best_val = v;
                    best_idx = i;
                }
            }
            output_data[ki] = best_idx;
            selected_val[ki] = best_val;
            values[best_idx] = -INFINITY;
        }

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
