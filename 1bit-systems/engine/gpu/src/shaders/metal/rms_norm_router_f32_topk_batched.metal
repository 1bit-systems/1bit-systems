#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Gemma MoE router helper: normalize hidden with the router scale row, then
// compute F32 top-k routing from that normalized row in one dispatch.

struct RmsNormRouterF32TopkBatchedPush {
    uint n_experts;
    uint K;
    uint k;
    uint router_offset;
    uint input_offset;
    uint input_stride;
    uint output_stride;
    uint norm_weight_offset;
    uint shared_gate_offset;
    uint shared_input_offset;
    uint shared_input_stride;
    uint has_shared_gate;
    uint moe_norm_weight_offset;
    uint moe_norm_output_offset;
    uint moe_norm_output_stride;
    uint has_moe_norm;
    uint logit_scale_bits;
    float eps;
};

#define TG_SIZE 512
#define SIMD_WIDTH 32
#define N_SIMDGROUPS (TG_SIZE / SIMD_WIDTH)
#define ROWS_PER_TG (N_SIMDGROUPS * 2)
#define MAX_EXPERTS 256
#define MAX_K_USED 16
#define MAX_K_VEC4 1024

kernel void main0(
    device const float* W [[buffer(0)]],
    constant RmsNormRouterF32TopkBatchedPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device const float* norm_weight [[buffer(3)]],
    device uint* output_data [[buffer(4)]],
    device const float* shared_input_buf [[buffer(5)]],
    device const float* W_shared_gate [[buffer(6)]],
    device float* shared_gate_out [[buffer(7)]],
    device const float* moe_norm_weight [[buffer(8)]],
    device float* moe_norm_out [[buffer(9)]],
    uint token_idx [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float4 x_cache4[MAX_K_VEC4];
    threadgroup float partial_sums[N_SIMDGROUPS];
    threadgroup float shared_partials[N_SIMDGROUPS];
    threadgroup float values[MAX_EXPERTS];
    threadgroup float selected_val[MAX_K_USED];
    const float logit_scale = as_type<float>(p.logit_scale_bits);

    if (p.n_experts == 0u || p.n_experts > MAX_EXPERTS ||
        p.k == 0u || p.k > MAX_K_USED ||
        (p.K & 3u) != 0u || (p.K >> 2) > MAX_K_VEC4) {
        return;
    }

    device const float* input = X + (p.input_offset >> 2) + token_idx * p.input_stride;
    device const float* scale = norm_weight + (p.norm_weight_offset >> 2);
    const uint k_vec4 = p.K >> 2;

    float sum_sq = 0.0f;
    for (uint i = local_id; i < k_vec4; i += TG_SIZE) {
        const float4 x = *(device const float4*)(input + (i << 2));
        x_cache4[i] = x;
        sum_sq += dot(x, x);
    }

    const float sg_sum = simd_sum(sum_sq);
    if (lane == 0u) {
        partial_sums[sg_idx] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partial = (lane < N_SIMDGROUPS) ? partial_sums[lane] : 0.0f;
    const float total_sq = simd_sum(partial);
    const float rms_inv = fast::rsqrt(fast::divide(total_sq, float(p.K)) + p.eps);

    device const float* moe_scale = moe_norm_weight + (p.moe_norm_weight_offset >> 2);
    device float* moe_out = moe_norm_out + (p.moe_norm_output_offset >> 2) + token_idx * p.moe_norm_output_stride;
    for (uint i = local_id; i < k_vec4; i += TG_SIZE) {
        const float4 raw_x = x_cache4[i];
        if (p.has_moe_norm != 0u) {
            const float4 moe_w = *(device const float4*)(moe_scale + (i << 2));
            *(device float4*)(moe_out + (i << 2)) = raw_x * rms_inv * moe_w;
        }
        const float4 w = *(device const float4*)(scale + (i << 2));
        x_cache4[i] = raw_x * rms_inv * w;
    }

    if (local_id < MAX_EXPERTS) {
        values[local_id] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint weight_base = p.router_offset >> 2;
    for (uint row_block = 0u; row_block < p.n_experts; row_block += ROWS_PER_TG) {
        const uint base_row = row_block + sg_idx * 2u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;

        if (base_row < p.n_experts) {
            device const float* row0 = W + weight_base + base_row * p.K;
            device const float* row1 = row0 + p.K;

            for (uint vi = lane; vi < k_vec4; vi += SIMD_WIDTH) {
                const float4 x = x_cache4[vi];
                acc0 += dot(*(device const float4*)(row0 + (vi << 2)), x);
                if (base_row + 1u < p.n_experts) {
                    acc1 += dot(*(device const float4*)(row1 + (vi << 2)), x);
                }
            }

            const float2 sums = simd_sum(float2(acc0, acc1));
            const uint store_row = base_row + lane;
            if (lane < 2u && store_row < p.n_experts) {
                values[store_row] = (lane == 0u) ? sums.x : sums.y;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Adapt vLLM's fused top-k/shared-gate materialization to Gemma's scaled
    // router: shared-gate uses the FFN norm row, not the router-scaled row.
    float shared_acc = 0.0f;
    if (p.has_shared_gate != 0u) {
        device const float* shared_input = shared_input_buf + (p.shared_input_offset >> 2) + token_idx * p.shared_input_stride;
        device const float* shared_row = W_shared_gate + (p.shared_gate_offset >> 2);
        for (uint vi = local_id; vi < k_vec4; vi += TG_SIZE) {
            shared_acc += dot(*(device const float4*)(shared_row + (vi << 2)), *(device const float4*)(shared_input + (vi << 2)));
        }
    }
    const float shared_sum = simd_sum(shared_acc);
    if (lane == 0u) {
        shared_partials[sg_idx] = shared_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p.has_shared_gate != 0u && local_id == 0u) {
        float shared_total = 0.0f;
        #pragma unroll
        for (uint i = 0u; i < N_SIMDGROUPS; ++i) {
            shared_total += shared_partials[i];
        }
        shared_gate_out[token_idx] = shared_total;
    }

    if (p.n_experts == 128u && p.k == 8u) {
        if (sg_idx == 0u) {
            float selected_score[8];
            uint selected_mask = 0u;
            #pragma unroll
            for (uint slot = 0u; slot < 8u; ++slot) {
                selected_score[slot] = -INFINITY;
            }

            #pragma unroll
            for (uint slot = 0u; slot < 8u; ++slot) {
                float lane_best = -INFINITY;
                uint lane_best_idx = 0xffffffffu;
                #pragma unroll
                for (uint lane_row = 0u; lane_row < 4u; ++lane_row) {
                    const uint expert = lane + (lane_row << 5);
                    const float score = ((selected_mask & (1u << lane_row)) == 0u) ? values[expert] : -INFINITY;
                    if (score > lane_best) {
                        lane_best = score;
                        lane_best_idx = expert;
                    }
                }
                const float best_val = simd_max(lane_best);
                const uint best_idx = simd_min((lane_best == best_val) ? lane_best_idx : 0xffffffffu);
                selected_score[slot] = best_val;
                if ((best_idx & 31u) == lane) {
                    selected_mask |= 1u << (best_idx >> 5);
                }
                if (lane == 0u) {
                    output_data[token_idx * p.output_stride + slot] = best_idx;
                }
            }

            const bool weight_lane = lane < 8u;
            const float score = weight_lane ? selected_score[lane] * logit_scale : -INFINITY;
            const float max_sel = simd_max(score);
            const float exp_score = weight_lane ? fast::exp(score - max_sel) : 0.0f;
            const float sum = simd_sum(exp_score);
            const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
            if (weight_lane) {
                output_data[token_idx * p.output_stride + 8u + lane] = as_type<uint>(exp_score * inv_sum);
            }
        }
        return;
    }

    if (local_id == 0u) {
        const uint k = min(p.k, uint(MAX_K_USED));
        for (uint slot = 0u; slot < k; slot++) {
            float best_val = -INFINITY;
            uint best_idx = 0u;
            for (uint expert = 0u; expert < p.n_experts; expert++) {
                const float v = values[expert];
                if (v > best_val) {
                    best_val = v;
                    best_idx = expert;
                }
            }
            output_data[token_idx * p.output_stride + slot] = best_idx;
            selected_val[slot] = best_val;
            values[best_idx] = -INFINITY;
        }

        float max_sel = -INFINITY;
        for (uint slot = 0u; slot < k; slot++) {
            selected_val[slot] *= logit_scale;
            max_sel = max(max_sel, selected_val[slot]);
        }

        float sum = 0.0f;
        for (uint slot = 0u; slot < k; slot++) {
            const float e = fast::exp(selected_val[slot] - max_sel);
            selected_val[slot] = e;
            sum += e;
        }

        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (uint slot = 0u; slot < k; slot++) {
            output_data[token_idx * p.output_stride + p.k + slot] = as_type<uint>(selected_val[slot] * inv_sum);
        }
    }
}
