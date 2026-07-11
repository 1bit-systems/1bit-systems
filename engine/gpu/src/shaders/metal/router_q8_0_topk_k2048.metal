#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Exact Qwen3.6 Q8_0 router for hidden_dim=2048, n_experts=256, topk=8.
//
// Adapts the reference implementation `ggml-metal.metal::kernel_mul_mv_q8_0_f32_impl` by
// baking the hot K=2048 row walk into the router/top-k kernel. The top-k
// finalization keeps the same vLLM-style compact ids + normalized weights
// contract as router_q8_0_topk.metal.

struct RouterQ8TopkPush {
    uint n_experts;
    uint K;
    uint k;
    uint a_offset;
    uint x_offset;
};

kernel void main0(
    constant RouterQ8TopkPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device uint* output_data [[buffer(3)]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    if (p.K != 2048u || p.n_experts != 256u || p.k != 8u) return;

    threadgroup float values[256];

    device const float* input = X + (p.x_offset >> 2);
    const ulong row_bytes = 2176ull; // 64 Q8_0 blocks * 34 bytes
    const uint row_pairs = 128u;

    for (uint pair = sg_idx; pair < row_pairs; pair += simdgroups_per_tg) {
        const uint base_row = pair << 1;
        device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
        device const uchar* row1 = row0 + row_bytes;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        #pragma unroll
        for (uint pass = 0u; pass < 2u; ++pass) {
            const uint bi = lane + (pass << 5);
            device const uchar* blk0 = row0 + ulong(bi) * 34ull;
            device const uchar* blk1 = row1 + ulong(bi) * 34ull;
            const float s0 = float(as_type<half>(*(device const ushort*)(blk0)));
            const float s1 = float(as_type<half>(*(device const ushort*)(blk1)));
            device const packed_char4* q0 = (device const packed_char4*)(blk0 + 2u);
            device const packed_char4* q1 = (device const packed_char4*)(blk1 + 2u);
            const uint x_base = bi << 5;

            #pragma unroll
            for (uint vi = 0u; vi < 8u; ++vi) {
                const float4 x = *(device const float4*)(input + x_base + (vi << 2));
                acc0 = fma(s0, dot(float4(char4(q0[vi])), x), acc0);
                acc1 = fma(s1, dot(float4(char4(q1[vi])), x), acc1);
            }
        }

        const float sum0 = simd_sum(acc0);
        const float sum1 = simd_sum(acc1);
        if (lane == 0u) {
            values[base_row] = sum0;
            values[base_row + 1u] = sum1;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
            for (uint lane_row = 0u; lane_row < 8u; ++lane_row) {
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
                output_data[slot] = best_idx;
            }
        }

        const bool weight_lane = lane < 8u;
        const float score = weight_lane ? selected_score[lane] : -INFINITY;
        const float max_sel = simd_max(score);
        const float exp_score = weight_lane ? fast::exp(score - max_sel) : 0.0f;
        const float sum = simd_sum(exp_score);
        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        if (weight_lane) {
            output_data[8u + lane] = as_type<uint>(exp_score * inv_sum);
        }
    }
}
