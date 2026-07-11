#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Exact Qwen3.6 residual+RMSNorm+router/top-k for private-repacked Q8 weights.
//
// This is the repacked companion to residual_rms_norm_router_q8_0_topk_k2048:
// it preserves the fused residual/norm/router contract, but uses the same
// coalesced Q8 block layout as ZINC's repacked DMMV kernels.

struct Params {
    uint n;
    float eps;
    float scale;
    uint residual_offset;
    uint n_experts;
    uint K;
    uint k;
    uint a_offset;
};

#define SIMD_WIDTH 32
#define ROUTER_TG_SIZE 1024
#define NORM_SIMDGROUPS (ROUTER_TG_SIZE / SIMD_WIDTH)
#define MAX_HIDDEN 2048

kernel void main0(
    constant Params& p [[buffer(0)]],
    device float* hidden [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* norm_out [[buffer(3)]],
    device const float* norm_weight [[buffer(4)]],
    device const uchar* W [[buffer(5)]],
    device uint* output_data [[buffer(6)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    if (p.n != MAX_HIDDEN || p.K != MAX_HIDDEN || p.n_experts != 256u || p.k != 8u) {
        return;
    }

    threadgroup float norm_cache[MAX_HIDDEN];
    threadgroup float partial_sums[NORM_SIMDGROUPS];
    threadgroup float values[256];

    float sum_sq = 0.0f;
    for (uint i = local_id; i < MAX_HIDDEN; i += ROUTER_TG_SIZE) {
        const float h = fma(p.scale, residual[p.residual_offset + i], hidden[i]);
        hidden[i] = h;
        norm_cache[i] = h;
        sum_sq = fma(h, h, sum_sq);
    }

    const float sg_sum = simd_sum(sum_sq);
    if (lane == 0u) {
        partial_sums[sg_idx] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float v = (lane < NORM_SIMDGROUPS) ? partial_sums[lane] : 0.0f;
    const float total_sq = simd_sum(v);
    const float rms_inv = fast::rsqrt(fast::divide(total_sq, float(p.n)) + p.eps);

    for (uint i = local_id; i < MAX_HIDDEN; i += ROUTER_TG_SIZE) {
        const float nval = norm_weight[i] * (norm_cache[i] * rms_inv);
        norm_cache[i] = nval;
        norm_out[i] = nval;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr ulong group_bytes = 1088ul;
    constexpr ulong row_bytes = 2176ul;
    const uint row_pairs = 128u;

    for (uint pair = sg_idx; pair < row_pairs; pair += simdgroups_per_tg) {
        const uint base_row = pair << 1;
        device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
        device const uchar* row1 = row0 + row_bytes;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        #pragma unroll
        for (uint gi = 0u; gi < 2u; ++gi) {
            device const uchar* g0 = row0 + ulong(gi) * group_bytes;
            device const uchar* g1 = row1 + ulong(gi) * group_bytes;
            const float s0 = float(as_type<half>(*(device const ushort*)(g0 + lane * 2u)));
            const float s1 = float(as_type<half>(*(device const ushort*)(g1 + lane * 2u)));
            const uint x_base = (gi * 32u + lane) << 5;

            #pragma unroll
            for (uint vi = 0u; vi < 8u; ++vi) {
                const uint qo = 64u + vi * 128u + lane * 4u;
                const char4 q0 = as_type<char4>(*(device const int*)(g0 + qo));
                const char4 q1 = as_type<char4>(*(device const int*)(g1 + qo));
                const uint xi = x_base + (vi << 2);
                const float4 x = float4(norm_cache[xi], norm_cache[xi + 1u], norm_cache[xi + 2u], norm_cache[xi + 3u]);
                acc0 = fma(s0, dot(float4(q0), x), acc0);
                acc1 = fma(s1, dot(float4(q1), x), acc1);
            }
        }

        // Pack the two final-reduction `simd_sum` calls into one
        // `simd_sum(float2)` + lane-parallel 2-row writeback. Apple9's
        // vector `simd_sum` lowers to a single log2(32)=5-level butterfly
        // that transfers 64-bit packed lanes per `shuffle_xor` instead of
        // two independent 32-bit trees, cutting cross-lane shuffle traffic
        // ~2× on the per-simdgroup tail of the hot fused
        // residual+RMSNorm+router+top-k kernel (1436 calls/req across 30
        // MoE-routed SSM layers × 36 steps; 128 row_pairs/call distributed
        // across `simdgroups_per_tg` SGs ⇒ ~184K simdgroup-tail reductions
        // per request). The downstream lane-uniform writeback into
        // threadgroup `values[]` consumes both sums as simdgroup-broadcast
        // scalars (sums.x / sums.y), so picking float2 components by lane
        // is bit-equivalent to the prior two scalar `simd_sum` calls. Same
        // proven pattern as cycle ~86 (`dmmv_q5k_moe_k512.metal`) and the
        // dual-row Q8 family across cycles 75/82/83. Lane 0 / lane 1 write
        // to adjacent threadgroup slots `values[base_row]` /
        // `values[base_row + 1u]`; each pair iteration owns disjoint
        // base_row positions (pair stride = simdgroups_per_tg, pair << 1)
        // so no cross-SG TG-mem races.
        const float2 sums = simd_sum(float2(acc0, acc1));
        if (lane < 2u) {
            const float local_sum = (lane == 0u) ? sums.x : sums.y;
            values[base_row + lane] = local_sum;
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
