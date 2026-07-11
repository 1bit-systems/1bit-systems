#include <metal_stdlib>
using namespace metal;

// Q5_K grouped MoE DMMV for route-packed prefill.
//
// This is the vLLM-style expert-major route layout used by moe_route_pack:
// grid.y is the real expert id, grid.z selects up to eight packed route ids for
// that expert, and each simdgroup computes one output row for those routes.

struct MoeColsDmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint expert_stride;
    uint x_offset;
    uint y_offset;
    uint ids_stride;
    uint x_route_divisor;
    uint use_active_blocks;
};

inline float2 get_scale_min_k5(uint j, device const uchar* scales) {
    if (j < 4u) {
        return float2(float(scales[j] & 63u), float(scales[4u + j] & 63u));
    }
    return float2(
        float((scales[4u + j] & 0x0Fu) | ((scales[j - 4u] >> 6u) << 4u)),
        float((scales[4u + j] >> 4u) | ((scales[j] >> 6u) << 4u))
    );
}

#define NUM_COLS 8u
#define ROWS_PER_TG 8u

kernel void main0(
    device const uchar* W                     [[buffer(0)]],
    constant MoeColsDmmvPush& p               [[buffer(1)]],
    device const float* X                     [[buffer(2)]],
    device float* Y                           [[buffer(3)]],
    device const uint* counts                 [[buffer(4)]],
    device const uint* packed_ids             [[buffer(5)]],
    device const uint* active_blocks          [[buffer(6)]],
    device const uint* active_block_count     [[buffer(7)]],
    uint3 tg_pos                              [[threadgroup_position_in_grid]],
    uint tid                                  [[thread_index_in_simdgroup]],
    uint sgid                                 [[simdgroup_index_in_threadgroup]]
) {
    if (p.use_active_blocks != 0u && tg_pos.y >= active_block_count[0]) {
        return;
    }

    const uint block_entry = (p.use_active_blocks != 0u) ? active_blocks[tg_pos.y] : 0u;
    const uint expert_id = (p.use_active_blocks != 0u) ? (block_entry & 0xFFFFu) : tg_pos.y;
    const uint row = tg_pos.x * ROWS_PER_TG + sgid;
    if (row >= p.M) {
        return;
    }

    const uint packed_base = (p.use_active_blocks != 0u) ? ((block_entry >> 16u) * NUM_COLS) : (tg_pos.z * NUM_COLS);
    const uint count = counts[expert_id];
    if (packed_base >= count) {
        return;
    }

    const bool active0 = packed_base + 0u < count;
    const bool active1 = packed_base + 1u < count;
    const bool active2 = packed_base + 2u < count;
    const bool active3 = packed_base + 3u < count;
    const bool active4 = packed_base + 4u < count;
    const bool active5 = packed_base + 5u < count;
    const bool active6 = packed_base + 6u < count;
    const bool active7 = packed_base + 7u < count;

    device const uint* expert_ids = packed_ids + expert_id * p.ids_stride;
    const uint route0 = active0 ? expert_ids[packed_base + 0u] : 0u;
    const uint route1 = active1 ? expert_ids[packed_base + 1u] : 0u;
    const uint route2 = active2 ? expert_ids[packed_base + 2u] : 0u;
    const uint route3 = active3 ? expert_ids[packed_base + 3u] : 0u;
    const uint route4 = active4 ? expert_ids[packed_base + 4u] : 0u;
    const uint route5 = active5 ? expert_ids[packed_base + 5u] : 0u;
    const uint route6 = active6 ? expert_ids[packed_base + 6u] : 0u;
    const uint route7 = active7 ? expert_ids[packed_base + 7u] : 0u;

    const uint x_div = max(p.x_route_divisor, 1u);
    device const float* x_base = X + (p.x_offset / 4u);
    device const float* x0 = x_base + (route0 / x_div) * p.K;
    device const float* x1 = x_base + (route1 / x_div) * p.K;
    device const float* x2 = x_base + (route2 / x_div) * p.K;
    device const float* x3 = x_base + (route3 / x_div) * p.K;
    device const float* x4 = x_base + (route4 / x_div) * p.K;
    device const float* x5 = x_base + (route5 / x_div) * p.K;
    device const float* x6 = x_base + (route6 / x_div) * p.K;
    device const float* x7 = x_base + (route7 / x_div) * p.K;

    const uint blocks_per_row = p.K / 256u;
    const ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    device const uchar* row_ptr = W + expert_base + ulong(row) * ulong(blocks_per_row) * 176ull;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);

    for (uint b = 0u; b < blocks_per_row; b++) {
        device const uchar* block = row_ptr + b * 176u;

        const float d = float(as_type<half>(*(device const ushort*)(block)));
        const float dmin = float(as_type<half>(*(device const ushort*)(block + 2)));
        device const uchar* scales = block + 4u;
        device const uchar* high_bits = block + 16u;
        device const uchar* quants = block + 48u;

        const uint qh_val = uint(high_bits[tid]);
        const uint col_base = b * 256u;

        for (uint g = 0u; g < 4u; g++) {
            const uint sb_lo = g * 2u;
            const uint sb_hi = g * 2u + 1u;
            const float2 sm_lo = get_scale_min_k5(sb_lo, scales);
            const float2 sm_hi = get_scale_min_k5(sb_hi, scales);
            const float factor_lo = d * sm_lo.x;
            const float bias_lo = dmin * sm_lo.y;
            const float factor_hi = d * sm_hi.x;
            const float bias_hi = dmin * sm_hi.y;

            const uint q_byte = uint(quants[g * 32u + tid]);
            const float v_lo = factor_lo * float((q_byte & 0x0Fu) | (((qh_val >> sb_lo) & 1u) << 4u)) - bias_lo;
            const float v_hi = factor_hi * float((q_byte >> 4u) | (((qh_val >> sb_hi) & 1u) << 4u)) - bias_hi;

            const uint col_lo = col_base + g * 64u + tid;
            const uint col_hi = col_lo + 32u;

            if (active0) {
                acc0.x += v_lo * x0[col_lo] + v_hi * x0[col_hi];
            }
            if (active1) {
                acc0.y += v_lo * x1[col_lo] + v_hi * x1[col_hi];
            }
            if (active2) {
                acc0.z += v_lo * x2[col_lo] + v_hi * x2[col_hi];
            }
            if (active3) {
                acc0.w += v_lo * x3[col_lo] + v_hi * x3[col_hi];
            }
            if (active4) {
                acc1.x += v_lo * x4[col_lo] + v_hi * x4[col_hi];
            }
            if (active5) {
                acc1.y += v_lo * x5[col_lo] + v_hi * x5[col_hi];
            }
            if (active6) {
                acc1.z += v_lo * x6[col_lo] + v_hi * x6[col_hi];
            }
            if (active7) {
                acc1.w += v_lo * x7[col_lo] + v_hi * x7[col_hi];
            }
        }
    }

    const float out0 = simd_sum(acc0.x);
    const float out1 = simd_sum(acc0.y);
    const float out2 = simd_sum(acc0.z);
    const float out3 = simd_sum(acc0.w);
    const float out4 = simd_sum(acc1.x);
    const float out5 = simd_sum(acc1.y);
    const float out6 = simd_sum(acc1.z);
    const float out7 = simd_sum(acc1.w);

    device float* y_base = Y + (p.y_offset / 4u);
    if (tid == 0u) {
        if (active0) y_base[route0 * p.M + row] = out0;
        if (active1) y_base[route1 * p.M + row] = out1;
        if (active2) y_base[route2 * p.M + row] = out2;
        if (active3) y_base[route3 * p.M + row] = out3;
        if (active4) y_base[route4 * p.M + row] = out4;
        if (active5) y_base[route5 * p.M + row] = out5;
        if (active6) y_base[route6 * p.M + row] = out6;
        if (active7) y_base[route7 * p.M + row] = out7;
    }
}
