#include <metal_stdlib>
using namespace metal;

// Q6_K grouped MoE DMMV for route-packed prefill.
//
// This follows the vLLM-style expert-major route layout produced by
// moe_route_pack: grid.y is the real expert id, grid.z selects up to eight
// packed route ids for that expert, and each simdgroup computes one output
// row while reusing the same dequantized Q6_K weights across those routes.

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

inline float fp16_to_fp32(uint h) {
    return float(as_type<half>(ushort(h)));
}

inline float s8_to_f32(uint x) {
    return float((x < 128u) ? int(x) : (int(x) - 256));
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
    device const uchar* row_ptr = W + expert_base + ulong(row) * ulong(blocks_per_row) * 210ull;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);

    for (uint b = 0u; b < blocks_per_row; b++) {
        device const uchar* block = row_ptr + b * 210u;
        const float d = fp16_to_fp32(uint(block[208u]) | (uint(block[209u]) << 8u));

        for (uint g = 0u; g < 2u; g++) {
            const uint qs_lo_base = g * 64u;
            const uint qs_hi_base = 128u + g * 32u;
            const uint scale_base = 192u + g * 8u;
            const uint scale_group = tid / 16u;

            const uint ql0 = uint(block[qs_lo_base + tid]);
            const uint ql1 = uint(block[qs_lo_base + 32u + tid]);
            const uint qh = uint(block[qs_hi_base + tid]);
            const float d_sc0 = d * s8_to_f32(uint(block[scale_base + scale_group]));
            const float d_sc1 = d * s8_to_f32(uint(block[scale_base + 2u + scale_group]));
            const float d_sc2 = d * s8_to_f32(uint(block[scale_base + 4u + scale_group]));
            const float d_sc3 = d * s8_to_f32(uint(block[scale_base + 6u + scale_group]));

            const float q0 = float((ql0 & 0x0Fu) | ((qh & 0x03u) << 4u)) - 32.0f;
            const float q1 = float((ql1 & 0x0Fu) | (((qh >> 2u) & 0x03u) << 4u)) - 32.0f;
            const float q2 = float((ql0 >> 4u) | (((qh >> 4u) & 0x03u) << 4u)) - 32.0f;
            const float q3 = float((ql1 >> 4u) | (((qh >> 6u) & 0x03u) << 4u)) - 32.0f;

            const uint base_col = b * 256u + g * 128u + tid;

            if (active0) {
                acc0.x += (d_sc0 * q0) * x0[base_col];
                acc0.x += (d_sc1 * q1) * x0[base_col + 32u];
                acc0.x += (d_sc2 * q2) * x0[base_col + 64u];
                acc0.x += (d_sc3 * q3) * x0[base_col + 96u];
            }
            if (active1) {
                acc0.y += (d_sc0 * q0) * x1[base_col];
                acc0.y += (d_sc1 * q1) * x1[base_col + 32u];
                acc0.y += (d_sc2 * q2) * x1[base_col + 64u];
                acc0.y += (d_sc3 * q3) * x1[base_col + 96u];
            }
            if (active2) {
                acc0.z += (d_sc0 * q0) * x2[base_col];
                acc0.z += (d_sc1 * q1) * x2[base_col + 32u];
                acc0.z += (d_sc2 * q2) * x2[base_col + 64u];
                acc0.z += (d_sc3 * q3) * x2[base_col + 96u];
            }
            if (active3) {
                acc0.w += (d_sc0 * q0) * x3[base_col];
                acc0.w += (d_sc1 * q1) * x3[base_col + 32u];
                acc0.w += (d_sc2 * q2) * x3[base_col + 64u];
                acc0.w += (d_sc3 * q3) * x3[base_col + 96u];
            }
            if (active4) {
                acc1.x += (d_sc0 * q0) * x4[base_col];
                acc1.x += (d_sc1 * q1) * x4[base_col + 32u];
                acc1.x += (d_sc2 * q2) * x4[base_col + 64u];
                acc1.x += (d_sc3 * q3) * x4[base_col + 96u];
            }
            if (active5) {
                acc1.y += (d_sc0 * q0) * x5[base_col];
                acc1.y += (d_sc1 * q1) * x5[base_col + 32u];
                acc1.y += (d_sc2 * q2) * x5[base_col + 64u];
                acc1.y += (d_sc3 * q3) * x5[base_col + 96u];
            }
            if (active6) {
                acc1.z += (d_sc0 * q0) * x6[base_col];
                acc1.z += (d_sc1 * q1) * x6[base_col + 32u];
                acc1.z += (d_sc2 * q2) * x6[base_col + 64u];
                acc1.z += (d_sc3 * q3) * x6[base_col + 96u];
            }
            if (active7) {
                acc1.w += (d_sc0 * q0) * x7[base_col];
                acc1.w += (d_sc1 * q1) * x7[base_col + 32u];
                acc1.w += (d_sc2 * q2) * x7[base_col + 64u];
                acc1.w += (d_sc3 * q3) * x7[base_col + 96u];
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
