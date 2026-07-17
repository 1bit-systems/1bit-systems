#include <metal_stdlib>
using namespace metal;

// Q5_1 grouped MoE DMMV for batched Gemma expert-down prefill.
//
// Dispatch: grid (ceil(M / 8), active route blocks, 1),
// threadgroup (64, 1, 1). grid.y is the real expert id. grid.z selects a
// packed route-id block from moe_route_pack.metal's [expert][ids_stride] table.
//
// Each simdgroup owns one output row and computes up to eight routed token
// vectors for that row while reading/dequantizing the Q5_1 weights once.

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

    device const uint* expert_ids = packed_ids + expert_id * p.ids_stride;
    const uint x_div = max(p.x_route_divisor, 1u);
    device const float* x_base = X + (p.x_offset / 4u);

    const uint nb = p.K / 32u;
    const uint bpb = 24u;
    const ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    device const uchar* src = W + expert_base + ulong(row) * ulong(nb) * ulong(bpb);

    if (packed_base + NUM_COLS <= count) {
        const uint route0 = expert_ids[packed_base + 0u];
        const uint route1 = expert_ids[packed_base + 1u];
        const uint route2 = expert_ids[packed_base + 2u];
        const uint route3 = expert_ids[packed_base + 3u];
        const uint route4 = expert_ids[packed_base + 4u];
        const uint route5 = expert_ids[packed_base + 5u];
        const uint route6 = expert_ids[packed_base + 6u];
        const uint route7 = expert_ids[packed_base + 7u];

        device const float* x0 = x_base + (route0 / x_div) * p.K;
        device const float* x1 = x_base + (route1 / x_div) * p.K;
        device const float* x2 = x_base + (route2 / x_div) * p.K;
        device const float* x3 = x_base + (route3 / x_div) * p.K;
        device const float* x4 = x_base + (route4 / x_div) * p.K;
        device const float* x5 = x_base + (route5 / x_div) * p.K;
        device const float* x6 = x_base + (route6 / x_div) * p.K;
        device const float* x7 = x_base + (route7 / x_div) * p.K;

        float4 acc0_full = float4(0.0f);
        float4 acc1_full = float4(0.0f);

        for (uint b = tid; b < nb; b += 32u) {
            device const uchar* block = src + b * bpb;

            const float d = float(*((device const half*)block));
            const float m = float(*((device const half*)(block + 2)));
            const uint qh = uint(block[4]) | (uint(block[5]) << 8)
                          | (uint(block[6]) << 16) | (uint(block[7]) << 24);
            device const uchar* qs = block + 8;
            const uint base = b * 32u;

            float4 sum_qx0 = float4(0.0f);
            float4 sum_x0 = float4(0.0f);
            float4 sum_qx1 = float4(0.0f);
            float4 sum_x1 = float4(0.0f);
            for (uint j = 0u; j < 16u; j++) {
                const uchar q_byte = qs[j];
                const uint lo = q_byte & 0x0F;
                const uint hi = q_byte >> 4;
                const uint q0 = lo | (((qh >> j) & 1u) << 4);
                const uint q1 = hi | (((qh >> (j + 16u)) & 1u) << 4);

                const float4 x_lo0 = float4(
                    x0[base + j],
                    x1[base + j],
                    x2[base + j],
                    x3[base + j]
                );
                const float4 x_hi0 = float4(
                    x0[base + 16u + j],
                    x1[base + 16u + j],
                    x2[base + 16u + j],
                    x3[base + 16u + j]
                );
                const float4 x_lo1 = float4(
                    x4[base + j],
                    x5[base + j],
                    x6[base + j],
                    x7[base + j]
                );
                const float4 x_hi1 = float4(
                    x4[base + 16u + j],
                    x5[base + 16u + j],
                    x6[base + 16u + j],
                    x7[base + 16u + j]
                );

                sum_qx0 += float(q0) * x_lo0 + float(q1) * x_hi0;
                sum_x0 += x_lo0 + x_hi0;
                sum_qx1 += float(q0) * x_lo1 + float(q1) * x_hi1;
                sum_x1 += x_lo1 + x_hi1;
            }

            acc0_full += d * sum_qx0 + m * sum_x0;
            acc1_full += d * sum_qx1 + m * sum_x1;
        }

        const float out0 = simd_sum(acc0_full.x);
        const float out1 = simd_sum(acc0_full.y);
        const float out2 = simd_sum(acc0_full.z);
        const float out3 = simd_sum(acc0_full.w);
        const float out4 = simd_sum(acc1_full.x);
        const float out5 = simd_sum(acc1_full.y);
        const float out6 = simd_sum(acc1_full.z);
        const float out7 = simd_sum(acc1_full.w);

        device float* y_base = Y + (p.y_offset / 4u);
        if (tid == 0u) {
            y_base[route0 * p.M + row] = out0;
            y_base[route1 * p.M + row] = out1;
            y_base[route2 * p.M + row] = out2;
            y_base[route3 * p.M + row] = out3;
            y_base[route4 * p.M + row] = out4;
            y_base[route5 * p.M + row] = out5;
            y_base[route6 * p.M + row] = out6;
            y_base[route7 * p.M + row] = out7;
        }
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

    const uint route0 = active0 ? expert_ids[packed_base + 0u] : 0u;
    const uint route1 = active1 ? expert_ids[packed_base + 1u] : 0u;
    const uint route2 = active2 ? expert_ids[packed_base + 2u] : 0u;
    const uint route3 = active3 ? expert_ids[packed_base + 3u] : 0u;
    const uint route4 = active4 ? expert_ids[packed_base + 4u] : 0u;
    const uint route5 = active5 ? expert_ids[packed_base + 5u] : 0u;
    const uint route6 = active6 ? expert_ids[packed_base + 6u] : 0u;
    const uint route7 = active7 ? expert_ids[packed_base + 7u] : 0u;

    device const float* x0 = x_base + (route0 / x_div) * p.K;
    device const float* x1 = x_base + (route1 / x_div) * p.K;
    device const float* x2 = x_base + (route2 / x_div) * p.K;
    device const float* x3 = x_base + (route3 / x_div) * p.K;
    device const float* x4 = x_base + (route4 / x_div) * p.K;
    device const float* x5 = x_base + (route5 / x_div) * p.K;
    device const float* x6 = x_base + (route6 / x_div) * p.K;
    device const float* x7 = x_base + (route7 / x_div) * p.K;

    float4 acc0 = float4(0.0f);
    const bool single_col_tail = packed_base + 1u >= count;
    if (single_col_tail) {
        float acc = 0.0f;
        for (uint b = tid; b < nb; b += 32u) {
            device const uchar* block = src + b * bpb;

            const float d = float(*((device const half*)block));
            const float m = float(*((device const half*)(block + 2)));
            const uint qh = uint(block[4]) | (uint(block[5]) << 8)
                          | (uint(block[6]) << 16) | (uint(block[7]) << 24);
            device const uchar* qs = block + 8;
            const uint base = b * 32u;

            float sum_qx = 0.0f;
            float sum_x = 0.0f;
            for (uint j = 0u; j < 16u; j++) {
                const uchar q_byte = qs[j];
                const uint lo = q_byte & 0x0F;
                const uint hi = q_byte >> 4;
                const uint q0 = lo | (((qh >> j) & 1u) << 4);
                const uint q1 = hi | (((qh >> (j + 16u)) & 1u) << 4);

                const float x_lo = x0[base + j];
                const float x_hi = x0[base + 16u + j];
                sum_qx += float(q0) * x_lo + float(q1) * x_hi;
                sum_x += x_lo + x_hi;
            }

            acc += d * sum_qx + m * sum_x;
        }

        const float out = simd_sum(acc);
        device float* y_base = Y + (p.y_offset / 4u);
        if (tid == 0u) {
            y_base[route0 * p.M + row] = out;
        }
        return;
    }

    const bool two_col_tail = packed_base + 2u >= count;
    if (two_col_tail) {
        float2 acc = float2(0.0f);
        for (uint b = tid; b < nb; b += 32u) {
            device const uchar* block = src + b * bpb;

            const float d = float(*((device const half*)block));
            const float m = float(*((device const half*)(block + 2)));
            const uint qh = uint(block[4]) | (uint(block[5]) << 8)
                          | (uint(block[6]) << 16) | (uint(block[7]) << 24);
            device const uchar* qs = block + 8;
            const uint base = b * 32u;

            float2 sum_qx = float2(0.0f);
            float2 sum_x = float2(0.0f);
            for (uint j = 0u; j < 16u; j++) {
                const uchar q_byte = qs[j];
                const uint lo = q_byte & 0x0F;
                const uint hi = q_byte >> 4;
                const uint q0 = lo | (((qh >> j) & 1u) << 4);
                const uint q1 = hi | (((qh >> (j + 16u)) & 1u) << 4);

                const float2 x_lo = float2(x0[base + j], x1[base + j]);
                const float2 x_hi = float2(x0[base + 16u + j], x1[base + 16u + j]);
                sum_qx += float(q0) * x_lo + float(q1) * x_hi;
                sum_x += x_lo + x_hi;
            }

            acc += d * sum_qx + m * sum_x;
        }

        const float out0 = simd_sum(acc.x);
        const float out1 = simd_sum(acc.y);

        device float* y_base = Y + (p.y_offset / 4u);
        if (tid == 0u) {
            y_base[route0 * p.M + row] = out0;
            y_base[route1 * p.M + row] = out1;
        }
        return;
    }

    const bool three_col_tail = packed_base + 3u >= count;
    if (three_col_tail) {
        float3 acc = float3(0.0f);
        for (uint b = tid; b < nb; b += 32u) {
            device const uchar* block = src + b * bpb;

            const float d = float(*((device const half*)block));
            const float m = float(*((device const half*)(block + 2)));
            const uint qh = uint(block[4]) | (uint(block[5]) << 8)
                          | (uint(block[6]) << 16) | (uint(block[7]) << 24);
            device const uchar* qs = block + 8;
            const uint base = b * 32u;

            float3 sum_qx = float3(0.0f);
            float3 sum_x = float3(0.0f);
            for (uint j = 0u; j < 16u; j++) {
                const uchar q_byte = qs[j];
                const uint lo = q_byte & 0x0F;
                const uint hi = q_byte >> 4;
                const uint q0 = lo | (((qh >> j) & 1u) << 4);
                const uint q1 = hi | (((qh >> (j + 16u)) & 1u) << 4);

                const float3 x_lo = float3(x0[base + j], x1[base + j], x2[base + j]);
                const float3 x_hi = float3(x0[base + 16u + j], x1[base + 16u + j], x2[base + 16u + j]);
                sum_qx += float(q0) * x_lo + float(q1) * x_hi;
                sum_x += x_lo + x_hi;
            }

            acc += d * sum_qx + m * sum_x;
        }

        const float out0 = simd_sum(acc.x);
        const float out1 = simd_sum(acc.y);
        const float out2 = simd_sum(acc.z);

        device float* y_base = Y + (p.y_offset / 4u);
        if (tid == 0u) {
            y_base[route0 * p.M + row] = out0;
            y_base[route1 * p.M + row] = out1;
            y_base[route2 * p.M + row] = out2;
        }
        return;
    }

    const bool four_col_tail = packed_base + 4u >= count;
    if (four_col_tail) {
        for (uint b = tid; b < nb; b += 32u) {
            device const uchar* block = src + b * bpb;

            const float d = float(*((device const half*)block));
            const float m = float(*((device const half*)(block + 2)));
            const uint qh = uint(block[4]) | (uint(block[5]) << 8)
                          | (uint(block[6]) << 16) | (uint(block[7]) << 24);
            device const uchar* qs = block + 8;
            const uint base = b * 32u;

            float4 sum_qx0 = float4(0.0f);
            float4 sum_x0 = float4(0.0f);
            for (uint j = 0u; j < 16u; j++) {
                const uchar q_byte = qs[j];
                const uint lo = q_byte & 0x0F;
                const uint hi = q_byte >> 4;
                const uint q0 = lo | (((qh >> j) & 1u) << 4);
                const uint q1 = hi | (((qh >> (j + 16u)) & 1u) << 4);

                const float4 x_lo0 = float4(
                    active0 ? x0[base + j] : 0.0f,
                    active1 ? x1[base + j] : 0.0f,
                    active2 ? x2[base + j] : 0.0f,
                    active3 ? x3[base + j] : 0.0f
                );
                const float4 x_hi0 = float4(
                    active0 ? x0[base + 16u + j] : 0.0f,
                    active1 ? x1[base + 16u + j] : 0.0f,
                    active2 ? x2[base + 16u + j] : 0.0f,
                    active3 ? x3[base + 16u + j] : 0.0f
                );

                sum_qx0 += float(q0) * x_lo0 + float(q1) * x_hi0;
                sum_x0 += x_lo0 + x_hi0;
            }

            acc0 += d * sum_qx0 + m * sum_x0;
        }

        const float out0 = simd_sum(acc0.x);
        const float out1 = simd_sum(acc0.y);
        const float out2 = simd_sum(acc0.z);
        const float out3 = simd_sum(acc0.w);

        device float* y_base = Y + (p.y_offset / 4u);
        if (tid == 0u) {
            if (active0) y_base[route0 * p.M + row] = out0;
            if (active1) y_base[route1 * p.M + row] = out1;
            if (active2) y_base[route2 * p.M + row] = out2;
            if (active3) y_base[route3 * p.M + row] = out3;
        }
        return;
    }

    const bool five_col_tail = packed_base + 5u >= count;
    if (five_col_tail) {
        float acc4 = 0.0f;
        for (uint b = tid; b < nb; b += 32u) {
            device const uchar* block = src + b * bpb;

            const float d = float(*((device const half*)block));
            const float m = float(*((device const half*)(block + 2)));
            const uint qh = uint(block[4]) | (uint(block[5]) << 8)
                          | (uint(block[6]) << 16) | (uint(block[7]) << 24);
            device const uchar* qs = block + 8;
            const uint base = b * 32u;

            float4 sum_qx0 = float4(0.0f);
            float4 sum_x0 = float4(0.0f);
            float sum_qx4 = 0.0f;
            float sum_x4 = 0.0f;
            for (uint j = 0u; j < 16u; j++) {
                const uchar q_byte = qs[j];
                const uint lo = q_byte & 0x0F;
                const uint hi = q_byte >> 4;
                const uint q0 = lo | (((qh >> j) & 1u) << 4);
                const uint q1 = hi | (((qh >> (j + 16u)) & 1u) << 4);

                const float4 x_lo0 = float4(
                    x0[base + j],
                    x1[base + j],
                    x2[base + j],
                    x3[base + j]
                );
                const float4 x_hi0 = float4(
                    x0[base + 16u + j],
                    x1[base + 16u + j],
                    x2[base + 16u + j],
                    x3[base + 16u + j]
                );
                const float x_lo4 = x4[base + j];
                const float x_hi4 = x4[base + 16u + j];

                sum_qx0 += float(q0) * x_lo0 + float(q1) * x_hi0;
                sum_x0 += x_lo0 + x_hi0;
                sum_qx4 += float(q0) * x_lo4 + float(q1) * x_hi4;
                sum_x4 += x_lo4 + x_hi4;
            }

            acc0 += d * sum_qx0 + m * sum_x0;
            acc4 += d * sum_qx4 + m * sum_x4;
        }

        const float out0 = simd_sum(acc0.x);
        const float out1 = simd_sum(acc0.y);
        const float out2 = simd_sum(acc0.z);
        const float out3 = simd_sum(acc0.w);
        const float out4 = simd_sum(acc4);

        device float* y_base = Y + (p.y_offset / 4u);
        if (tid == 0u) {
            y_base[route0 * p.M + row] = out0;
            y_base[route1 * p.M + row] = out1;
            y_base[route2 * p.M + row] = out2;
            y_base[route3 * p.M + row] = out3;
            y_base[route4 * p.M + row] = out4;
        }
        return;
    }

    float4 acc1 = float4(0.0f);

    for (uint b = tid; b < nb; b += 32u) {
        device const uchar* block = src + b * bpb;

        const float d = float(*((device const half*)block));
        const float m = float(*((device const half*)(block + 2)));
        const uint qh = uint(block[4]) | (uint(block[5]) << 8)
                      | (uint(block[6]) << 16) | (uint(block[7]) << 24);
        device const uchar* qs = block + 8;
        const uint base = b * 32u;

        float4 sum_qx0 = float4(0.0f);
        float4 sum_x0 = float4(0.0f);
        float4 sum_qx1 = float4(0.0f);
        float4 sum_x1 = float4(0.0f);
        for (uint j = 0u; j < 16u; j++) {
            const uchar q_byte = qs[j];
            const uint lo = q_byte & 0x0F;
            const uint hi = q_byte >> 4;
            const uint q0 = lo | (((qh >> j) & 1u) << 4);
            const uint q1 = hi | (((qh >> (j + 16u)) & 1u) << 4);

            const float4 x_lo0 = float4(
                active0 ? x0[base + j] : 0.0f,
                active1 ? x1[base + j] : 0.0f,
                active2 ? x2[base + j] : 0.0f,
                active3 ? x3[base + j] : 0.0f
            );
            const float4 x_hi0 = float4(
                active0 ? x0[base + 16u + j] : 0.0f,
                active1 ? x1[base + 16u + j] : 0.0f,
                active2 ? x2[base + 16u + j] : 0.0f,
                active3 ? x3[base + 16u + j] : 0.0f
            );
            const float4 x_lo1 = float4(
                active4 ? x4[base + j] : 0.0f,
                active5 ? x5[base + j] : 0.0f,
                active6 ? x6[base + j] : 0.0f,
                active7 ? x7[base + j] : 0.0f
            );
            const float4 x_hi1 = float4(
                active4 ? x4[base + 16u + j] : 0.0f,
                active5 ? x5[base + 16u + j] : 0.0f,
                active6 ? x6[base + 16u + j] : 0.0f,
                active7 ? x7[base + 16u + j] : 0.0f
            );

            sum_qx0 += float(q0) * x_lo0 + float(q1) * x_hi0;
            sum_x0 += x_lo0 + x_hi0;
            sum_qx1 += float(q0) * x_lo1 + float(q1) * x_hi1;
            sum_x1 += x_lo1 + x_hi1;
        }

        acc0 += d * sum_qx0 + m * sum_x0;
        acc1 += d * sum_qx1 + m * sum_x1;
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
