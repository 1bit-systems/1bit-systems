#include <metal_stdlib>
using namespace metal;

// Q4_K grouped MoE DMMV for batched Gemma fused gate/up prefill.
//
// Dispatch: grid (ceil(M / 8), active route blocks, 1),
// threadgroup (64, 1, 1). grid.y is the real expert id. grid.z selects a
// packed route-id block from moe_route_pack.metal's [expert][ids_stride] table.
//
// Each simdgroup owns one output row and computes up to eight routed token
// vectors for that row while reading/dequantizing the Q4_K weights once.

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

inline float2 get_scale_min_k4(uint j, device const uchar* sc) {
    if (j < 4) {
        return float2(float(sc[j] & 63), float(sc[j + 4] & 63));
    }
    return float2(
        float((sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4)),
        float(((sc[j + 4] >> 4) & 0x0F) | ((sc[j] >> 6) << 4))
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

    const uint nb = p.K / 256u;
    const uint bpb = 144u;
    const ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    device const uchar* src = W + expert_base + ulong(row) * ulong(nb) * ulong(bpb);

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);

    for (uint b = 0u; b < nb; b++) {
        device const uchar* block = src + b * bpb;

        const float d = float(as_type<half>(*(device const ushort*)(block)));
        const float dmin = float(as_type<half>(*(device const ushort*)(block + 2)));
        device const uchar* scales = block + 4;
        device const uchar* quants = block + 16;

        const uint byte_off = tid * 4u;
        const uint j = byte_off / 32u;
        const uint local_off = byte_off % 32u;

        const uchar4 qbytes = *(device const uchar4*)(quants + byte_off);
        const float2 sm_lo = get_scale_min_k4(j * 2u, scales);
        const float2 sm_hi = get_scale_min_k4(j * 2u + 1u, scales);

        const float d_sc_lo = d * sm_lo.x;
        const float d_m_lo = dmin * sm_lo.y;
        const float d_sc_hi = d * sm_hi.x;
        const float d_m_hi = dmin * sm_hi.y;

        const uint col_lo = b * 256u + j * 64u + local_off;
        const uint col_hi = col_lo + 32u;

        const uchar4 q_lo = uchar4(
            qbytes.x & 0x0F,
            qbytes.y & 0x0F,
            qbytes.z & 0x0F,
            qbytes.w & 0x0F
        );
        const uchar4 q_hi = uchar4(
            qbytes.x >> 4,
            qbytes.y >> 4,
            qbytes.z >> 4,
            qbytes.w >> 4
        );

        const float4 lo_vals = fma(float4(q_lo), float4(d_sc_lo), float4(-d_m_lo));
        const float4 hi_vals = fma(float4(q_hi), float4(d_sc_hi), float4(-d_m_hi));

        if (active0) {
            const float4 x_lo = *(device const float4*)(x0 + col_lo);
            const float4 x_hi = *(device const float4*)(x0 + col_hi);
            acc0.x += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
        }
        if (active1) {
            const float4 x_lo = *(device const float4*)(x1 + col_lo);
            const float4 x_hi = *(device const float4*)(x1 + col_hi);
            acc0.y += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
        }
        if (active2) {
            const float4 x_lo = *(device const float4*)(x2 + col_lo);
            const float4 x_hi = *(device const float4*)(x2 + col_hi);
            acc0.z += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
        }
        if (active3) {
            const float4 x_lo = *(device const float4*)(x3 + col_lo);
            const float4 x_hi = *(device const float4*)(x3 + col_hi);
            acc0.w += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
        }
        if (active4) {
            const float4 x_lo = *(device const float4*)(x4 + col_lo);
            const float4 x_hi = *(device const float4*)(x4 + col_hi);
            acc1.x += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
        }
        if (active5) {
            const float4 x_lo = *(device const float4*)(x5 + col_lo);
            const float4 x_hi = *(device const float4*)(x5 + col_hi);
            acc1.y += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
        }
        if (active6) {
            const float4 x_lo = *(device const float4*)(x6 + col_lo);
            const float4 x_hi = *(device const float4*)(x6 + col_hi);
            acc1.z += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
        }
        if (active7) {
            const float4 x_lo = *(device const float4*)(x7 + col_lo);
            const float4 x_hi = *(device const float4*)(x7 + col_hi);
            acc1.w += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
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
