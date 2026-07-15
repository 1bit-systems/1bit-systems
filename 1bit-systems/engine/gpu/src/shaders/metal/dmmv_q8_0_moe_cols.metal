#include <metal_stdlib>
using namespace metal;

// Q8_0 grouped MoE DMMV for batched Gemma expert-down prefill.
//
// This is the route-column sibling of dmmv_q8_0_moe.metal. The decode shader
// consumes one token's selected expert IDs; this kernel consumes the packed
// expert-major route IDs produced by moe_route_pack_blocks.metal, matching the
// q4_k/q5_1/q5_k/q6_k column kernels used by structural prefill.

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
    uint lane                                 [[thread_index_in_simdgroup]],
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

    const uint nb = p.K / 32u;
    const ulong row_bytes = ulong(nb) * 34ull;
    const ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    device const uchar* src = W + expert_base + ulong(row) * row_bytes;

    float4 acc0 = float4(0.0f);
    float4 acc1 = float4(0.0f);

    for (uint b = lane; b < nb; b += 32u) {
        device const uchar* block = src + b * 34u;
        const float d = float(as_type<half>(*(device const ushort*)block));
        device const packed_char4* qs = (device const packed_char4*)(block + 2u);
        const uint base = b << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const float4 q = float4(char4(qs[vi]));
            const uint off = base + (vi << 2);
            acc0.x += active0 ? d * dot(q, *(device const float4*)(x0 + off)) : 0.0f;
            acc0.y += active1 ? d * dot(q, *(device const float4*)(x1 + off)) : 0.0f;
            acc0.z += active2 ? d * dot(q, *(device const float4*)(x2 + off)) : 0.0f;
            acc0.w += active3 ? d * dot(q, *(device const float4*)(x3 + off)) : 0.0f;
            acc1.x += active4 ? d * dot(q, *(device const float4*)(x4 + off)) : 0.0f;
            acc1.y += active5 ? d * dot(q, *(device const float4*)(x5 + off)) : 0.0f;
            acc1.z += active6 ? d * dot(q, *(device const float4*)(x6 + off)) : 0.0f;
            acc1.w += active7 ? d * dot(q, *(device const float4*)(x7 + off)) : 0.0f;
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
    if (lane == 0u) {
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
