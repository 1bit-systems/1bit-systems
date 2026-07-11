#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Batched MXFP4 MoE DMMV. Each 512-thread threadgroup holds one input vector
// in threadgroup memory and assigns one simdgroup to each of 16 output rows.

struct MoeDmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint expert_stride;
    uint x_expert_stride;
    uint x_offset;
    uint y_offset;
};

constant float kvalues_mxfp4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline float e8m0_to_fp32(uchar x) {
    uint bits = (x == 0u) ? 0x00400000u : (uint(x) << 23);
    return as_type<float>(bits);
}

#define TG_SIZE 512
#define ROWS_PER_TG 16
#define MAX_K 4096

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant MoeDmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    device const uint* expert_ids [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float x_cache[MAX_K];

    const uint local_id = local_pos.x;
    const uint expert_slot = tg_pos.y;
    device const float* input = X + (p.x_offset >> 2) + expert_slot * p.x_expert_stride;
    for (uint i = local_id; i < p.K; i += TG_SIZE) {
        x_cache[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tg_pos.x * ROWS_PER_TG + sg_idx;
    if (row >= p.M) return;

    const uint expert_id = expert_ids[expert_slot];
    const uint blocks_per_row = p.K >> 5;
    const ulong row_bytes = ulong(blocks_per_row) * 17ul;
    const ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    device const uchar* src = W + expert_base + ulong(row) * row_bytes;

    float sum = 0.0f;
    for (uint b = lane; b < blocks_per_row; b += 32u) {
        device const uchar* block = src + ulong(b) * 17ul;
        const float d = e8m0_to_fp32(block[0]);
        device const uchar* qs = block + 1;
        const uint base = b << 5;

        #pragma unroll
        for (uint j = 0u; j < 16u; ++j) {
            const uchar q = qs[j];
            sum = fma(d * kvalues_mxfp4[q & 0x0Fu], x_cache[base + j], sum);
            sum = fma(d * kvalues_mxfp4[q >> 4], x_cache[base + 16u + j], sum);
        }
    }

    const float total = simd_sum(sum);
    if (lane == 0u) {
        device float* out = Y + (p.y_offset >> 2) + expert_slot * p.M;
        out[row] = total;
    }
}
