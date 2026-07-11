#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Large-M Q8_0 matvec for GPT-OSS lm_head (K <= 4096).
// Stages the input vector once per threadgroup and reuses it across 32 rows.

#define TG_SIZE 512
#define ROWS_PER_TG ((TG_SIZE / 32) * 2)
#define MAX_K_VEC4 1024

kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float4 x_cache4[MAX_K_VEC4];

    device const float* input = X + (p.x_offset >> 2);
    const uint k_vec4 = p.K >> 2;

    for (uint i = local_id; i < k_vec4; i += TG_SIZE) {
        x_cache4[i] = *(device const float4*)(input + (i << 2));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base_row = tg_id * ROWS_PER_TG + sg_idx * 2u;
    if (base_row >= p.M) return;

    device float* output = Y + (p.y_offset >> 2);
    const uint blocks_per_row = p.K >> 5;
    const ulong row_bytes = ulong(blocks_per_row) * 34ull;
    device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
    device const uchar* row1 = row0 + row_bytes;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint bi = lane; bi < blocks_per_row; bi += 32u) {
        device const uchar* blk0 = row0 + bi * 34u;
        device const uchar* blk1 = row1 + bi * 34u;
        const float s0 = float(as_type<half>(*(device const ushort*)(blk0)));
        const float s1 = float(as_type<half>(*(device const ushort*)(blk1)));
        device const packed_char4* q0 = (device const packed_char4*)(blk0 + 2u);
        device const packed_char4* q1 = (device const packed_char4*)(blk1 + 2u);
        const uint x_base = bi << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const float4 x = x_cache4[(x_base >> 2) + vi];
            acc0 = fma(s0, dot(float4(char4(q0[vi])), x), acc0);
            acc1 = fma(s1, dot(float4(char4(q1[vi])), x), acc1);
        }
    }

    const float sum0 = simd_sum(acc0);
    if (lane == 0u) output[base_row] = sum0;

    if (base_row + 1u < p.M) {
        const float sum1 = simd_sum(acc1);
        if (lane == 0u) output[base_row + 1u] = sum1;
    }
}
