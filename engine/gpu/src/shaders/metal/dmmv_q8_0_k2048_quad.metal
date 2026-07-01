#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Q8_0 DMMV specialization for Qwen3.6 SSM qkv/gate projections (K=2048).
//
// This keeps the accepted four-row simdgroup geometry from dmmv_q8_0_quad, but
// bakes the hot K=2048 shape so each lane handles exactly two Q8 blocks without
// dynamic loop bounds or row-stride math in the inner path.
kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    const uint base_row = (tg_id * simdgroups_per_tg + sg_idx) * 4u;
    if (base_row >= p.M) return;

    device const float* input = X + (p.x_offset >> 2);
    device float* output = Y + (p.y_offset >> 2);

    const ulong row_bytes = 2176ull; // 64 Q8_0 blocks * 34 bytes
    const bool has1 = base_row + 1u < p.M;
    const bool has2 = base_row + 2u < p.M;
    const bool has3 = base_row + 3u < p.M;

    device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
    device const uchar* row1 = has1 ? (row0 + row_bytes) : row0;
    device const uchar* row2 = has2 ? (row0 + 2ull * row_bytes) : row0;
    device const uchar* row3 = has3 ? (row0 + 3ull * row_bytes) : row0;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    const uint bi0 = lane;
    device const uchar* blk00 = row0 + bi0 * 34u;
    device const uchar* blk01 = row1 + bi0 * 34u;
    device const uchar* blk02 = row2 + bi0 * 34u;
    device const uchar* blk03 = row3 + bi0 * 34u;

    const float s00 = float(as_type<half>(*(device const ushort*)(blk00)));
    const float s01 = float(as_type<half>(*(device const ushort*)(blk01)));
    const float s02 = float(as_type<half>(*(device const ushort*)(blk02)));
    const float s03 = float(as_type<half>(*(device const ushort*)(blk03)));

    device const packed_char4* q00 = (device const packed_char4*)(blk00 + 2u);
    device const packed_char4* q01 = (device const packed_char4*)(blk01 + 2u);
    device const packed_char4* q02 = (device const packed_char4*)(blk02 + 2u);
    device const packed_char4* q03 = (device const packed_char4*)(blk03 + 2u);
    const uint x_base0 = bi0 << 5;

    #pragma unroll
    for (uint vi = 0u; vi < 8u; ++vi) {
        const float4 x = *(device const float4*)(input + x_base0 + (vi << 2));
        acc0 = fma(s00, dot(float4(char4(q00[vi])), x), acc0);
        acc1 = fma(s01, dot(float4(char4(q01[vi])), x), acc1);
        acc2 = fma(s02, dot(float4(char4(q02[vi])), x), acc2);
        acc3 = fma(s03, dot(float4(char4(q03[vi])), x), acc3);
    }

    const uint bi1 = lane + 32u;
    device const uchar* blk10 = row0 + bi1 * 34u;
    device const uchar* blk11 = row1 + bi1 * 34u;
    device const uchar* blk12 = row2 + bi1 * 34u;
    device const uchar* blk13 = row3 + bi1 * 34u;

    const float s10 = float(as_type<half>(*(device const ushort*)(blk10)));
    const float s11 = float(as_type<half>(*(device const ushort*)(blk11)));
    const float s12 = float(as_type<half>(*(device const ushort*)(blk12)));
    const float s13 = float(as_type<half>(*(device const ushort*)(blk13)));

    device const packed_char4* q10 = (device const packed_char4*)(blk10 + 2u);
    device const packed_char4* q11 = (device const packed_char4*)(blk11 + 2u);
    device const packed_char4* q12 = (device const packed_char4*)(blk12 + 2u);
    device const packed_char4* q13 = (device const packed_char4*)(blk13 + 2u);
    const uint x_base1 = bi1 << 5;

    #pragma unroll
    for (uint vi = 0u; vi < 8u; ++vi) {
        const float4 x = *(device const float4*)(input + x_base1 + (vi << 2));
        acc0 = fma(s10, dot(float4(char4(q10[vi])), x), acc0);
        acc1 = fma(s11, dot(float4(char4(q11[vi])), x), acc1);
        acc2 = fma(s12, dot(float4(char4(q12[vi])), x), acc2);
        acc3 = fma(s13, dot(float4(char4(q13[vi])), x), acc3);
    }

    const float sum0 = simd_sum(acc0);
    const float sum1 = simd_sum(acc1);
    const float sum2 = simd_sum(acc2);
    const float sum3 = simd_sum(acc3);
    if (lane == 0u) {
        output[base_row] = sum0;
        if (has1) output[base_row + 1u] = sum1;
        if (has2) output[base_row + 2u] = sum2;
        if (has3) output[base_row + 3u] = sum3;
    }
}
