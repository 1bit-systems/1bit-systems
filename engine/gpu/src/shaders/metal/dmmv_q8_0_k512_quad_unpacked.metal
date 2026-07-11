#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Pre-cycle-70 variant of dmmv_q8_0_k512_quad.metal.
//
// Identical kernel except the per-simdgroup eight-row reduction tail uses
// eight scalar `simd_sum(float)` calls (and a lane-0 serial writeback fan-out)
// instead of the cycle-70 packed `simd_sum(float4)` low/high pair. Kept
// reachable so the prefill path (selected when `engine.in_prefill_phase ==
// true`) can sidestep the cross-effort prefill drift the float4 pack
// contributed to without giving up the per-decode-token win it banks on the
// Qwen3.6 shared-expert down projection (M=2048, K=512, ~1436 calls/req).
// Mirrors cycle 79's pattern for `dmmv_q8_0_repacked_k2048_nr2_qwen.metal`.
// See the dispatch routing in the `prefer_qwen_shared_down_q8_quad` branch
// of `forward_metal.zig`.
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
    const uint base_row = (tg_id * simdgroups_per_tg + sg_idx) * 8u;
    if (base_row >= p.M) return;

    device const float* input = X + (p.x_offset >> 2);
    device float* output = Y + (p.y_offset >> 2);

    const uint blocks_per_row = 16u; // K=512, Q8_0 block size=32
    const ulong row_bytes = ulong(blocks_per_row) * 34ull;
    const bool has1 = base_row + 1u < p.M;
    const bool has2 = base_row + 2u < p.M;
    const bool has3 = base_row + 3u < p.M;
    const bool has4 = base_row + 4u < p.M;
    const bool has5 = base_row + 5u < p.M;
    const bool has6 = base_row + 6u < p.M;
    const bool has7 = base_row + 7u < p.M;

    device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
    device const uchar* row1 = has1 ? (row0 + row_bytes) : row0;
    device const uchar* row2 = has2 ? (row0 + 2ull * row_bytes) : row0;
    device const uchar* row3 = has3 ? (row0 + 3ull * row_bytes) : row0;
    device const uchar* row4 = has4 ? (row0 + 4ull * row_bytes) : row0;
    device const uchar* row5 = has5 ? (row0 + 5ull * row_bytes) : row0;
    device const uchar* row6 = has6 ? (row0 + 6ull * row_bytes) : row0;
    device const uchar* row7 = has7 ? (row0 + 7ull * row_bytes) : row0;

    const uint block_idx = lane >> 1;
    const uint half_sel = lane & 1u;
    const uint half_base = half_sel << 4; // 0 or 16 elements within the Q8_0 block
    const uint x_base = (block_idx << 5) + half_base;

    device const uchar* blk0 = row0 + block_idx * 34u;
    device const uchar* blk1 = row1 + block_idx * 34u;
    device const uchar* blk2 = row2 + block_idx * 34u;
    device const uchar* blk3 = row3 + block_idx * 34u;
    device const uchar* blk4 = row4 + block_idx * 34u;
    device const uchar* blk5 = row5 + block_idx * 34u;
    device const uchar* blk6 = row6 + block_idx * 34u;
    device const uchar* blk7 = row7 + block_idx * 34u;

    const float s0 = float(as_type<half>(*(device const ushort*)(blk0)));
    const float s1 = float(as_type<half>(*(device const ushort*)(blk1)));
    const float s2 = float(as_type<half>(*(device const ushort*)(blk2)));
    const float s3 = float(as_type<half>(*(device const ushort*)(blk3)));
    const float s4 = float(as_type<half>(*(device const ushort*)(blk4)));
    const float s5 = float(as_type<half>(*(device const ushort*)(blk5)));
    const float s6 = float(as_type<half>(*(device const ushort*)(blk6)));
    const float s7 = float(as_type<half>(*(device const ushort*)(blk7)));

    device const packed_char4* q0 = (device const packed_char4*)(blk0 + 2u + half_base);
    device const packed_char4* q1 = (device const packed_char4*)(blk1 + 2u + half_base);
    device const packed_char4* q2 = (device const packed_char4*)(blk2 + 2u + half_base);
    device const packed_char4* q3 = (device const packed_char4*)(blk3 + 2u + half_base);
    device const packed_char4* q4 = (device const packed_char4*)(blk4 + 2u + half_base);
    device const packed_char4* q5 = (device const packed_char4*)(blk5 + 2u + half_base);
    device const packed_char4* q6 = (device const packed_char4*)(blk6 + 2u + half_base);
    device const packed_char4* q7 = (device const packed_char4*)(blk7 + 2u + half_base);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    float acc4 = 0.0f;
    float acc5 = 0.0f;
    float acc6 = 0.0f;
    float acc7 = 0.0f;

    #pragma unroll
    for (uint vi = 0u; vi < 4u; ++vi) {
        const float4 x = *(device const float4*)(input + x_base + (vi << 2));
        acc0 = fma(s0, dot(float4(char4(q0[vi])), x), acc0);
        acc1 = fma(s1, dot(float4(char4(q1[vi])), x), acc1);
        acc2 = fma(s2, dot(float4(char4(q2[vi])), x), acc2);
        acc3 = fma(s3, dot(float4(char4(q3[vi])), x), acc3);
        acc4 = fma(s4, dot(float4(char4(q4[vi])), x), acc4);
        acc5 = fma(s5, dot(float4(char4(q5[vi])), x), acc5);
        acc6 = fma(s6, dot(float4(char4(q6[vi])), x), acc6);
        acc7 = fma(s7, dot(float4(char4(q7[vi])), x), acc7);
    }

    const float sum0 = simd_sum(acc0);
    const float sum1 = simd_sum(acc1);
    const float sum2 = simd_sum(acc2);
    const float sum3 = simd_sum(acc3);
    const float sum4 = simd_sum(acc4);
    const float sum5 = simd_sum(acc5);
    const float sum6 = simd_sum(acc6);
    const float sum7 = simd_sum(acc7);
    if (lane == 0u) {
        output[base_row] = sum0;
        if (has1) output[base_row + 1u] = sum1;
        if (has2) output[base_row + 2u] = sum2;
        if (has3) output[base_row + 3u] = sum3;
        if (has4) output[base_row + 4u] = sum4;
        if (has5) output[base_row + 5u] = sum5;
        if (has6) output[base_row + 6u] = sum6;
        if (has7) output[base_row + 7u] = sum7;
    }
}
