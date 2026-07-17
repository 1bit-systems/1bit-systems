#include <metal_stdlib>
using namespace metal;

struct DualQ8DmmvPush {
    uint M0;
    uint M1;
    uint K;
    uint a0_offset;
    uint a1_offset;
    uint x_offset;
    uint y0_offset;
    uint y1_offset;
};

// Paired equal-shape Q8_0 DMMV fused with SwiGLU.
//
// Qwen3.6 shared expert gate/up are both [512, 2048] Q8_0 matrices and the
// token-major path only consumes them as silu(gate) * up. Reuse the paired
// input loads from dmmv_q8_0_pair, but write the activation directly.
kernel void main0(
    constant DualQ8DmmvPush& p [[buffer(0)]],
    device const uchar* W0 [[buffer(1)]],
    device const uchar* W1 [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* Y [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    const uint base_row = (tg_id * simdgroups_per_tg + sg_idx) * 2u;
    if (base_row >= p.M0 || base_row >= p.M1) return;

    device const float* input = X + (p.x_offset >> 2);
    device float* output = Y + (p.y0_offset >> 2);

    const uint blocks_per_row = p.K >> 5;
    const ulong row_bytes = ulong(blocks_per_row) * 34ull;
    const bool has_next = base_row + 1u < p.M0 && base_row + 1u < p.M1;
    device const uchar* row00 = W0 + p.a0_offset + ulong(base_row) * row_bytes;
    device const uchar* row01 = has_next ? (row00 + row_bytes) : row00;
    device const uchar* row10 = W1 + p.a1_offset + ulong(base_row) * row_bytes;
    device const uchar* row11 = has_next ? (row10 + row_bytes) : row10;

    float gate0 = 0.0f;
    float gate1 = 0.0f;
    float up0 = 0.0f;
    float up1 = 0.0f;

    for (uint bi = lane; bi < blocks_per_row; bi += 32u) {
        device const uchar* blk00 = row00 + bi * 34u;
        device const uchar* blk01 = row01 + bi * 34u;
        device const uchar* blk10 = row10 + bi * 34u;
        device const uchar* blk11 = row11 + bi * 34u;
        const float s00 = float(as_type<half>(*(device const ushort*)(blk00)));
        const float s01 = float(as_type<half>(*(device const ushort*)(blk01)));
        const float s10 = float(as_type<half>(*(device const ushort*)(blk10)));
        const float s11 = float(as_type<half>(*(device const ushort*)(blk11)));
        device const packed_char4* q00 = (device const packed_char4*)(blk00 + 2u);
        device const packed_char4* q01 = (device const packed_char4*)(blk01 + 2u);
        device const packed_char4* q10 = (device const packed_char4*)(blk10 + 2u);
        device const packed_char4* q11 = (device const packed_char4*)(blk11 + 2u);
        const uint x_base = bi << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const float4 x = *(device const float4*)(input + x_base + (vi << 2));
            gate0 = fma(s00, dot(float4(char4(q00[vi])), x), gate0);
            gate1 = fma(s01, dot(float4(char4(q01[vi])), x), gate1);
            up0 = fma(s10, dot(float4(char4(q10[vi])), x), up0);
            up1 = fma(s11, dot(float4(char4(q11[vi])), x), up1);
        }
    }

    // Cycle ~64: pack the four final-reduction simd_sum calls into one
    // simd_sum(float4) — Apple's `simd_sum` is overloaded for vector types on
    // Apple9 (M4) and emits a single log2(32)=5-level shuffle_xor tree that
    // transfers 128-bit packed lanes per shuffle instead of four independent
    // 32-bit trees. Cuts shuffle_xor count ~4× on the per-simdgroup tail of
    // hot kernel #4 (224 ms/req across 1436 dispatches, ~16% of timed kernel
    // time). The downstream lane<2 paired writeback consumes the totals as
    // simdgroup-uniform scalars, so picking float4 components by lane is
    // bit-equivalent. Mirrors cycle-62 (Q5_K quad) and cycle-63 (Q4_K pair).
    const float4 totals = simd_sum(float4(gate0, up0, gate1, up1));
    // Distribute the two row writes across lanes 0 and 1 so the pair of
    // output stores at base_row..base_row+1 issues as one coalesced 8-byte
    // transaction (mirrors the cycle-27 conv1d-fused sibling and the cycle-32
    // 4-row pattern). Lane 1's write is gated on `has_next` for the boundary
    // simdgroup. Both gate1/up1 simd_sums are uniform within the simdgroup
    // (`has_next` is uniform), so hoisting them out is bit-equivalent.
    if (lane < 2u && (lane == 0u || has_next)) {
        const float gate_sum = (lane == 0u) ? totals.x : totals.z;
        const float up_sum = (lane == 0u) ? totals.y : totals.w;
        output[base_row + lane] = gate_sum * fast::divide(1.0f, 1.0f + fast::exp(-gate_sum)) * up_sum;
    }
}
