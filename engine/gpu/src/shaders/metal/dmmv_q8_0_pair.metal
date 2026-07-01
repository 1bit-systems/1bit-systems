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

// Paired equal-shape Q8_0 DMMV.
//
// Each simdgroup computes two output rows from W0 and the matching two rows
// from W1, reusing the same X vector loads. This is intended for Gemma
// attention K/V projections where both matrices are [kv_dim, hidden_dim].
kernel void main0(
    constant DualQ8DmmvPush& p [[buffer(0)]],
    device const uchar* W0 [[buffer(1)]],
    device const uchar* W1 [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* Y0 [[buffer(4)]],
    device float* Y1 [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    const uint base_row = (tg_id * simdgroups_per_tg + sg_idx) * 2u;
    if (base_row >= p.M0 || base_row >= p.M1) return;

    device const float* input = X + (p.x_offset >> 2);
    device float* output0 = Y0 + (p.y0_offset >> 2);
    device float* output1 = Y1 + (p.y1_offset >> 2);

    const uint blocks_per_row = p.K >> 5;
    const ulong row_bytes = ulong(blocks_per_row) * 34ull;
    const bool has_next = base_row + 1u < p.M0 && base_row + 1u < p.M1;
    device const uchar* row00 = W0 + p.a0_offset + ulong(base_row) * row_bytes;
    device const uchar* row01 = has_next ? (row00 + row_bytes) : row00;
    device const uchar* row10 = W1 + p.a1_offset + ulong(base_row) * row_bytes;
    device const uchar* row11 = has_next ? (row10 + row_bytes) : row10;

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

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
            acc00 = fma(s00, dot(float4(char4(q00[vi])), x), acc00);
            acc01 = fma(s01, dot(float4(char4(q01[vi])), x), acc01);
            acc10 = fma(s10, dot(float4(char4(q10[vi])), x), acc10);
            acc11 = fma(s11, dot(float4(char4(q11[vi])), x), acc11);
        }
    }

    // Cycle ~85: pack the two `simd_sum(float2)` calls (cycle 83's intermediate
    // form) into a single `simd_sum(float4)` — Apple9 lowers vector `simd_sum`
    // to one log2(32)=5-level butterfly that transfers 128-bit packed lanes per
    // `shuffle_xor` instead of two parallel 64-bit trees. Same proven pattern
    // as cycle ~64 (sibling `dmmv_q8_0_pair_swiglu`'s float4 pack), cycle ~71
    // (`dmmv_q8_0_k4096_quad` 4→1 float4), and cycle ~76
    // (`dmmv_q8_0_repacked_k4096_qwen_gated` 4→1 float4). The per-row reduction
    // tree stays identical (5-level butterfly on each accumulator) — only the
    // packing density across lane width changes, so the pack is bit-equivalent.
    //
    // Combined with lane-parallel 4-row writeback (lanes 0..3 each write one
    // output instead of lane 0 serializing all 4 stores) the per-simdgroup tail
    // collapses to ONE store-instruction issue cycle from FOUR. Hot uses on
    // Qwen3.6: (a) full-attn K+V paired projection (M=kv_dim=512, K=2048,
    // ~360 calls/req via cycle-30 block=32 → 1 SG/TG × 256 TGs = 256 SG tails
    // per call ⇒ ~92K SG tails/req); (b) full-attn Q+attn_gate paired (M=4096
    // K=2048 via cycle-15 block=256); (c) shared expert gate/up fallback.
    // has_next is uniform within the simdgroup (depends only on base_row from
    // tg_id/sg_idx/simdgroups_per_tg), so picking sub_row/buf_select by lane
    // index is uniform-safe. When !has_next, acc01/acc11 alias acc00/acc10
    // (the loop reads row01==row00, row11==row10), so sums.z/w equal sums.x/y
    // — gated writes for lanes 2..3 still preserve correctness.
    //
    // Float4 component → output mapping:
    //   sums.x = Σacc00 → output0[base_row]       (lane 0)
    //   sums.y = Σacc10 → output1[base_row]       (lane 1)
    //   sums.z = Σacc01 → output0[base_row + 1]   (lane 2, gated by has_next)
    //   sums.w = Σacc11 → output1[base_row + 1]   (lane 3, gated by has_next)
    const float4 sums = simd_sum(float4(acc00, acc10, acc01, acc11));
    if (lane < 4u) {
        const uint sub_row = lane >> 1;
        const bool active = (sub_row == 0u) || has_next;
        if (active) {
            device float* out = ((lane & 1u) == 0u) ? output0 : output1;
            const float val = (lane == 0u) ? sums.x
                            : (lane == 1u) ? sums.y
                            : (lane == 2u) ? sums.z
                            :                sums.w;
            out[base_row + sub_row] = val;
        }
    }
}
