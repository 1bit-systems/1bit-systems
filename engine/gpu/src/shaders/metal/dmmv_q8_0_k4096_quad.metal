#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Q8_0 DMMV specialization for Qwen3.6 SSM out projections (K=4096).
//
// Four adjacent rows share each loaded activation vector inside one simdgroup.
// The weight stream remains dominant, but this halves simdgroup count versus
// the accepted nr=2 K=4096 path and mirrors the kept K=2048 quad geometry.
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

    const ulong row_bytes = 4352ull; // 128 Q8_0 blocks * 34 bytes
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

    #pragma unroll
    for (uint chunk = 0u; chunk < 4u; ++chunk) {
        const uint bi = lane + chunk * 32u;
        device const uchar* blk0 = row0 + bi * 34u;
        device const uchar* blk1 = row1 + bi * 34u;
        device const uchar* blk2 = row2 + bi * 34u;
        device const uchar* blk3 = row3 + bi * 34u;

        const float s0 = float(as_type<half>(*(device const ushort*)(blk0)));
        const float s1 = float(as_type<half>(*(device const ushort*)(blk1)));
        const float s2 = float(as_type<half>(*(device const ushort*)(blk2)));
        const float s3 = float(as_type<half>(*(device const ushort*)(blk3)));

        device const packed_char4* q0 = (device const packed_char4*)(blk0 + 2u);
        device const packed_char4* q1 = (device const packed_char4*)(blk1 + 2u);
        device const packed_char4* q2 = (device const packed_char4*)(blk2 + 2u);
        device const packed_char4* q3 = (device const packed_char4*)(blk3 + 2u);
        const uint x_base = bi << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const float4 x = *(device const float4*)(input + x_base + (vi << 2));
            acc0 = fma(s0, dot(float4(char4(q0[vi])), x), acc0);
            acc1 = fma(s1, dot(float4(char4(q1[vi])), x), acc1);
            acc2 = fma(s2, dot(float4(char4(q2[vi])), x), acc2);
            acc3 = fma(s3, dot(float4(char4(q3[vi])), x), acc3);
        }
    }

    // Cycle ~71: pack the four final-reduction `simd_sum` calls into one
    // `simd_sum(float4)` — Apple9 lowers vector `simd_sum` to a single
    // log2(32)=5-level butterfly that transfers 128-bit packed lanes per
    // shuffle_xor instead of four independent 32-bit trees, cutting cross-lane
    // shuffle traffic ~4× on the per-simdgroup tail of the Qwen3.6 SSM out
    // projection (M=2048, K=4096) and full-attention output projection (same
    // shape) — both share this quad-row Q8 K=4096 path. Per profile: 1080
    // SSM-out calls/req × 32 TGs × 16 SGs = ~552K simdgroup-tail reductions
    // per request, plus the full-attn out path. Same proven pattern as
    // cycle ~62 (`dmmv_q5k_moe_k512_quad`), cycle ~64 (`dmmv_q8_0_pair_swiglu`),
    // and cycle ~70 (`dmmv_q8_0_k512_quad` — 8-row variant via two float4
    // packs). Downstream lane<4 writeback consumes the four sums as
    // simdgroup-uniform scalars, so picking float4 components by lane is
    // bit-equivalent. When has1/has2/has3 are false (M%4 != 0 tail), row1/2/3
    // alias row0 so acc1/2/3 equal acc0 — pre-pack simd_sums were already
    // executed unconditionally and discarded by the has gates, so the pack
    // preserves identical semantics.
    const float4 sums = simd_sum(float4(acc0, acc1, acc2, acc3));
    // Parallelize the 4-row writeback across lanes 0..3 (lane 0 serial 4
    // stores → lanes 0..3 coalesced 16-byte store). After simd_sum all four
    // sums are broadcast to every lane in registers; base_row+0..+3 are four
    // contiguous floats so the Qwen3.6-35B production SSM/attn out case
    // (M=2048, M%4==0 ⇒ every simdgroup writes a full quad) issues a single
    // coalesced 16-byte store instead of four serial lane-0 stores. The
    // has1/has2/has3 predicates handle the generic-validation tail (test
    // M=21 ⇒ last simdgroup has only row 20 valid). Mirrors cycle-27/32/38/
    // 39/40/41/43/45/62/64/70 lane-parallel writeback discipline across the
    // Q8/Q5_K family.
    if (lane < 4u) {
        const bool has = (lane == 0u) ? true
                       : (lane == 1u) ? has1
                       : (lane == 2u) ? has2
                       : has3;
        if (has) {
            const float local_sum = (lane == 0u) ? sums.x
                                  : (lane == 1u) ? sums.y
                                  : (lane == 2u) ? sums.z
                                  : sums.w;
            output[base_row + lane] = local_sum;
        }
    }
}
