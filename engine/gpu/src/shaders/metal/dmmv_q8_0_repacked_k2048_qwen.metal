#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Qwen3.6 exact repacked Q8_0 DMMV for K=2048 and M%4==0.
//
// This is the tail-free sibling of dmmv_q8_0_repacked_k2048_quad.metal. It
// follows the reference implementation's fixed row-group Q8 matvec discipline and is only selected
// for production Qwen SSM/full-attn projections whose row counts are multiples
// of four; the generic quad kernel remains available for validation tail cases.
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

    constexpr ulong group_bytes = 1088ul;
    constexpr ulong row_bytes = 2176ul;

    device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
    device const uchar* row1 = row0 + row_bytes;
    device const uchar* row2 = row0 + 2ul * row_bytes;
    device const uchar* row3 = row0 + 3ul * row_bytes;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    #pragma unroll
    for (uint gi = 0u; gi < 2u; ++gi) {
        device const uchar* g0 = row0 + ulong(gi) * group_bytes;
        device const uchar* g1 = row1 + ulong(gi) * group_bytes;
        device const uchar* g2 = row2 + ulong(gi) * group_bytes;
        device const uchar* g3 = row3 + ulong(gi) * group_bytes;

        const float s0 = float(as_type<half>(*(device const ushort*)(g0 + lane * 2u)));
        const float s1 = float(as_type<half>(*(device const ushort*)(g1 + lane * 2u)));
        const float s2 = float(as_type<half>(*(device const ushort*)(g2 + lane * 2u)));
        const float s3 = float(as_type<half>(*(device const ushort*)(g3 + lane * 2u)));
        const uint x_base = (gi * 32u + lane) << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const uint qo = 64u + vi * 128u + lane * 4u;
            const char4 q0 = as_type<char4>(*(device const int*)(g0 + qo));
            const char4 q1 = as_type<char4>(*(device const int*)(g1 + qo));
            const char4 q2 = as_type<char4>(*(device const int*)(g2 + qo));
            const char4 q3 = as_type<char4>(*(device const int*)(g3 + qo));
            const float4 x = *(device const float4*)(input + x_base + (vi << 2));

            acc0 = fma(s0, dot(float4(q0), x), acc0);
            acc1 = fma(s1, dot(float4(q1), x), acc1);
            acc2 = fma(s2, dot(float4(q2), x), acc2);
            acc3 = fma(s3, dot(float4(q3), x), acc3);
        }
    }

    // Cycle ~73: pack the four final-reduction `simd_sum` calls into one
    // `simd_sum(float4)` — Apple9 lowers vector `simd_sum` to a single
    // log2(32)=5-level butterfly that transfers 128-bit packed lanes per
    // shuffle_xor instead of four independent 32-bit trees, cutting cross-lane
    // shuffle traffic ~4× on the per-simdgroup tail of the Qwen3.6 SSM gate
    // projection (M=4096, K=2048) and full-attn output projection (M=2048,
    // K=2048) — both share this Qwen exact-repacked Q8 K=2048 quad-row path.
    // Per profile: SSM gate ≈ 15K TGs/token × 32 decode tokens ≈ 480K
    // simdgroup-tail reductions per request + full-attn-out ≈ 80K/req. Same
    // proven pattern as cycle ~62 (`dmmv_q5k_moe_k512_quad`), cycle ~64
    // (`dmmv_q8_0_pair_swiglu`), cycle ~70 (`dmmv_q8_0_k512_quad` — 8-row
    // variant via two float4 packs), cycle ~71 (`dmmv_q8_0_k4096_quad` —
    // the K=4096 sibling of this kernel). Downstream lane<4 writeback
    // consumes the four sums as simdgroup-uniform scalars, so picking
    // float4 components by lane is bit-equivalent. The M%4==0 invariant
    // guaranteed by the dispatch site (`prefer_qwen_repacked_q8_quad`
    // requires `M % 4 == 0`) plus the `base_row >= p.M` early-return at
    // the top means all four rows owned by this simdgroup are always
    // valid — no aliasing safety dance needed (unlike the generic quad
    // siblings).
    const float4 sums = simd_sum(float4(acc0, acc1, acc2, acc3));
    // Parallelize the 4-row writeback across lanes 0..3 (already kept).
    // After the packed simd_sum all four components are broadcast on every
    // lane; base_row+0..+3 are four contiguous floats so lanes 0..3 of this
    // simdgroup issue a single coalesced 16-byte store. Mirrors cycle-27/
    // 32/38/39/40/41/43/45/62/64/70/71 lane-parallel writeback discipline.
    if (lane < 4u) {
        const float local_sum = (lane == 0u) ? sums.x
                              : (lane == 1u) ? sums.y
                              : (lane == 2u) ? sums.z
                              : sums.w;
        output[base_row + lane] = local_sum;
    }
}
