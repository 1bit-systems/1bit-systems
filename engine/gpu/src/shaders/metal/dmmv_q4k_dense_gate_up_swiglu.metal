#include <metal_stdlib>
using namespace metal;

// Dense Qwen3 single-token gate/up Q4_K matvec fused with SwiGLU.
//
// Sibling of dmmv_q4k_dense_gate_up_geglu.metal — same row layout
// (NSG=2, NR0=2, 64 threads/threadgroup), same q4_K block dot product,
// only the activation differs:
//   SwiGLU(gate, up) = (gate * sigmoid(gate)) * up
// vs the GeGLU variant used by Gemma.
//
// Saves 2 dispatches + 1 barrier per Qwen3 dense FFN layer compared to
// the un-fused gate / up / swiglu sequence, and skips the DRAM round
// trip through gate_buf and up_buf for the intermediate FFN width.
//
// Cycle 5 (reverted): NR0 bump 2→4 (8 rows/TG) measured 43.66 vs 44.5
// baseline median (-2%) on Qwen3-8B M1 Max. The doubled per-TG arithmetic
// hurt more than the halved TG count helped — the GPU was already
// saturating on the 3072-TG dispatch and TG launch overhead is not the
// lever on M1 Max. Reverted to NR0=2.

struct DualQ4KDmmvPush {
    uint M0;
    uint M1;
    uint K;
    uint a0_offset;
    uint a1_offset;
    uint x_offset;
    uint y0_offset;
    uint y1_offset;
};

#define NSG 2
#define NR0 2
#define QK_K 256
#define BLOCK_SIZE 144
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#ifndef ZINC_Q4K_GATE_UP_SWIGLU_FIXED_BLOCKS
#define ZINC_Q4K_GATE_UP_SWIGLU_FIXED_BLOCKS 0
#endif

inline float swiglu(float gate, float up) {
    // SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
    // fast::exp maps to Apple GPU hardware exp2 (vs precise::exp polynomial).
    // fast::divide maps to Apple GPU hardware reciprocal+mul (vs precise IEEE
    // division), saving ~10 cycles per call. Fires inter_dim × n_layers
    // times per token (~442K calls/token on Qwen3-8B).
    return up * gate * fast::divide(1.0f, 1.0f + fast::exp(-gate));
}

kernel void main0(
    device const uchar* W0 [[buffer(0)]],
    device const uchar* W1 [[buffer(1)]],
    constant DualQ4KDmmvPush& p [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* activatedY [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

#if ZINC_Q4K_GATE_UP_SWIGLU_FIXED_BLOCKS
    constexpr int nb = ZINC_Q4K_GATE_UP_SWIGLU_FIXED_BLOCKS;
#else
    const int nb = p.K / QK_K;
#endif
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
#if ZINC_Q4K_GATE_UP_SWIGLU_FIXED_BLOCKS
    constexpr int row_bytes = ZINC_Q4K_GATE_UP_SWIGLU_FIXED_BLOCKS * BLOCK_SIZE;
#else
    const int row_bytes = nb * BLOCK_SIZE;
#endif

    device const uchar* gate_src = W0 + p.a0_offset;
    device const uchar* up_src = W1 + p.a1_offset;
    device const float* x = X + (p.x_offset / 4);
    device float* out = activatedY + (p.y0_offset / 4);

    float gate_sum[NR0] = {0.f, 0.f};
    float up_sum[NR0] = {0.f, 0.f};

    device const float* y4 = x + ix * QK_K + 64 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy;

        // Explicit float4 loads of the four 8-float slices (offsets 0,32,128,160
        // from y4 — all 32-byte aligned by construction of ix,iq,ir). This
        // forces 8×16-byte coalesced loads instead of relying on the compiler
        // to vectorize 32 scalar `y4[i]` reads across the `q4k_block_dot`
        // helper-function boundary that consumes yl/yh next. dot(v, 1) gives
        // the sumy partials in a single fused-mul-add chain per slice.
        const device float4* y4v = (const device float4*)y4;
        const float4 a0 = y4v[0];   const float4 a1 = y4v[1];
        const float4 b0 = y4v[8];   const float4 b1 = y4v[9];
        const float4 c0 = y4v[32];  const float4 c1 = y4v[33];
        const float4 d0 = y4v[40];  const float4 d1 = y4v[41];

        // Cycle 84: store the per-i y-gather directly as `float4 yl4_arr[4]`
        // / `float4 yh4_arr[4]` arrays instead of the legacy `float yl[16]`
        // / `float yh[16]` scalar layout. The inner FMA loop consumes
        // exactly one float4 per i from each side — yl4_arr[i] =
        // (yl[2i], yl[2i+1], yl[2i+8], yl[2i+9]) and similarly for yh —
        // which under the old layout was reconstructed each iteration via
        // a 4-lane indexed gather `float4(yl[2*i + 0], yl[2*i + 1],
        // yl[2*i + 8], yl[2*i + 9])`. The compiler usually folds these
        // gathers into register moves under FOR_UNROLL with compile-time
        // indices, but only when it keeps the 16-float-wide flat array
        // fully in SSA registers; once that array is fat enough, register
        // pressure can demote it to thread-private stack and the gather
        // becomes 4 scalar loads. Pre-storing as float4 yl4_arr[4] /
        // yh4_arr[4] makes the natural data shape explicit (8 vector
        // registers, no gather) and mirrors cycle 83's win on
        // dmmv_q6k_llama.metal. The lane mapping is yl4_arr[i] =
        // (a*.xy, b*.xy) for i∈{0,2} and (a*.zw, b*.zw) for i∈{1,3} where
        // a*/b* picks between (a0,b0)/(a1,b1) based on i/2. This shader
        // is the hottest Q4_K kernel — ffn_gate+ffn_up on Qwen3-8B dense
        // (~50% of Q4_K bytes/token, Q4_K = 71.6% of decode bytes/token).
        const float4 yl4_arr[4] = {
            float4(a0.xy, b0.xy),
            float4(a0.zw, b0.zw),
            float4(a1.xy, b1.xy),
            float4(a1.zw, b1.zw),
        };
        const float4 yh4_arr[4] = {
            float4(c0.xy, d0.xy),
            float4(c0.zw, d0.zw),
            float4(c1.xy, d1.xy),
            float4(c1.zw, d1.zw),
        };

        // Cycle 87: compute sumy from the yl4_arr / yh4_arr register vectors
        // built above instead of from the original a0/a1/b0/b1/c0/c1/d0/d1
        // loads. The 8 dot-of-ones partials (16 reads of a*/b*/c*/d* + 8
        // horizontal reductions + 4 scalar adds) form a separate consumer of
        // the original float4 y4v loads that runs *parallel* to the yl4_arr/
        // yh4_arr build — so the compiler must keep all 8 a*/b*/c*/d* vectors
        // live until both consumers complete. Since the inner FMA loop (lines
        // 255-296) only ever reads from yl4_arr/yh4_arr, we can derive sumy
        // from those same arrays and let a*/b*/c*/d* die at the shuffle.
        //
        // Lane mapping: yl4_arr[0]=(a0.x,a0.y,b0.x,b0.y), [1]=(a0.z,a0.w,b0.z,
        // b0.w), [2]=(a1.x,a1.y,b1.x,b1.y), [3]=(a1.z,a1.w,b1.z,b1.w). Summing
        // i=0..3 gives partials at lanes 0,1 = (a0+a1) split into halves and
        // lanes 2,3 = (b0+b1) split. So (yl_tot.x + yl_tot.y) = sum(a0+a1) =
        // sumy[0], (yl_tot.z + yl_tot.w) = sum(b0+b1) = sumy[1], and the
        // analogous holds for yh4_arr → sumy[2,3].
        //
        // Cost: 6 float4 adds + 4 scalar adds (vs prior 8 dot4 + 4 scalar);
        // both shapes are ~28 effective adds, so this is a register-pressure
        // optimization rather than an arithmetic-count reduction. Freeing
        // 8 float4 registers (32 floats, ~128B of register file) earlier in
        // the schedule gives the compiler more room for the q1v/q2v loads
        // and the 32 FMA chain that follow. Hottest Q4_K shader (ffn_gate +
        // ffn_up on Qwen3-8B dense, ~50% of Q4_K bytes/token).
        const float4 yl_tot = (yl4_arr[0] + yl4_arr[1]) + (yl4_arr[2] + yl4_arr[3]);
        const float4 yh_tot = (yh4_arr[0] + yh4_arr[1]) + (yh4_arr[2] + yh4_arr[3]);
        sumy[0] = yl_tot.x + yl_tot.y;
        sumy[1] = yl_tot.z + yl_tot.w;
        sumy[2] = yh_tot.x + yh_tot.y;
        sumy[3] = yh_tot.z + yh_tot.w;

        // Cycle 34: inline the q4k_block_dot_pair helper and interleave a
        // 4-way row0/row1 × gate/up FOR_UNROLL. Loads all 4 blocks' q1v/q2v/
        // sc16/dh up front and folds 64 independent FMAs into a single i=0..3
        // unrolled loop — extends cycle 33's row0/row1 inline pattern (which
        // landed in dmmv_q4k.metal) by also interleaving the gate/up axis on
        // top, matching the cross-axis interleaving cycle 32 applied within
        // the helper. Frees the compiler to schedule 4 independent FMA chains
        // simultaneously, which the helper-function boundary previously
        // blocked. Covers ffn_gate+ffn_up on Qwen3-8B dense path (~50% of
        // Q4_K bytes/token).
        constexpr ushort kmask1 = 0x3f3f;
        constexpr ushort kmask2 = 0x0f0f;
        constexpr ushort kmask3 = 0xc0c0;

        const int dst_row_0 = first_row + 0;
        const int dst_row_1 = first_row + 1;
        const ulong row_off_0 = ulong(dst_row_0) * ulong(row_bytes) + ulong(ib) * BLOCK_SIZE;
        const ulong row_off_1 = ulong(dst_row_1) * ulong(row_bytes) + ulong(ib) * BLOCK_SIZE;

        device const uchar* block_g0 = gate_src + row_off_0;
        device const uchar* block_u0 = up_src + row_off_0;
        device const uchar* block_g1 = gate_src + row_off_1;
        device const uchar* block_u1 = up_src + row_off_1;

        // Cycle 80: fuse the half2 dh-load (block+0..3) and packed_uint3 sc_u
        // load (block+4..15) per row site into one `packed_uint4` 16-byte
        // block-header load. Q4_K block layout places [d (half), dmin (half),
        // sc_u (12 bytes)] contiguously at offsets 0..15 with 4-byte alignment
        // — exactly the natural shape for packed_uint4 (16 bytes, 4-byte
        // aligned). Per ib × 4 row sites (g0, u0, g1, u1): collapses 8 device
        // loads (4 half2 + 4 packed_uint3) → 4 packed_uint4 loads. Builds on
        // cycle 66 (half2 dh) + cycle 74 (packed_uint3 sc_u) by collapsing
        // them into the natural single-block-header read shape, mirroring
        // cycle 78's win on dmmv_q4k_qk_dual.metal (+0.6 tok/s) and cycle 79's
        // port to dmmv_q4k.metal. This shader handles ffn_gate + ffn_up on
        // Qwen3-8B dense (~50% of Q4_K bytes/token, the hottest Q4_K kernel).
        const packed_uint4 hdr_g0 = *((device const packed_uint4*)block_g0);
        const packed_uint4 hdr_u0 = *((device const packed_uint4*)block_u0);
        const packed_uint4 hdr_g1 = *((device const packed_uint4*)block_g1);
        const packed_uint4 hdr_u1 = *((device const packed_uint4*)block_u1);
        const uint sc_shift = uint(iq) * 16u;
        device const ushort* q1_g0 = (device const ushort*)(block_g0 + 16) + 16 * iq + 4 * ir;
        device const ushort* q1_u0 = (device const ushort*)(block_u0 + 16) + 16 * iq + 4 * ir;
        device const ushort* q1_g1 = (device const ushort*)(block_g1 + 16) + 16 * iq + 4 * ir;
        device const ushort* q1_u1 = (device const ushort*)(block_u1 + 16) + 16 * iq + 4 * ir;
        const half2 dh_g0_h2 = as_type<half2>(hdr_g0.x);
        const half2 dh_u0_h2 = as_type<half2>(hdr_u0.x);
        const half2 dh_g1_h2 = as_type<half2>(hdr_g1.x);
        const half2 dh_u1_h2 = as_type<half2>(hdr_u1.x);

        // Cycle 70: store sc16 as `ushort4` register vectors instead of
        // stack-allocated `ushort[4]` arrays accessed via `(uchar*)` byte
        // alias. The previous form forced the compiler to materialize
        // sc16 in thread-private memory so the byte alias could read
        // individual lanes; the new form keeps the four packed scales in
        // SSA-eligible registers and lets the per-ib sc_pos / sc_neg
        // byte gathers compile to vector AND + vector shift rather than
        // 8 scalar uchar loads from spilled stack memory. Port of cycle
        // 69 (same change in dmmv_q4k.metal, 2 sc16 sites); this shader
        // has 4 sc16 sites (g0, u0, g1, u1) so the impact area is ~2×:
        // 32 scalar uchar loads per ib → 16 packed byte-extractions.
        // dmmv_q4k_dense_gate_up_swiglu.metal is the hottest Q4_K
        // shader (~50% of Q4_K bytes/token = ffn_gate+ffn_up on
        // Qwen3-8B dense; Q4_K = 71.6% of decode bytes/token).
        const uint3 sc_u3v_g0 = uint3(hdr_g0.y, hdr_g0.z, hdr_g0.w);
        const ushort sc_0_g0 = ushort((sc_u3v_g0.x >> sc_shift) & 0xFFFFu);
        const ushort sc_2_g0 = ushort((sc_u3v_g0.y >> sc_shift) & 0xFFFFu);
        const ushort sc_4_g0 = ushort((sc_u3v_g0.z >> sc_shift) & 0xFFFFu);
        const ushort4 sc16_g0 = ushort4(
            sc_0_g0 & kmask1,
            sc_2_g0 & kmask1,
            ((sc_4_g0 >> 0) & kmask2) | ((sc_0_g0 & kmask3) >> 2),
            ((sc_4_g0 >> 4) & kmask2) | ((sc_2_g0 & kmask3) >> 2));

        const uint3 sc_u3v_u0 = uint3(hdr_u0.y, hdr_u0.z, hdr_u0.w);
        const ushort sc_0_u0 = ushort((sc_u3v_u0.x >> sc_shift) & 0xFFFFu);
        const ushort sc_2_u0 = ushort((sc_u3v_u0.y >> sc_shift) & 0xFFFFu);
        const ushort sc_4_u0 = ushort((sc_u3v_u0.z >> sc_shift) & 0xFFFFu);
        const ushort4 sc16_u0 = ushort4(
            sc_0_u0 & kmask1,
            sc_2_u0 & kmask1,
            ((sc_4_u0 >> 0) & kmask2) | ((sc_0_u0 & kmask3) >> 2),
            ((sc_4_u0 >> 4) & kmask2) | ((sc_2_u0 & kmask3) >> 2));

        const uint3 sc_u3v_g1 = uint3(hdr_g1.y, hdr_g1.z, hdr_g1.w);
        const ushort sc_0_g1 = ushort((sc_u3v_g1.x >> sc_shift) & 0xFFFFu);
        const ushort sc_2_g1 = ushort((sc_u3v_g1.y >> sc_shift) & 0xFFFFu);
        const ushort sc_4_g1 = ushort((sc_u3v_g1.z >> sc_shift) & 0xFFFFu);
        const ushort4 sc16_g1 = ushort4(
            sc_0_g1 & kmask1,
            sc_2_g1 & kmask1,
            ((sc_4_g1 >> 0) & kmask2) | ((sc_0_g1 & kmask3) >> 2),
            ((sc_4_g1 >> 4) & kmask2) | ((sc_2_g1 & kmask3) >> 2));

        const uint3 sc_u3v_u1 = uint3(hdr_u1.y, hdr_u1.z, hdr_u1.w);
        const ushort sc_0_u1 = ushort((sc_u3v_u1.x >> sc_shift) & 0xFFFFu);
        const ushort sc_2_u1 = ushort((sc_u3v_u1.y >> sc_shift) & 0xFFFFu);
        const ushort sc_4_u1 = ushort((sc_u3v_u1.z >> sc_shift) & 0xFFFFu);
        const ushort4 sc16_u1 = ushort4(
            sc_0_u1 & kmask1,
            sc_2_u1 & kmask1,
            ((sc_4_u1 >> 0) & kmask2) | ((sc_0_u1 & kmask3) >> 2),
            ((sc_4_u1 >> 4) & kmask2) | ((sc_2_u1 & kmask3) >> 2));

        const ushort4 q1v_g0 = *((device const ushort4*)q1_g0);
        const ushort4 q2v_g0 = *((device const ushort4*)(q1_g0 + 32));
        const ushort4 q1v_u0 = *((device const ushort4*)q1_u0);
        const ushort4 q2v_u0 = *((device const ushort4*)(q1_u0 + 32));
        const ushort4 q1v_g1 = *((device const ushort4*)q1_g1);
        const ushort4 q2v_g1 = *((device const ushort4*)(q1_g1 + 32));
        const ushort4 q1v_u1 = *((device const ushort4*)q1_u1);
        const ushort4 q2v_u1 = *((device const ushort4*)(q1_u1 + 32));

        float4 acc1_g0 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2_g0 = {0.f, 0.f, 0.f, 0.f};
        float4 acc1_u0 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2_u0 = {0.f, 0.f, 0.f, 0.f};
        float4 acc1_g1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2_g1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc1_u1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2_u1 = {0.f, 0.f, 0.f, 0.f};

        constexpr ushort4 nibble_mask = ushort4(0x000F, 0x0F00, 0x00F0, 0xF000);

        FOR_UNROLL (short i = 0; i < 4; ++i) {
            // Cycle 49: vectorize the per-quant nibble-mask expansion. Cycle 46
            // already packed the FMAs into explicit float4 form, but each
            // `float4 q1m_X = float4(qi & 0x000F, qi & 0x0F00, qi & 0x00F0, qi & 0xF000)`
            // still expressed 4 scalar AND ops + 4 scalar int→float conversions
            // per nibble-set. Replace with `float4(ushort4(qi) & nibble_mask)` —
            // explicit broadcast-then-vector-AND-then-vector-convert. Each
            // ushort4(qi) is a register splat, the AND lowers to a single 4-wide
            // vector AND against the constexpr mask, and the float4(ushort4)
            // conversion is a single vector instruction on Apple7 instead of 4
            // separate lane converts. 8 mask expansions × 4 unrolled i iterations
            // = 32 expansions per ib iteration in this hottest Q4_K shader
            // (~50% of Q4_K bytes/token = ffn_gate+ffn_up on Qwen3-8B). Mirrors
            // the cycle 44-48 philosophy of telling the compiler the SIMD shape
            // explicitly instead of relying on it to lift lane-by-lane forms.
            const float4 yl4 = yl4_arr[i];
            const float4 yh4 = yh4_arr[i];
            const ushort q1_g0i = q1v_g0[i];
            const ushort q1_u0i = q1v_u0[i];
            const ushort q1_g1i = q1v_g1[i];
            const ushort q1_u1i = q1v_u1[i];
            const ushort q2_g0i = q2v_g0[i];
            const ushort q2_u0i = q2v_u0[i];
            const ushort q2_g1i = q2v_g1[i];
            const ushort q2_u1i = q2v_u1[i];
            const float4 q1m_g0 = float4(ushort4(q1_g0i) & nibble_mask);
            const float4 q1m_u0 = float4(ushort4(q1_u0i) & nibble_mask);
            const float4 q1m_g1 = float4(ushort4(q1_g1i) & nibble_mask);
            const float4 q1m_u1 = float4(ushort4(q1_u1i) & nibble_mask);
            const float4 q2m_g0 = float4(ushort4(q2_g0i) & nibble_mask);
            const float4 q2m_u0 = float4(ushort4(q2_u0i) & nibble_mask);
            const float4 q2m_g1 = float4(ushort4(q2_g1i) & nibble_mask);
            const float4 q2m_u1 = float4(ushort4(q2_u1i) & nibble_mask);
            acc1_g0 = fma(yl4, q1m_g0, acc1_g0);
            acc1_u0 = fma(yl4, q1m_u0, acc1_u0);
            acc1_g1 = fma(yl4, q1m_g1, acc1_g1);
            acc1_u1 = fma(yl4, q1m_u1, acc1_u1);
            acc2_g0 = fma(yh4, q2m_g0, acc2_g0);
            acc2_u0 = fma(yh4, q2m_u0, acc2_u0);
            acc2_g1 = fma(yh4, q2m_g1, acc2_g1);
            acc2_u1 = fma(yh4, q2m_u1, acc2_u1);
        }

        // Cycle 52: port cycle 51's vectorized per-ib reduction from
        // dmmv_q4k.metal here. Replace each row's 4-term head sum
        // `(acc[even] + 1/256*acc[odd]) * sc8[*]` and 4-term tail
        // `sumy[k]*sc8[*]` with one `fma`-built `head_pair` float4 plus a
        // `dot(head_pair, sc_pos4)` and a `dot(sumy, sc_neg4)`. Two rows ×
        // (gate + up) = 4 independent reductions per ib here vs 2 in the
        // base kernel, so ~2× the impact area of cycle 51. Maps the per-
        // block accumulator collapse onto Apple7's 4-wide ALU shape. The
        // swiglu kernel handles ffn_gate + ffn_up on Qwen3-8B dense
        // (~50% of Q4_K bytes/token), the hottest Q4_K kernel.
        const float4 head_pair_g0 = fma(
            float4(acc1_g0[1], acc1_g0[3], acc2_g0[1], acc2_g0[3]),
            float4(1.f / 256.f),
            float4(acc1_g0[0], acc1_g0[2], acc2_g0[0], acc2_g0[2]));
        const float4 head_pair_u0 = fma(
            float4(acc1_u0[1], acc1_u0[3], acc2_u0[1], acc2_u0[3]),
            float4(1.f / 256.f),
            float4(acc1_u0[0], acc1_u0[2], acc2_u0[0], acc2_u0[2]));
        const float4 head_pair_g1 = fma(
            float4(acc1_g1[1], acc1_g1[3], acc2_g1[1], acc2_g1[3]),
            float4(1.f / 256.f),
            float4(acc1_g1[0], acc1_g1[2], acc2_g1[0], acc2_g1[2]));
        const float4 head_pair_u1 = fma(
            float4(acc1_u1[1], acc1_u1[3], acc2_u1[1], acc2_u1[3]),
            float4(1.f / 256.f),
            float4(acc1_u1[0], acc1_u1[2], acc2_u1[0], acc2_u1[2]));
        // Cycle 70: derive sc_pos / sc_neg via vector byte-extraction
        // from the ushort4 sc16. sc8_X[0..7] (the old uchar* alias)
        // maps to {sc16.x.lo, sc16.x.hi, sc16.y.lo, sc16.y.hi,
        // sc16.z.lo, sc16.z.hi, sc16.w.lo, sc16.w.hi}, so:
        //   sc_pos = (sc16.x.lo, sc16.x.hi/16, sc16.z.lo, sc16.z.hi/16)
        //   sc_neg = (sc16.y.lo, sc16.y.hi,    sc16.w.lo, sc16.w.hi)
        // Builds each float4 in 1 vector AND + 1 ushort4→float4 widen
        // (+ 1 vector mul for sc_pos), replacing 4 scalar byte loads +
        // 2 scalar muls. × 4 sc16 sites per ib (g0, u0, g1, u1).
        constexpr ushort4 lo_mask = ushort4(0x00FFu);
        constexpr float4 sc_pos_scale = float4(1.f, 1.f / 16.f, 1.f, 1.f / 16.f);
        const float4 sc_pos_g0 = float4(ushort4(sc16_g0.x, sc16_g0.x >> 8, sc16_g0.z, sc16_g0.z >> 8) & lo_mask) * sc_pos_scale;
        const float4 sc_pos_u0 = float4(ushort4(sc16_u0.x, sc16_u0.x >> 8, sc16_u0.z, sc16_u0.z >> 8) & lo_mask) * sc_pos_scale;
        const float4 sc_pos_g1 = float4(ushort4(sc16_g1.x, sc16_g1.x >> 8, sc16_g1.z, sc16_g1.z >> 8) & lo_mask) * sc_pos_scale;
        const float4 sc_pos_u1 = float4(ushort4(sc16_u1.x, sc16_u1.x >> 8, sc16_u1.z, sc16_u1.z >> 8) & lo_mask) * sc_pos_scale;
        const float4 sc_neg_g0 = float4(ushort4(sc16_g0.y, sc16_g0.y >> 8, sc16_g0.w, sc16_g0.w >> 8) & lo_mask);
        const float4 sc_neg_u0 = float4(ushort4(sc16_u0.y, sc16_u0.y >> 8, sc16_u0.w, sc16_u0.w >> 8) & lo_mask);
        const float4 sc_neg_g1 = float4(ushort4(sc16_g1.y, sc16_g1.y >> 8, sc16_g1.w, sc16_g1.w >> 8) & lo_mask);
        const float4 sc_neg_u1 = float4(ushort4(sc16_u1.y, sc16_u1.y >> 8, sc16_u1.w, sc16_u1.w >> 8) & lo_mask);
        // Cycle 60: vectorize the cross-(gate,up) × cross-row final reduction.
        // The 4 independent per-block updates here (gate0, up0, gate1, up1)
        // each compute `dh[0] * dot(head, sc_pos) - dh[1] * dot(sumy, sc_neg)`
        // — 4 scalar `a*b - c*d` chains in lane-by-lane form. Pack the 4
        // (dh_d, dh_dmin, head_dot, tail_dot) tuples into float4s and fold
        // the 4 reductions into one vector mul + one vector fnma. Mirrors
        // cycle 51-53's "vectorize the per-ib reduction" pattern, but
        // applied at the outer (row × projection) axis instead of the
        // inner head/tail axis. Apple7 ALU is 4-wide; the indexed scalar
        // form was 4 separate scalar mul + 4 scalar mul + 4 scalar sub per
        // ib (12 scalar ops); the new form is 1 vec mul + 1 vec fnma + 4
        // scalar accumulator adds (~6 ops). This is the hottest Q4_K shader
        // (~50% of Q4_K bytes/token = ffn_gate+ffn_up on Qwen3-8B dense).
        //
        // Cycles 66/80: dh half2 pairs come from the fused packed_uint4
        // block-header load up top (see cycle 80 comment near hdr_*).
        //
        // Cycle 82: express the per-ib cross-row delta as an explicit
        // `fma(dh_d, head_dots, -dh_dmin * tail_dots)` instead of the
        // algebraically equivalent `dh_d * head_dots - dh_dmin * tail_dots`.
        // The plain `a*b - c*d` form is 2 vector muls + 1 vector sub (3 ops)
        // and only collapses to 1 mul + 1 fma when the metal compiler is
        // willing to fuse across the `-` (IEEE-strict mode won't, fast::*
        // usually does — but the helper is consumed across an explicit
        // float4 boundary). The explicit fma form is unconditionally 1
        // vector mul + 1 vector fma, matching the cycle 61 pattern that
        // already landed in dmmv_q4k.metal line 283 (cross-row 2-wide) and
        // cycle 64's port to dmmv_q4k_qk_dual.metal line 233. This shader
        // is the only Q4_K sibling still using the implicit form — porting
        // makes the 3 hot Q4_K matvec kernels uniform on the same explicit
        // fma shape. Hottest Q4_K shader (~50% of Q4_K bytes/token = 71.6%
        // of decode bytes/token via Q4_K).
        const float4 dh_d = float4(float(dh_g0_h2.x), float(dh_u0_h2.x), float(dh_g1_h2.x), float(dh_u1_h2.x));
        const float4 dh_dmin = float4(float(dh_g0_h2.y), float(dh_u0_h2.y), float(dh_g1_h2.y), float(dh_u1_h2.y));
        const float4 head_dots = float4(
            dot(head_pair_g0, sc_pos_g0),
            dot(head_pair_u0, sc_pos_u0),
            dot(head_pair_g1, sc_pos_g1),
            dot(head_pair_u1, sc_pos_u1));
        const float4 tail_dots = float4(
            dot(sumy, sc_neg_g0),
            dot(sumy, sc_neg_u0),
            dot(sumy, sc_neg_g1),
            dot(sumy, sc_neg_u1));
        const float4 delta = fma(dh_d, head_dots, -dh_dmin * tail_dots);
        gate_sum[0] += delta[0];
        up_sum[0]   += delta[1];
        gate_sum[1] += delta[2];
        up_sum[1]   += delta[3];

        y4 += 4 * QK_K;
    }

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const int dst_row = first_row + row;
        if (dst_row < int(p.M0)) {
            const float gate_total = simd_sum(gate_sum[row]);
            const float up_total = simd_sum(up_sum[row]);
            if (tiisg == 0) {
                out[dst_row] = swiglu(gate_total, up_total);
            }
        }
    }
}
