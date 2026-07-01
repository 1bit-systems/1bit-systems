#include <metal_stdlib>
using namespace metal;

// Push constants for DMMV dispatch (matches Zig DmmvPush layout).
struct DmmvPush {
    uint M;        // rows
    uint K;        // cols
    uint a_offset; // byte offset into weight matrix
    uint x_offset; // byte offset into input vector
    uint y_offset; // byte offset into output vector
};

// Port of the reference implementation's kernel_mul_mv_q4_K_f32 (non-ext variant).
// Matches the exact floating-point accumulation pattern for bit-identical results.
//
// Thread organization (matches the reference implementation with N_SG_Q4_K=2, N_R0_Q4_K=2):
//   64 threads per threadgroup = 2 simdgroups x 32 threads
//   Each simdgroup processes 2 rows => 4 rows per threadgroup
//
// Q4_K block layout (144 bytes, 256 elements):
//   [0..1]   d    (float16)
//   [2..3]   dmin (float16)
//   [4..15]  scales (12 bytes, packed 6-bit scale/min pairs)
//   [16..143] qs  (128 bytes, 256 x 4-bit quants)

#ifndef ZINC_Q4K_NSG
#define ZINC_Q4K_NSG 2
#endif

#define NSG   ZINC_Q4K_NSG
#define NR0   2
#define QK_K  256
#define BLOCK_SIZE 144
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#ifndef ZINC_Q4K_FIXED_BLOCKS
#define ZINC_Q4K_FIXED_BLOCKS 0
#endif

#ifndef ZINC_Q4K_ACCUMULATE_OUTPUT
#define ZINC_Q4K_ACCUMULATE_OUTPUT 0
#endif

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant DmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const short ix = tiisg / 8;  // 0..3
    const short it = tiisg % 8;  // 0..7
    const short iq = it / 4;     // 0 or 1
    const short ir = it % 4;     // 0..3

#if ZINC_Q4K_FIXED_BLOCKS
    constexpr int nb = ZINC_Q4K_FIXED_BLOCKS;
#else
    const int nb = p.K / QK_K;   // blocks per row
#endif

    const int r0 = tgpig.x;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    // nb01 in the reference implementation is the byte stride per row = nb * sizeof(block_q4_K)
#if ZINC_Q4K_FIXED_BLOCKS
    constexpr int nb01 = ZINC_Q4K_FIXED_BLOCKS * BLOCK_SIZE;
#else
    const int nb01 = nb * BLOCK_SIZE;
#endif

    device const uchar* src0 = W + p.a_offset;
    device const float* src1 = X + (p.x_offset / 4);

    device const uchar* x_base = src0 + (uint64_t)first_row * nb01;
    device const float* y = src1;

    // Keep the reference implementation's NR0=2 row pair in a vector accumulator. The per-block
    // fold already produces a float2 delta; carrying that shape to the final
    // simd reductions avoids a two-slot thread-private scalar array.
    float2 sumf = float2(0.0f);

    device const float* y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy;

        // Explicit float4 loads of the four 8-float slices (offsets 0,32,128,160
        // from y4 — all 32-byte aligned by construction of ix,iq,ir). Forces
        // 8×16-byte coalesced loads instead of 32 scalar `y4[i]` reads, and folds
        // the sumy partials into 4 fused dot(v,1) chains. Mirrors the cycle 23
        // pattern that landed in dmmv_q4k_dense_gate_up_swiglu.metal; this kernel
        // handles attn_qkv/attn_o/ffn_down on the Qwen3 dense path (~53% of all
        // Q4_K bytes/token, complementing the ~50% FFN gate/up traffic already
        // covered by the swiglu variant).
        const device float4* y4v = (const device float4*)y4;
        const float4 a0 = y4v[0];   const float4 a1 = y4v[1];
        const float4 b0 = y4v[8];   const float4 b1 = y4v[9];
        const float4 c0 = y4v[32];  const float4 c1 = y4v[33];
        const float4 d0 = y4v[40];  const float4 d1 = y4v[41];

        // Cycle 85: store the per-i y-gather directly as `float4 yl4_arr[4]`
        // / `float4 yh4_arr[4]` register-vector arrays instead of the legacy
        // `float yl[16]` / `float yh[16]` scalar layout. The inner FMA loop
        // consumes exactly one float4 per i from each side — yl4_arr[i] =
        // (yl[2i], yl[2i+1], yl[2i+8], yl[2i+9]) — which under the old
        // layout was reconstructed each iteration via a 4-lane indexed
        // gather `float4(yl[2*i + 0], yl[2*i + 1], yl[2*i + 8], yl[2*i + 9])`
        // off 16-wide flat stack arrays filled via 32 scalar lane writes.
        // Eliminates 32 scalar stack writes and 8 4-lane gather
        // reconstructions per ib, keeping all y data in SSA-eligible vector
        // registers. The lane mapping is yl4_arr[i] = (a*.xy, b*.xy) for
        // i∈{0,2} and (a*.zw, b*.zw) for i∈{1,3} where a*/b* picks between
        // (a0,b0)/(a1,b1) based on i/2. Mirrors cycle 84's port to
        // dmmv_q4k_dense_gate_up_swiglu.metal (hottest Q4_K shader,
        // ffn_gate+ffn_up) and cycle 83's original in dmmv_q6k_llama.metal.
        // dmmv_q4k.metal serves attn_qkv V + attn_o + ffn_down on Qwen3-8B
        // dense (~53% of Q4_K bytes/token slice, complementing the swiglu
        // kernel's ffn_gate+ffn_up share; Q4_K = 71.6% of decode bytes/token).
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

        // Cycle 88: port cycle 87's sumy-from-yl4_arr derivation. Compute
        // sumy from the yl4_arr / yh4_arr register vectors built above
        // instead of from the original a0/a1/b0/b1/c0/c1/d0/d1 float4 loads.
        // The 8 dot-of-ones partials (16 reads of a*/b*/c*/d* + 8 horizontal
        // reductions + 4 scalar adds) form a separate consumer of the
        // original y4v float4 loads that runs *parallel* to the yl4_arr/
        // yh4_arr shuffle build — so the compiler must keep all 8 a*/b*/c*/d*
        // vectors live until both consumers complete. Since the inner FMA
        // loop only ever reads from yl4_arr / yh4_arr (lines 217-218), we
        // can derive sumy from those same arrays and let a*/b*/c*/d* die
        // immediately after the shuffle.
        //
        // Lane mapping (from yl4_arr build above): yl4_arr[0]=(a0.x,a0.y,
        // b0.x,b0.y), [1]=(a0.z,a0.w,b0.z,b0.w), [2]=(a1.x,a1.y,b1.x,b1.y),
        // [3]=(a1.z,a1.w,b1.z,b1.w). Summing i=0..3 gives partials at
        // lanes 0,1 = (a0+a1) split into halves and lanes 2,3 = (b0+b1)
        // split. So (yl_tot.x + yl_tot.y) = sum(a0+a1) = sumy[0] and
        // (yl_tot.z + yl_tot.w) = sum(b0+b1) = sumy[1]; analogous for
        // yh4_arr → sumy[2,3].
        //
        // Cost: 6 float4 adds + 4 scalar adds vs prior 8 dot4 + 4 scalar
        // (~28 effective adds either way). The win is register-file: ~128B
        // of a*/b*/c*/d* state can be freed before the q1v/q2v ushort4
        // loads and the 32-FMA inner chain. Mirrors cycle 87's port to
        // dmmv_q4k_dense_gate_up_swiglu.metal. dmmv_q4k.metal serves
        // attn_qkv V + attn_o + ffn_down on Qwen3-8B dense (~53% of Q4_K
        // bytes/token, complementing the swiglu kernel's ~50%
        // ffn_gate+ffn_up share; Q4_K = 71.6% of decode bytes/token).
        const float4 yl_tot = (yl4_arr[0] + yl4_arr[1]) + (yl4_arr[2] + yl4_arr[3]);
        const float4 yh_tot = (yh4_arr[0] + yh4_arr[1]) + (yh4_arr[2] + yh4_arr[3]);
        sumy[0] = yl_tot.x + yl_tot.y;
        sumy[1] = yl_tot.z + yl_tot.w;
        sumy[2] = yh_tot.x + yh_tot.y;
        sumy[3] = yh_tot.z + yh_tot.w;

        // Cycle 33: interleave row0/row1 of NR0=2. Load both rows' sc_u/q1/q2/dh
        // up front and run a single FOR_UNROLL i=0..3 that updates both rows'
        // acc1/acc2 alternately. Same algorithm as the sequential row loop, but
        // removes the per-row serialization chain (sumf[0] depends on row-0
        // accumulator completion before row-1's loads start in the original).
        // Mirrors cycle 32's interleaved gate+up pattern from the swiglu helper,
        // applied here across row0/row1 of the same matrix instead of across
        // gate/up matrices. Covers attn_qkv/attn_o/ffn_down on Qwen3 dense.
        //
        // Cycle 79: port cycle 78's packed_uint4 block-header fusion to
        // dmmv_q4k.metal. Fuses each row's 4-byte half2 dh-load (block+0..3)
        // and 12-byte packed_uint3 sc_u-load (block+4..15) into a single
        // 16-byte coalesced `packed_uint4` load. The Q4_K block layout places
        // `[d (half), dmin (half), sc_u (12 bytes)]` contiguously at offsets
        // 0..15 with 4-byte alignment — exactly the natural shape for
        // packed_uint4 (16 bytes, 4-byte aligned). Per ib × NR0=2 rows this
        // folds 4 device loads (2 × half2 + 2 × packed_uint3) → 2
        // packed_uint4 loads. The 8 bytes of dh-pair register state for
        // both rows is held across the full per-ib body until the final
        // cross-row fold — negligible. Builds on cycle 67 (half2 dh-load
        // here) + cycle 73 (packed_uint3 sc_u-load here) by collapsing them
        // into the natural single-block-header read shape, mirroring cycle
        // 78's win on dmmv_q4k_qk_dual.metal (49.4 → 50.0 tok/s).
        // dmmv_q4k.metal serves attn_qkv V + attn_o + ffn_down on Qwen3-8B
        // dense (~53% Q4_K slice, complementing the swiglu kernel's
        // ffn_gate+ffn_up share).
        device const uchar* blk_0 = x_base + (uint64_t)ib * BLOCK_SIZE;
        device const uchar* blk_1 = blk_0 + nb01;
        const packed_uint4 hdr_0 = *((device const packed_uint4*)blk_0);
        const packed_uint4 hdr_1 = *((device const packed_uint4*)blk_1);
        const half2 dh_pair_0 = as_type<half2>(hdr_0.x);
        const half2 dh_pair_1 = as_type<half2>(hdr_1.x);
        const uint sc_shift = uint(iq) * 16u;
        device const ushort* q1_0 = (device const ushort*)(blk_0 + 16) + 16 * iq + 4 * ir;
        device const ushort* q1_1 = q1_0 + nb01 / 2;

        // Cycle 73: collapse the 3 scalar `sc_u_X[0/1/2]` uint reads into a
        // single `packed_uint3` load per row. Now extracted from the
        // packed_uint4 block-header loaded above (cycle 79).
        // Cycle 69: store sc16 as `ushort4` register vectors instead of
        // stack-allocated `ushort[4]` arrays accessed via `(uchar*)` byte
        // alias. The previous form forced the compiler to materialize
        // sc16 in thread-private memory so the byte alias could read
        // individual lanes; the new form keeps the four packed scales in
        // SSA-eligible registers and lets the per-ib sc_pos / sc_neg
        // byte gathers compile to vector AND + vector shift rather than
        // 8 scalar uchar loads from spilled stack memory. Same compiler-
        // hint philosophy as cycles 49/50 (nibble-mask vectorize), 51
        // (per-ib reduction), and 61 (cross-row reduction).
        const uint3 sc_u3v_0 = uint3(hdr_0.y, hdr_0.z, hdr_0.w);
        const ushort sc_0_0 = ushort((sc_u3v_0.x >> sc_shift) & 0xFFFFu);
        const ushort sc_2_0 = ushort((sc_u3v_0.y >> sc_shift) & 0xFFFFu);
        const ushort sc_4_0 = ushort((sc_u3v_0.z >> sc_shift) & 0xFFFFu);
        const ushort4 sc16_0 = ushort4(
            sc_0_0 & kmask1,
            sc_2_0 & kmask1,
            ((sc_4_0 >> 0) & kmask2) | ((sc_0_0 & kmask3) >> 2),
            ((sc_4_0 >> 4) & kmask2) | ((sc_2_0 & kmask3) >> 2));

        const uint3 sc_u3v_1 = uint3(hdr_1.y, hdr_1.z, hdr_1.w);
        const ushort sc_0_1 = ushort((sc_u3v_1.x >> sc_shift) & 0xFFFFu);
        const ushort sc_2_1 = ushort((sc_u3v_1.y >> sc_shift) & 0xFFFFu);
        const ushort sc_4_1 = ushort((sc_u3v_1.z >> sc_shift) & 0xFFFFu);
        const ushort4 sc16_1 = ushort4(
            sc_0_1 & kmask1,
            sc_2_1 & kmask1,
            ((sc_4_1 >> 0) & kmask2) | ((sc_0_1 & kmask3) >> 2),
            ((sc_4_1 >> 4) & kmask2) | ((sc_2_1 & kmask3) >> 2));

        const ushort4 q1v_0 = *((device const ushort4*)q1_0);
        const ushort4 q2v_0 = *((device const ushort4*)(q1_0 + 32));
        const ushort4 q1v_1 = *((device const ushort4*)q1_1);
        const ushort4 q2v_1 = *((device const ushort4*)(q1_1 + 32));

        float4 acc1_0 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2_0 = {0.f, 0.f, 0.f, 0.f};
        float4 acc1_1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2_1 = {0.f, 0.f, 0.f, 0.f};

        constexpr ushort4 nibble_mask = ushort4(0x000F, 0x0F00, 0x00F0, 0xF000);

        FOR_UNROLL (short i = 0; i < 4; ++i) {
            // Cycle 45: replace 16 indexed scalar `accX_Y[k] += y* * (q* & mask)`
            // writes (4 acc-slots × 4 accumulators) per `i` iteration with 4
            // explicit float4 `fma(y4, q4, acc4)` calls. yl0..yl9 are shared
            // across NR0=2 rows, so packing them once into `yl4`/`yh4` and
            // pairing with per-row masked-quant float4s exposes the 4-wide
            // SIMD operation directly to the metal compiler instead of
            // relying on it to lift 4 lane-by-lane indexed writes into a
            // single vector FMA. Apple7 ALU is naturally 4-wide; the indexed
            // scalar form sometimes leaves the compiler emitting 4 narrower
            // FMAs back-to-back instead of a single quad FMA. Mirrors cycle
            // 44 (the same change to dmmv_q6k_llama.metal). Q4_K is 71.6% of
            // decode bytes/token vs Q6_K's 28.4%, so ~2.5× the impact area.
            const float4 yl4 = yl4_arr[i];
            const float4 yh4 = yh4_arr[i];
            const ushort q1_0i = q1v_0[i];
            const ushort q1_1i = q1v_1[i];
            const ushort q2_0i = q2v_0[i];
            const ushort q2_1i = q2v_1[i];
            // Cycle 50: vectorize 4-lane nibble-mask expansion. Replace
            // `float4(qi & 0x000F, qi & 0x0F00, qi & 0x00F0, qi & 0xF000)` —
            // 4 scalar ANDs + 4 scalar int→float casts per `qi` — with
            // `float4(ushort4(qi) & nibble_mask)`, which lowers to 1
            // ushort4 splat, 1 ushort4 AND vs a constexpr mask, then 1
            // ushort4→float4 widen. Mirrors cycle 49 (same change to
            // dmmv_q4k_dense_gate_up_swiglu.metal). dmmv_q4k.metal is the
            // generic Q4_K matvec used for attn_qkv / attn_o on Qwen3-8B,
            // a meaningful slice of the Q4_K 71.6% bytes/token share.
            const float4 q1m_0 = float4(ushort4(q1_0i) & nibble_mask);
            const float4 q1m_1 = float4(ushort4(q1_1i) & nibble_mask);
            const float4 q2m_0 = float4(ushort4(q2_0i) & nibble_mask);
            const float4 q2m_1 = float4(ushort4(q2_1i) & nibble_mask);
            acc1_0 = fma(yl4, q1m_0, acc1_0);
            acc1_1 = fma(yl4, q1m_1, acc1_1);
            acc2_0 = fma(yh4, q2m_0, acc2_0);
            acc2_1 = fma(yh4, q2m_1, acc2_1);
        }

        // Cycle 51: vectorize the per-ib reduction. Replace the 4-term
        // sum-of-(pair * sc) head expression with `dot(pair4, sc_pos4)` and
        // the 4-term `sumy[k]*sc8[*]` tail with `dot(sumy, sc_neg4)`, after
        // fusing the `acc*[even] + 1/256 * acc*[odd]` pair-builder into a
        // single vector `fma`. Mirrors cycles 44-50's "expose the SIMD shape
        // explicitly" philosophy — the indexed scalar form here was 4 scalar
        // FMAs + 4 muls + 3 adds (head) + 4 muls + 3 adds (tail) per row;
        // the new form is 1 vector FMA + 2 dot4 per row, the natural shape
        // for Apple7's 4-wide ALU. Two rows means we save ~22 scalar ops
        // per ib in favor of ~6 vector ops per ib. dmmv_q4k.metal serves
        // attn_qkv / attn_o / ffn_down on Qwen3-8B (Q4_K = 71.6% of
        // decode bytes/token).
        const float4 head_pair_0 = fma(
            float4(acc1_0[1], acc1_0[3], acc2_0[1], acc2_0[3]),
            float4(1.f / 256.f),
            float4(acc1_0[0], acc1_0[2], acc2_0[0], acc2_0[2]));
        const float4 head_pair_1 = fma(
            float4(acc1_1[1], acc1_1[3], acc2_1[1], acc2_1[3]),
            float4(1.f / 256.f),
            float4(acc1_1[0], acc1_1[2], acc2_1[0], acc2_1[2]));
        // Cycle 69: derive sc_pos / sc_neg via vector byte-extraction
        // from the ushort4 sc16. sc8_X[0..7] (the old uchar* alias)
        // maps to {sc16.x.lo, sc16.x.hi, sc16.y.lo, sc16.y.hi,
        // sc16.z.lo, sc16.z.hi, sc16.w.lo, sc16.w.hi}, so:
        //   sc_pos = (sc16.x.lo, sc16.x.hi/16, sc16.z.lo, sc16.z.hi/16)
        //   sc_neg = (sc16.y.lo, sc16.y.hi,    sc16.w.lo, sc16.w.hi)
        // Builds each float4 in 1 vector AND + 1 ushort4→float4 widen
        // (+ 1 vector mul for sc_pos), replacing 4 scalar byte loads +
        // 2 scalar muls. Per ib × NR0=2 rows × dmmv_q4k.metal's share
        // (~53% of Q4_K bytes/token = attn_qkv V + attn_o + lm_head on
        // Qwen3-8B dense).
        constexpr ushort4 lo_mask = ushort4(0x00FFu);
        const ushort4 sc_pos_bytes_0 = ushort4(sc16_0.x, sc16_0.x >> 8, sc16_0.z, sc16_0.z >> 8) & lo_mask;
        const ushort4 sc_pos_bytes_1 = ushort4(sc16_1.x, sc16_1.x >> 8, sc16_1.z, sc16_1.z >> 8) & lo_mask;
        constexpr float4 sc_pos_scale = float4(1.f, 1.f / 16.f, 1.f, 1.f / 16.f);
        const float4 sc_pos_0 = float4(sc_pos_bytes_0) * sc_pos_scale;
        const float4 sc_pos_1 = float4(sc_pos_bytes_1) * sc_pos_scale;
        const float4 sc_neg_0 = float4(ushort4(sc16_0.y, sc16_0.y >> 8, sc16_0.w, sc16_0.w >> 8) & lo_mask);
        const float4 sc_neg_1 = float4(ushort4(sc16_1.y, sc16_1.y >> 8, sc16_1.w, sc16_1.w >> 8) & lo_mask);
        // Cycle 61: port cycle 60's cross-row reduction vectorization. The two
        // independent `dh[0]*dot(head,sc_pos) - dh[1]*dot(sumy,sc_neg)` chains
        // (row 0, row 1) form a natural 2-wide ALU shape: pack the (d, dmin)
        // halves across both rows into float2s, the (head_dot, tail_dot) per
        // row into float2s, and fold the 2 scalar `a*b - c*d` chains into one
        // vec2 mul + one vec2 fma producing a float2 `delta` of the 2 per-row
        // increments. Mirrors cycle 60's 4-wide pattern in
        // dmmv_q4k_dense_gate_up_swiglu.metal, applied here at the cross-row
        // axis only (no gate/up pairing in the plain matvec). dmmv_q4k.metal
        // serves attn_qkv / attn_o / ffn_down on Qwen3-8B dense (Q4_K =
        // 71.6% of decode bytes/token, this kernel covers the ~53% slice
        // complementing the swiglu kernel's ffn_gate+ffn_up).
        // Cycle 67/79: the dh_pair_0/dh_pair_1 half2 values are now extracted
        // from the packed_uint4 block-header load at the top of this ib body
        // (cycle 79); build the cross-row `dh_d`/`dh_dmin` float2s via
        // `.x`/`.y` swizzles that lower to a single half2→float2 widen per
        // row instead of 2 scalar half→float casts.
        const float2 dh_d = float2(float(dh_pair_0.x), float(dh_pair_1.x));
        const float2 dh_dmin = float2(float(dh_pair_0.y), float(dh_pair_1.y));
        const float2 head_dots = float2(dot(head_pair_0, sc_pos_0), dot(head_pair_1, sc_pos_1));
        const float2 tail_dots = float2(dot(sumy, sc_neg_0), dot(sumy, sc_neg_1));
        const float2 delta = fma(dh_d, head_dots, -dh_dmin * tail_dots);
        sumf += delta;

        y4 += 4 * QK_K;
    }

    device float* dst_f32 = Y + (p.y_offset / 4);

    for (int row = 0; row < NR0 && first_row + row < (int)p.M; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
#if ZINC_Q4K_ACCUMULATE_OUTPUT
            dst_f32[first_row + row] += sum_all;
#else
            dst_f32[first_row + row] = sum_all;
#endif
        }
    }
}
