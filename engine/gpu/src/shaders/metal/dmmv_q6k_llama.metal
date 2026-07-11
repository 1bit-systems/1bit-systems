#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

inline float fp16_to_fp32(uint h) {
    return float(as_type<half>(ushort(h)));
}

inline float s8_to_f32(uint x) {
    return float((x < 128u) ? int(x) : (int(x) - 256));
}

// Port of the reference implementation's kernel_mul_mv_q6_K_f32 for dense single-token decode.
// Default thread organization matches the reference implementation's N_SG_Q6_K=2, N_R0_Q6_K=2:
// 64 threads = 2 simdgroups, each simdgroup computes 2 rows. Exact M4
// Qwen3.6 27B dense-down builds can override NSG at compile time to keep the
// same per-simdgroup row-pair math while grouping more independent row pairs
// into one Metal threadgroup.
#ifndef ZINC_Q6K_NSG
#define ZINC_Q6K_NSG 2
#endif

#define NSG ZINC_Q6K_NSG
#define NR0 2
#define QK_K 256
#define BLOCK_SIZE 210
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#ifndef ZINC_Q6K_FIXED_BLOCKS
#define ZINC_Q6K_FIXED_BLOCKS 0
#endif

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant DmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
#if ZINC_Q6K_FIXED_BLOCKS
    constexpr uint nb = ZINC_Q6K_FIXED_BLOCKS;
    constexpr uint row_bytes = ZINC_Q6K_FIXED_BLOCKS * BLOCK_SIZE;
#else
    const uint nb = p.K / QK_K;
    const uint row_bytes = nb * BLOCK_SIZE;
#endif
    const uint first_row = (tgpig.x * NSG + uint(sgitg)) * NR0;

    device const uchar* src0 = W + p.a_offset;
    device const float* src1 = X + (p.x_offset / 4u);

    constexpr uint kmask1 = 0x03u;
    constexpr uint kmask2 = 0x0Cu;
    constexpr uint kmask3 = 0x30u;
    constexpr uint kmask4 = 0xC0u;

    const ushort tid = tiisg / 2u;
    const ushort ix = tiisg % 2u;
    const ushort ip = tid / 8u;
    const ushort il = tid % 8u;
    const ushort l0 = 4u * il;
    const ushort is = 8u * ip + l0 / 16u;

    const uint y_offset = 128u * uint(ip) + uint(l0);
    const uint q_offset_l = 64u * uint(ip) + uint(l0);
    const uint q_offset_h = 32u * uint(ip) + uint(l0);

    // Keep the reference implementation's NR0=2 row pair as a vector accumulator. The inner
    // loop already builds a float2 delta for row0/row1; carrying that shape to
    // the final simd reductions avoids a two-slot thread-private scalar array.
    float2 sumf = float2(0.0f);

    for (uint bi = ix; bi < nb; bi += 2u) {
        device const float* y = src1 + bi * QK_K + y_offset;

        // Cycle 83: store the per-l y-gather directly as a `float4 yl4_arr[4]`
        // array instead of the previous `float yl[16]` scalar layout. The
        // inner FMA loop and the per-block yl_sum4 reduction both consume
        // exactly one float4 per l (the four lanes are y[l], y[l+32],
        // y[l+64], y[l+96]); under the old layout each consumer constructed
        // its float4 via a 4-lane indexed gather over yl[4l..4l+3] (twice
        // per ib: once at yl_sum4 construction time, four times inside the
        // FOR_UNROLL). The compiler usually folds those gathers into
        // register moves when FOR_UNROLL exposes all indices as compile-time
        // constants — but only when it keeps yl[] fully in SSA registers,
        // which is not guaranteed for a 16-float-wide flat array (it can
        // demote to thread-private stack on register pressure). Pre-storing
        // as float4 yl4_arr[4] makes the natural data shape explicit:
        // 4 vector registers, no gather, and yl_sum4 collapses to a tree of
        // 3 vector adds. Load the four contiguous 4-float slices as aligned
        // float4s, then transpose them into the per-l vectors consumed below.
        // dmmv_q6k_llama.metal handles lm_head on Qwen3-8B Q4_K_M (Q6_K =
        // 28.4% of decode bytes/token = 38.46 GiB/step).
        const device float4* y4v = (const device float4*)y;
        const float4 y0 = y4v[0];
        const float4 y1 = y4v[8];
        const float4 y2 = y4v[16];
        const float4 y3 = y4v[24];
        const float4 yl4_arr[4] = {
            float4(y0.x, y1.x, y2.x, y3.x),
            float4(y0.y, y1.y, y2.y, y3.y),
            float4(y0.z, y1.z, y2.z, y3.z),
            float4(y0.w, y1.w, y2.w, y3.w),
        };

        // Cycle 43: hoist the -32 zero-point subtraction out of the inner FMA loop.
        // Q6_K dequant is d * scale * (raw_q - 32) where raw_q ∈ [0,63]. Rewriting
        // d * Σ_k sc[2k] · Σ_l yl[4l+k] · (q_lk - 32) as
        // d * (Σ_k sc[2k] · raw_sum[k] - 32 · Σ_k sc[2k] · yl_sum[k])
        // lets the inner-loop quant production drop from
        //   float(int((q1b & 0x0F) | ((h & m) << 4)) - 32)
        // to
        //   float((q1b & 0x0F) | ((h & m) << 4))
        // saving 1 int-sub per quant (8/i × 4 = 32 ops per block per row, ×2 rows = 64).
        // yl_sum is shared across NR0=2 rows; the per-row tail correction adds
        // 4 fmas + 1 fnma per row. Mirrors the Q4_K dh_min · sumy correction
        // pattern that already lives in dmmv_q4k.metal's tail.
        //
        // Cycle 59: vectorize the 4 strided yl_sum reductions. The 4 sums
        // yl_sum_k = Σ_l yl[4l+k] for k=0..3 are 4 independent reductions
        // across the 4 "slabs" yl[0..3], yl[4..7], yl[8..11], yl[12..15] —
        // exactly what a 4-wide vector add chain expresses naturally. Replace
        // 12 scalar adds with 3 float4 adds in a 4-way tree
        // ((a+b)+(c+d)). Same compiler-hint pattern as cycles 44-58 — tell
        // the compiler the 4-wide ALU shape directly instead of letting it
        // infer it from indexed lane writes that may schedule as narrower
        // FMAs. Q6_K covers ffn_down on Qwen3-8B (28.4% of decode bytes/
        // token = 38.46 GiB/step).
        const float4 yl_sum4 = (yl4_arr[0] + yl4_arr[1]) + (yl4_arr[2] + yl4_arr[3]);

        // Cycle 36: interleave NR0=2 row0/row1 — load both rows' q1v4/q2v4/qhv4/
        // sc/d up front, then run a single FOR_UNROLL l=0..3 that updates both
        // rows' sums alternately. Removes the per-row serialization chain (row 1's
        // loads previously waited on row 0's accumulator chain). Mirrors cycle 33's
        // row-interleave pattern that landed in dmmv_q4k.metal (+0.3 tok/s on dense
        // Q4_K). Bounds check moved to output simd_sum loop (matches dmmv_q4k.metal
        // pattern); dispatched M for ffn_down (4096) and lm_head (151936) on
        // Qwen3-8B are both divisible by NSG*NR0=4.
        device const uchar* block0 = src0 + ulong(first_row) * ulong(row_bytes) + ulong(bi) * BLOCK_SIZE;
        device const uchar* block1 = block0 + row_bytes;

        const uchar4 q1v4_0 = *((device const uchar4*)(block0 + q_offset_l));
        const uchar4 q2v4_0 = *((device const uchar4*)(block0 + q_offset_l + 32u));
        const uchar4 qhv4_0 = *((device const uchar4*)(block0 + 128u + q_offset_h));
        // Cycle 42: cast scales pointer to `device const char*` (signed) so the
        // 4 per-block scale reads compile to ld.s8 + scvtf directly, replacing
        // the s8_to_f32 helper's ld.u8 + branch (or select) + scvtf chain. Q6_K
        // scales are signed 8-bit in [-128,127]; the old helper paid for the
        // sign-extend at runtime, the new path bakes it into the load opcode.
        // 8 calls per inner-loop iteration (4 scales × 2 rows) × nb=16 × 36
        // layers × decode step. Same memory traffic, fewer ALU ops.
        device const char* sc_0 = (device const char*)(block0 + 192u + uint(is));
        // Cycle 38: replace 2 scalar uchar reads + bit-shift + as_type<half> with a
        // single 2-byte aligned half load. block+208 is 2-byte aligned (BLOCK_SIZE=210
        // is even, row_bytes = nb*210 is even, base buffer is ≥16-byte aligned by Metal
        // convention). Saves the uint construction (block[208] | (block[209] << 8))
        // and the ushort→half bit-cast, leaving a direct half→float promotion.
        const float d_0 = float(*((device const half*)(block0 + 208)));

        const uchar4 q1v4_1 = *((device const uchar4*)(block1 + q_offset_l));
        const uchar4 q2v4_1 = *((device const uchar4*)(block1 + q_offset_l + 32u));
        const uchar4 qhv4_1 = *((device const uchar4*)(block1 + 128u + q_offset_h));
        device const char* sc_1 = (device const char*)(block1 + 192u + uint(is));
        const float d_1 = float(*((device const half*)(block1 + 208)));

        float4 sums_0 = float4(0.0f);
        float4 sums_1 = float4(0.0f);
        FOR_UNROLL (ushort l = 0u; l < 4u; ++l) {
            // Cycle 58: vectorize the per-lane bit-twiddling of Q6_K dequant.
            // The four output lanes [q00, q10, q20, q30] each take a 2-bit
            // window from `h` and OR it into the high bits of a 4-bit nibble
            // from q1b/q2b. The 2-bit windows are at positions 0..1, 2..3,
            // 4..5, 6..7 of h, all placed at output bit positions 4..5 —
            // i.e. a sliding-window extraction with stride 2. That collapses
            // to `(ushort4(h, h>>2, h>>4, h>>6) & 0x03) << 4`, a vector
            // right-shift splat + uniform AND + uniform left-shift. The
            // nibble part decomposes symmetrically: ushort4(q1b, q2b, q1b,
            // q2b) split into low/high via one ushort2 AND and one ushort2
            // shift. Replaces 16 scalar bit ops per row (8 ANDs + 4 shifts +
            // 4 ORs) with ~6 vector ops per row; same compiler-hint pattern
            // as cycles 44-50 on Q4_K. Q6_K covers ffn_down on Qwen3-8B
            // (28.4% of decode bytes/token).
            const ushort h0 = ushort(qhv4_0[l]);
            const ushort q1b0 = ushort(q1v4_0[l]);
            const ushort q2b0 = ushort(q2v4_0[l]);
            const ushort2 q12_0 = ushort2(q1b0, q2b0);
            const ushort4 q_base_0 = ushort4(q12_0 & ushort2(0x0F), q12_0 >> ushort2(4));
            const ushort4 h_part_0 = (ushort4(h0, h0 >> 2, h0 >> 4, h0 >> 6) & ushort4(0x03)) << ushort4(4);

            const ushort h1 = ushort(qhv4_1[l]);
            const ushort q1b1 = ushort(q1v4_1[l]);
            const ushort q2b1 = ushort(q2v4_1[l]);
            const ushort2 q12_1 = ushort2(q1b1, q2b1);
            const ushort4 q_base_1 = ushort4(q12_1 & ushort2(0x0F), q12_1 >> ushort2(4));
            const ushort4 h_part_1 = (ushort4(h1, h1 >> 2, h1 >> 4, h1 >> 6) & ushort4(0x03)) << ushort4(4);

            // Cycle 44: replace 16 indexed scalar `sums_X[k] += ylv * q` writes
            // (8 per row × 2 rows) with 2 explicit `float4` `fma(yl4, q4_X, sums_X)`
            // calls. ylv0..ylv3 are shared between row0 and row1, so packing them
            // once and using vector FMA exposes the 4-wide SIMD operation directly
            // to the metal compiler instead of relying on it to lift indexed
            // per-lane writes into a single vector op. Apple7 ALU is naturally
            // 4-wide; the indexed scalar form sometimes leaves the compiler
            // emitting 4 narrower FMAs back-to-back instead of a single quad FMA.
            const float4 yl4 = yl4_arr[l];
            const float4 q4_0 = float4(q_base_0 | h_part_0);
            const float4 q4_1 = float4(q_base_1 | h_part_1);
            sums_0 = fma(yl4, q4_0, sums_0);
            sums_1 = fma(yl4, q4_1, sums_1);
        }

        // Cycle 56: port cycle 51's vectorized per-block reduction from
        // dmmv_q4k.metal to this Q6_K kernel. Replace the two scalar 4-term
        // (sums·sc) head sums and 4-term (yl_sum·sc) tail sums per row with
        // one `fma(-32, dot(yl_sum,sc4), dot(sums,sc4))` per row. Per ib
        // iteration: ~16 scalar muls + 14 scalar adds → 4 vector dot4 + 2
        // vector fma. Mirrors the same change that landed in dmmv_q4k.metal
        // (cycle 51), dmmv_q4k_dense_gate_up_swiglu.metal (cycle 52), and
        // dmmv_q4k_qk_dual.metal (cycle 53). Q6_K is 28.4% of decode bytes/
        // token on Qwen3-8B Q4_K_M (38.46 GiB/step) — this kernel covers
        // ffn_down. Apple7 ALU is naturally 4-wide; the indexed scalar form
        // leaves the compiler emitting narrower lane-by-lane FMAs.
        // Q6_K uses scale bytes sc[0,2,4,6] for this lane group. Read them
        // as two unaligned packed vectors per row instead of four scalar
        // byte loads, matching the packed-load discipline used by the
        // reference-derived Q4_K row-pair kernels.
        const packed_char4 sc_lo_0 = *((device const packed_char4*)sc_0);
        const packed_char4 sc_hi_0 = *((device const packed_char4*)(sc_0 + 4));
        const packed_char4 sc_lo_1 = *((device const packed_char4*)sc_1);
        const packed_char4 sc_hi_1 = *((device const packed_char4*)(sc_1 + 4));
        const float4 sc4_0 = float4(float(sc_lo_0.x), float(sc_lo_0.z), float(sc_hi_0.x), float(sc_hi_0.z));
        const float4 sc4_1 = float4(float(sc_lo_1.x), float(sc_lo_1.z), float(sc_hi_1.x), float(sc_hi_1.z));
        // Cycle 62: port cycles 60/61's cross-row reduction vectorization to
        // this Q6_K kernel. The two independent
        // `d * fma(-32, dot(yl_sum4, sc4), dot(sums, sc4))` chains (row 0,
        // row 1) form a natural 2-wide ALU shape: pack the per-row `d`
        // scalars into a float2, the (head_dot, tail_dot) reductions per
        // row into float2s, and fold the 2 scalar `d * (head - 32*tail)`
        // chains into one vec2 fma (build `head - 32*tail`) + one vec2 mul
        // by `d`, producing a float2 `delta` of the 2 per-row increments.
        // Mirrors cycle 61's 2-wide pattern on dmmv_q4k.metal applied at
        // the cross-row axis. dmmv_q6k_llama.metal serves ffn_down on
        // Qwen3-8B dense (Q6_K = 28.4% of decode bytes/token, 38.46 GiB
        // per step). Per ib iteration: 2 scalar fmas + 2 scalar muls
        // (~6 scalar ops) → 1 vec2 fma + 1 vec2 mul (~2 vector ops),
        // matching Apple7's natural SIMD shape.
        const float2 dh_d = float2(d_0, d_1);
        const float2 head_dots = float2(dot(sums_0, sc4_0), dot(sums_1, sc4_1));
        const float2 tail_dots = float2(dot(yl_sum4, sc4_0), dot(yl_sum4, sc4_1));
        const float2 delta = dh_d * fma(float2(-32.0f), tail_dots, head_dots);
        sumf += delta;
    }

    device float* out = Y + (p.y_offset / 4u);
    // The Zig dispatcher only routes row-multiple-of-four dense shapes to this
    // llama-style Q6_K path, so the NR0=2 row pair is always in bounds.
    const float total0 = simd_sum(sumf.x);
    const float total1 = simd_sum(sumf.y);
    if (tiisg == 0u) {
        out[first_row + 0u] = total0;
        out[first_row + 1u] = total1;
    }
}
