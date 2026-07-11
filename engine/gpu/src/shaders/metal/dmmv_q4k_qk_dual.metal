#include <metal_stdlib>
using namespace metal;

// Dense Gemma single-token Q+K dual Q4_K matvec.
//
// Pairs the attention Q and K projections (same K dimension, both Q4_K, but
// distinct M_q and M_k) into a single dispatch. Adapts the reference implementation's
// kernel_mul_mv_q4_K_f32 row layout (NSG=2, NR0=2, 64 threads). Unlike the
// existing dmmv_q4k_dual_llama kernel — which dispatches the gate/up pair
// over grid.y={0,1} with grid.x = max(M_gate, M_up) and so wastes
// threadgroups when M0 != M1 — this kernel uses a single-axis row layout
// where rows 0..M_q-1 select the Q weight/output and rows M_q..M_q+M_k-1
// select the K weight/output. No threadgroup is launched for rows past
// M_q + M_k, so dense Gemma 31B (M_q=8192, M_k=4096) sees the same total
// threadgroups (3072) as the two separate single-projection dispatches —
// just one dispatch instead of two.
//
// Requires p.M_q % (NSG * NR0) == 0 so each threadgroup is wholly Q or
// wholly K (the dispatcher checks this; otherwise it falls back to the two
// single-projection dispatches).

struct QKDualPush {
    uint M_q;
    uint M_k;
    uint K;
    uint a_q_offset;
    uint a_k_offset;
    uint x_offset;
    uint y_q_offset;
    uint y_k_offset;
};

#define NSG 2
#define NR0 2
#define QK_K 256
#define BLOCK_SIZE 144
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Cycle 64: return per-row partials (head_dot, tail_dot, d, dmin) instead
// of a final scalar so the caller can fold the 2 cross-row
// `d*head - dmin*tail` chains into a single vec2 fma + vec2 mul. Mirrors
// cycle 61's 2-wide cross-row pattern on dmmv_q4k.metal — the qk_dual
// kernel's helper was hiding the cross-row vectorization opportunity by
// finalizing each row's reduction to scalar inside the helper.
inline float4 q4k_block_dot_parts(
    device const uchar* block,
    thread const float4* yl4_arr,
    thread const float4* yh4_arr,
    float4 sumy,
    ushort iq,
    ushort ir
) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    // Cycle 78: fuse the 4-byte half2 dh-load (block offset 0..3) and the
    // 12-byte packed_uint3 sc_u-load (block offset 4..15) into a single
    // 16-byte `packed_uint4` block-header load. The Q4_K block layout
    // places `[d (half), dmin (half), sc_u (12 bytes)]` contiguously at
    // offsets 0..15 with 4-byte alignment, exactly matching packed_uint4
    // (16 bytes, 4-byte aligned). Replaces 2 device loads (one 4-byte
    // half2 + one 12-byte uint3) with one 16-byte coalesced load. The
    // half2 dh value is held across ~30 lines until the final return,
    // but it's only 4 bytes of register state — negligible. Builds on
    // cycle 68 (half2 dh-load) and cycle 75 (packed_uint3 sc_u-load)
    // by collapsing them into the natural single-block-header read shape.
    // The helper is called twice per ib (once per NR0=2 row), so per-ib
    // this folds 4 loads → 2 packed_uint4 loads. dmmv_q4k_qk_dual.metal
    // handles Qwen3-8B attn_qkv Q+K paired (~18.5% of decode bytes/token
    // via the 25.08 GiB attn path).
    const packed_uint4 hdr = *((device const packed_uint4*)block);
    const half2 dh = as_type<half2>(hdr.x);
    const uint sc_shift = uint(iq) * 16u;
    device const ushort* q1 = (device const ushort*)(block + 16) + 16 * iq + 4 * ir;

    // Cycle 71: port cycle 69's ushort4-sc16 register-vector pattern to
    // qk_dual. Replace the stack-allocated `ushort sc16[4]` + uchar*
    // alias with a register-resident `ushort4`; the previous form
    // forced the compiler to materialize sc16 in thread-private memory
    // so the byte alias could address individual lanes. The new form
    // keeps the four packed scales in SSA-eligible registers and lowers
    // the per-ib sc_pos / sc_neg byte gathers to vector AND + vector
    // shift over packed lanes. The helper is called twice per ib (once
    // per row), so per-ib this folds 8 scalar uchar loads → 4 packed
    // ushort4 byte-extractions. Same compiler-hint philosophy as cycles
    // 49/50 (nibble-mask vectorize), 51 (per-ib reduction), 61
    // (cross-row reduction), 69 (dmmv_q4k.metal), 70 (swiglu).
    const uint3 sc_u3v = uint3(hdr.y, hdr.z, hdr.w);
    const ushort sc_0 = ushort((sc_u3v.x >> sc_shift) & 0xFFFFu);
    const ushort sc_2 = ushort((sc_u3v.y >> sc_shift) & 0xFFFFu);
    const ushort sc_4 = ushort((sc_u3v.z >> sc_shift) & 0xFFFFu);
    const ushort4 sc16 = ushort4(
        sc_0 & kmask1,
        sc_2 & kmask1,
        ((sc_4 >> 0) & kmask2) | ((sc_0 & kmask3) >> 2),
        ((sc_4 >> 4) & kmask2) | ((sc_2 & kmask3) >> 2));

    const ushort4 q1v = *((device const ushort4*)q1);
    const ushort4 q2v = *((device const ushort4*)(q1 + 32));

    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

    FOR_UNROLL (short i = 0; i < 4; ++i) {
        const float4 yl4 = yl4_arr[i];
        const float4 yh4 = yh4_arr[i];
        const ushort q1i = q1v[i];
        const ushort q2i = q2v[i];
        const float4 q1m = float4(q1i & 0x000F, q1i & 0x0F00, q1i & 0x00F0, q1i & 0xF000);
        const float4 q2m = float4(q2i & 0x000F, q2i & 0x0F00, q2i & 0x00F0, q2i & 0xF000);
        acc1 = fma(yl4, q1m, acc1);
        acc2 = fma(yh4, q2m, acc2);
    }

    const float4 head_pair = fma(
        float4(acc1[1], acc1[3], acc2[1], acc2[3]),
        float4(1.f / 256.f),
        float4(acc1[0], acc1[2], acc2[0], acc2[2]));
    // Cycle 71: derive sc_pos / sc_neg via vector byte-extraction from
    // the ushort4 sc16. sc8[0..7] (the old uchar* alias) maps to
    // {sc16.x.lo, sc16.x.hi, sc16.y.lo, sc16.y.hi, sc16.z.lo, sc16.z.hi,
    // sc16.w.lo, sc16.w.hi}, so:
    //   sc_pos = (sc16.x.lo, sc16.x.hi/16, sc16.z.lo, sc16.z.hi/16)
    //   sc_neg = (sc16.y.lo, sc16.y.hi,    sc16.w.lo, sc16.w.hi)
    constexpr ushort4 lo_mask = ushort4(0x00FFu);
    const ushort4 sc_pos_bytes = ushort4(sc16.x, sc16.x >> 8, sc16.z, sc16.z >> 8) & lo_mask;
    constexpr float4 sc_pos_scale = float4(1.f, 1.f / 16.f, 1.f, 1.f / 16.f);
    const float4 sc_pos = float4(sc_pos_bytes) * sc_pos_scale;
    const float4 sc_neg = float4(ushort4(sc16.y, sc16.y >> 8, sc16.w, sc16.w >> 8) & lo_mask);
    return float4(dot(head_pair, sc_pos), dot(sumy, sc_neg), float(dh.x), float(dh.y));
}

kernel void main0(
    device const uchar* W_q [[buffer(0)]],
    device const uchar* W_k [[buffer(1)]],
    constant QKDualPush& p [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* Y_q [[buffer(4)]],
    device float* Y_k [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int nb = int(p.K) / QK_K;
    const int linear_row = (int(tgpig.x) * NSG + sgitg) * NR0;
    const int row_bytes = nb * BLOCK_SIZE;

    // Decide projection from the simdgroup's first row. Caller guarantees
    // M_q is a multiple of NSG*NR0 so a TG never straddles the boundary.
    const bool is_k = linear_row >= int(p.M_q);
    device const uchar* src = is_k ? (W_k + p.a_k_offset) : (W_q + p.a_q_offset);
    device float* out = is_k ? (Y_k + (p.y_k_offset / 4)) : (Y_q + (p.y_q_offset / 4));
    const int dst_row_base = is_k ? (linear_row - int(p.M_q)) : linear_row;
    const int M = is_k ? int(p.M_k) : int(p.M_q);

    device const float* x = X + (p.x_offset / 4);

    float sumf[NR0] = {0.f, 0.f};

    device const float* y4 = x + ix * QK_K + 64 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy;

        // Cycle 57: port the explicit float4 y-load + dot4 sumy reduction
        // pattern from dmmv_q4k.metal. The previous scalar `sumy[k] += yl[i]`
        // accumulator chain inside the i=0..7 FOR_UNROLL had per-lane writes
        // into a float4 indexed by a compile-time constant — exactly the
        // shape cycles 44-51 found the Apple7 compiler does not always lift
        // to a single vector op. Replacing with 8×16-byte coalesced float4
        // loads of the four 8-float slices (offsets 0,32,128,160 — all
        // 32-byte aligned by construction of ix,iq,ir) folds the sumy
        // partials into 4 fused dot(v, 1) chains per ib instead of 32
        // serial scalar adds. y4 alignment: base = x + ix*QK_K + 64*iq +
        // 8*ir, all 32-byte boundaries. Mirrors cycle 23's win on
        // dmmv_q4k.metal. This kernel handles Qwen3-8B attn_qkv (Q+K
        // paired) — 25.08 GiB/step attn path = 18.5% of decode bytes/token.
        const device float4* y4v = (const device float4*)y4;
        const float4 a0 = y4v[0];   const float4 a1 = y4v[1];
        const float4 b0 = y4v[8];   const float4 b1 = y4v[9];
        const float4 c0 = y4v[32];  const float4 c1 = y4v[33];
        const float4 d0 = y4v[40];  const float4 d1 = y4v[41];

        // Cycle 86: store the per-i y-gather directly as `float4 yl4_arr[4]`
        // / `float4 yh4_arr[4]` register-vector arrays instead of the legacy
        // `float yl[16]` / `float yh[16]` scalar layout. The helper's FMA
        // loop consumes exactly one float4 per i from each side — yl4_arr[i]
        // = (yl[2i], yl[2i+1], yl[2i+8], yl[2i+9]) — which under the old
        // layout was reconstructed each iteration inside the helper via a
        // 4-lane indexed gather off 16-wide flat stack arrays filled here
        // via 32 scalar lane writes. Eliminates 32 scalar stack writes and
        // 8 4-lane gather reconstructions per ib (the helper runs twice per
        // ib for NR0=2 rows), keeping all y data in SSA-eligible vector
        // registers. The lane mapping is yl4_arr[i] = (a*.xy, b*.xy) for
        // i∈{0,2} and (a*.zw, b*.zw) for i∈{1,3} where a*/b* picks between
        // (a0,b0)/(a1,b1) based on i/2. Mirrors cycle 85's port to
        // dmmv_q4k.metal, cycle 84's port to dmmv_q4k_dense_gate_up_swiglu
        // .metal (hottest Q4_K shader, ffn_gate+ffn_up), and cycle 83's
        // original in dmmv_q6k_llama.metal. dmmv_q4k_qk_dual.metal pairs
        // Qwen3-8B attn_qkv Q+K (~18.5% of decode bytes/token via the
        // 25.08 GiB attn path; Q4_K = 71.6% of decode bytes/token).
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

        // Cycle 89: derive `sumy` from the `yl4_arr` / `yh4_arr` register
        // vectors built above instead of as a parallel second consumer of
        // the original a0/a1/b0/b1/c0/c1/d0/d1 float4 loads. The legacy
        // `sumy[k] = dot(a*, ones) + dot(a1, ones)` per slice forced the
        // compiler to keep all 8 raw load vectors live until both the
        // sumy reduction and the yl4_arr/yh4_arr build had completed; by
        // reducing through yl4_arr/yh4_arr (the same vectors the FMA loop
        // already consumes) a0..d1 can die immediately after the lane
        // shuffle, freeing ~128B of register file before the q1v/q2v
        // ushort4 loads + 32-FMA inner chain. Lane mapping (per yl4_arr
        // build above): yl4_arr[i].xy = a-family, yl4_arr[i].zw =
        // b-family; same with yh4_arr for c/d. Summing all 4 yl4_arr
        // gives yl_tot where (yl_tot.x+yl_tot.y) = sum(a0)+sum(a1) =
        // sumy[0] and (yl_tot.z+yl_tot.w) = sumy[1]; analogous for
        // yh_tot → sumy[2..3]. Cost equivalent (~28 effective adds
        // either way); pure register-pressure optimization. Mirrors
        // cycle 87 (swiglu) and cycle 88 (dmmv_q4k.metal). qk_dual serves
        // Qwen3-8B attn_qkv Q+K paired (~18.5% of decode bytes/token via
        // the 25.08 GiB attn path; Q4_K = 71.6% of decode bytes/token).
        const float4 yl_tot = (yl4_arr[0] + yl4_arr[1]) + (yl4_arr[2] + yl4_arr[3]);
        const float4 yh_tot = (yh4_arr[0] + yh4_arr[1]) + (yh4_arr[2] + yh4_arr[3]);
        sumy[0] = yl_tot.x + yl_tot.y;
        sumy[1] = yl_tot.z + yl_tot.w;
        sumy[2] = yh_tot.x + yh_tot.y;
        sumy[3] = yh_tot.z + yh_tot.w;

        // Cycle 64: cross-row 2-wide reduction. Call the parts helper for
        // both rows so head_dot/tail_dot/d/dmin from both rows are in scope
        // together, then fold the 2 scalar `d*head - dmin*tail` chains into
        // one vec2 fma + one vec2 mul producing a float2 `delta` of the 2
        // per-row increments. Mirrors cycle 61's pattern on dmmv_q4k.metal.
        // dmmv_q4k_qk_dual.metal serves Qwen3-8B attn_qkv (Q+K paired into
        // one dispatch). Dispatcher guarantees M_q % (NSG*NR0)==0, and for
        // Qwen3-8B both M_q=4096 and M_k=1024 are divisible by NSG*NR0=4
        // so both rows are always valid when this kernel fires; the per-
        // row `dst_row < M` guard is preserved by zeroing the parts when
        // a row is past M so the cross-row fold contributes 0 to sumf for
        // that row, matching the original per-row skip.
        const int dst_row_0 = dst_row_base + 0;
        const int dst_row_1 = dst_row_base + 1;
        const ulong row_off_0 = ulong(dst_row_0) * ulong(row_bytes) + ulong(ib) * BLOCK_SIZE;
        const ulong row_off_1 = ulong(dst_row_1) * ulong(row_bytes) + ulong(ib) * BLOCK_SIZE;
        const float4 parts_0 = (dst_row_0 < M)
            ? q4k_block_dot_parts(src + row_off_0, yl4_arr, yh4_arr, sumy, iq, ir)
            : float4(0.0f);
        const float4 parts_1 = (dst_row_1 < M)
            ? q4k_block_dot_parts(src + row_off_1, yl4_arr, yh4_arr, sumy, iq, ir)
            : float4(0.0f);
        const float2 dh_d = float2(parts_0[2], parts_1[2]);
        const float2 dh_dmin = float2(parts_0[3], parts_1[3]);
        const float2 head_dots = float2(parts_0[0], parts_1[0]);
        const float2 tail_dots = float2(parts_0[1], parts_1[1]);
        const float2 delta = fma(dh_d, head_dots, -dh_dmin * tail_dots);
        sumf[0] += delta[0];
        sumf[1] += delta[1];

        y4 += 4 * QK_K;
    }

    for (short row = 0; row < NR0; ++row) {
        const int dst_row = dst_row_base + row;
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0) {
            if (dst_row < M) out[dst_row] = total;
        }
    }
}
