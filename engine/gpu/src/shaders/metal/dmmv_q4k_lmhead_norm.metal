#include <metal_stdlib>
using namespace metal;

// Fused RMSNorm + Q4_K matvec for the dense Gemma 31B final LM head.
//
// Adapts the reference implementation `kernel_mul_mv_q4_K_f32` row layout
// (NSG=2, NR0=2, 64 threads/threadgroup) used by dmmv_q4k.metal,
// folding the preceding `rms_norm_mul` (final_norm) into the same
// dispatch — eliminating one dispatch + one barrier in the final phase
// per decode step. Each simdgroup independently computes the RMS
// reciprocal over the unnormalized hidden vector via simd_sum, then
// multiplies its X loads by `norm_weight[idx] * rms_inv` on the fly,
// matching the per-simdgroup-redundant-rms pattern used by
// dmmv_q8_0_dual_fused_norm.metal.

struct DmmvNormPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
    float eps;
};

#define NSG   2
#define NR0   2
#define QK_K  256
#define BLOCK_SIZE 144
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant DmmvNormPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    device const float* NW [[buffer(4)]],
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

    const int nb = p.K / QK_K;
    const int r0 = tgpig.x;
    const int first_row = (r0 * NSG + sgitg) * NR0;
    const int nb01 = nb * BLOCK_SIZE;

    device const uchar* src0 = W + p.a_offset;
    device const float* src1 = X + (p.x_offset / 4);

    // Step 1: RMS reduction over the raw hidden vector. Each simdgroup
    // independently reads strided X values, squares them, and reduces
    // via simd_sum. Cost per simdgroup is K/32 reads; with 32 threads
    // per simdgroup this fully covers K. The X buffer (5376 floats =
    // 21 KB on Gemma 31B) is L2-cache-resident across the whole LM
    // head dispatch, so the redundant per-simdgroup reads are cheap.
    float sq_sum = 0.0f;
    for (uint i = tiisg; i < p.K; i += 32u) {
        const float v = src1[i];
        sq_sum += v * v;
    }
    sq_sum = simd_sum(sq_sum);
    const float rms_inv = rsqrt(sq_sum / float(p.K) + p.eps);

    // Step 2: Q4_K matvec, applying `norm_weight * rms_inv` to each X
    // chunk on load. Output offsets within a 256-element block are
    // {0..7, 32..39, 128..135, 160..167} per the reference implementation layout, so
    // norm_weight is read at the matching offsets.
    device const uchar* x_base = src0 + (uint64_t)first_row * nb01;
    device const float* y = src1;
    device const float* nw = NW;

    float yl[16];
    float yh[16];

    float sumf[NR0] = {0.f, 0.f};

    const int x_lane_off = ix * QK_K + 64 * iq + 8 * ir;
    device const float* y4 = y + x_lane_off;
    device const float* nw4 = nw + x_lane_off;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        FOR_UNROLL (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0] * (nw4[i +   0] * rms_inv); sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32] * (nw4[i +  32] * rms_inv); sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128] * (nw4[i + 128] * rms_inv); sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160] * (nw4[i + 160] * rms_inv); sumy[3] += yh[i + 8];
        }

        device const ushort* sc = (device const ushort*)(x_base + (uint64_t)ib * BLOCK_SIZE + 4) + iq;
        device const ushort* q1 = (device const ushort*)(x_base + (uint64_t)ib * BLOCK_SIZE + 16) + 16 * iq + 4 * ir;
        device const half* dh = (device const half*)(x_base + (uint64_t)ib * BLOCK_SIZE);

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            // Cycle 72: port cycle 69's ushort4-sc16 register-vector pattern.
            // Replace `ushort sc16[4]` stack array + `(thread const uchar*)`
            // byte alias with a `ushort4` SSA-eligible register vector plus
            // precomputed lo/hi byte vectors via vector AND + vector shift.
            // The previous form forced the compiler to materialize sc16 in
            // thread-private memory so the byte alias could read individual
            // lanes; the new form keeps the four packed scales in registers
            // and lowers the 8 scalar uchar reads in the reduction body to
            // two vector byte-extractions. Same compiler-hint philosophy as
            // cycles 69 (dmmv_q4k.metal), 70 (swiglu), and 71 (qk_dual).
            // This kernel covers the fused final-norm + lm-head dispatch on
            // Qwen3-8B (vocab=151936, hidden_dim=4096, lm-head = 15.21 GiB/
            // step = ~11% of decode bytes/token). The cycle 54 attempt to
            // vectorize the full per-ib reduction here was reverted due to
            // register pressure from the per-simdgroup RMS prologue, so we
            // keep the scalar reduction shape — this change only touches
            // storage, not the reduction arithmetic.
            const ushort4 sc16 = ushort4(
                sc[0] & kmask1,
                sc[2] & kmask1,
                ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2),
                ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2));
            constexpr ushort4 lo_mask = ushort4(0x00FFu);
            const ushort4 sc8_lo = sc16 & lo_mask;
            const ushort4 sc8_hi = (sc16 >> 8) & lo_mask;

            // Cycle 48: port cycle 45's explicit-float4-fma pattern from
            // dmmv_q4k.metal to this lmhead_norm variant. Replace 8 indexed
            // scalar `accX[k] += y* * (q* & mask)` writes per `i` iteration
            // with 2 explicit float4 `fma(y4, q4_mask, acc)` calls. Packs
            // q1/q2 into ushort4 vector loads up front (q1[0..3] and q2[0..3]
            // are 4 contiguous ushorts at row-aligned addresses) and yl/yh
            // lanes into float4 vectors once per i, exposing the 4-wide SIMD
            // operation directly to the metal compiler. Mirrors the same
            // change in dmmv_q4k.metal (cycle 45), dmmv_q4k_dense_gate_up_
            // swiglu.metal (cycle 46) and dmmv_q4k_qk_dual.metal (cycle 47).
            // Used by the fused final-norm+LM-head dispatch on Gemma 31B.
            const ushort4 q1v = *((device const ushort4*)q1);
            const ushort4 q2v = *((device const ushort4*)(q1 + 32));

            // Cycle 77: port cycles 66/67/68's half2 dh-load pattern.
            // Q4_K block layout starts with `[d (half), dmin (half)]` at
            // byte offsets 0..3, exactly a half2 pair matching the natural
            // 4-byte block alignment. Replace the 2 scalar half reads
            // (`dh[0]` = d, `dh[1]` = dmin) in the reduction below with
            // a single packed half2 load + .x/.y swizzles that lower to
            // one half2→float2 widen per row instead of 2 scalar half→
            // float casts. Mirrors cycle 67 (dmmv_q4k.metal, 4→2 loads
            // per ib) and cycle 68 (dmmv_q4k_qk_dual.metal). The
            // lmhead_norm kernel keeps the scalar reduction shape (cycle
            // 54's full per-ib vectorization regressed here due to
            // register pressure from the per-simdgroup RMS prologue),
            // but a half2 load on its own does not add register pressure
            // — it just fuses the existing pair of scalar half loads
            // into one 4-byte aligned packed read. Unlike cycle 76's
            // packed_uint3 attempt (3 new uint registers per row), this
            // change keeps the same 2 float values (d, dmin) in scope.
            // Used by the fused final-norm + lm-head dispatch on
            // Qwen3-8B (vocab=151936, hidden_dim=4096, lm-head =
            // 15.21 GiB/step = ~11% of decode bytes/token).
            const half2 dh_pair = *((device const half2*)dh);
            const float d_val = float(dh_pair.x);
            const float dmin_val = float(dh_pair.y);

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            // Cycle 63: vectorize the per-quant nibble-mask expansion. Replace
            // `float4(qi & 0x000F, qi & 0x0F00, qi & 0x00F0, qi & 0xF000)` —
            // 4 scalar ANDs + 4 scalar int→float casts per nibble-set — with
            // `float4(ushort4(qi) & nibble_mask)`, which lowers to 1 ushort4
            // splat, 1 ushort4 AND vs a constexpr mask, then 1 ushort4→float4
            // widen. Mirrors cycle 49 (dmmv_q4k_dense_gate_up_swiglu.metal) and
            // cycle 50 (dmmv_q4k.metal). The fused-norm LM-head kernel is on
            // the Qwen3-8B decode hot path (vocab=151936, hidden_dim=4096,
            // lm-head = 15.21 GiB/step = ~11% of decode bytes/token), and the
            // nibble-mask change is a local expression-level rewrite that does
            // not raise register pressure — unlike cycle 54's per-ib reduction
            // attempt which regressed on this kernel due to the per-simdgroup
            // RMS prologue's existing register footprint.
            constexpr ushort4 nibble_mask = ushort4(0x000F, 0x0F00, 0x00F0, 0xF000);

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                const float4 yl4 = float4(yl[2 * i + 0], yl[2 * i + 1], yl[2 * i + 8], yl[2 * i + 9]);
                const float4 yh4 = float4(yh[2 * i + 0], yh[2 * i + 1], yh[2 * i + 8], yh[2 * i + 9]);
                const ushort q1i = q1v[i];
                const ushort q2i = q2v[i];
                const float4 q1m = float4(ushort4(q1i) & nibble_mask);
                const float4 q2m = float4(ushort4(q2i) & nibble_mask);
                acc1 = fma(yl4, q1m, acc1);
                acc2 = fma(yh4, q2m, acc2);
            }

            sumf[row] += d_val * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8_lo.x +
                    (acc1[2] + 1.f / 256.f * acc1[3]) * sc8_hi.x * 1.f / 16.f +
                    (acc2[0] + 1.f / 256.f * acc2[1]) * sc8_lo.z +
                    (acc2[2] + 1.f / 256.f * acc2[3]) * sc8_hi.z * 1.f / 16.f) -
                dmin_val * (sumy[0] * sc8_lo.y + sumy[1] * sc8_hi.y + sumy[2] * sc8_lo.w + sumy[3] * sc8_hi.w);

            q1 += nb01 / 2;
            sc += nb01 / 2;
            dh += nb01 / 2;
        }

        y4 += 4 * QK_K;
        nw4 += 4 * QK_K;
    }

    device float* dst_f32 = Y + (p.y_offset / 4);

    for (int row = 0; row < NR0 && first_row + row < (int)p.M; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}
