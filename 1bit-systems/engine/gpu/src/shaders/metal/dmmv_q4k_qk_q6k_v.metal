#include <metal_stdlib>
using namespace metal;

// Dense Gemma attention Q/K/V projection in one dispatch:
//   rows [0, M_q)             -> Q4_K Q
//   rows [M_q, M_q + M_k)     -> Q4_K K
//   rows [M_q + M_k, end)     -> Q6_K V
//
// This preserves the reference-style 64-thread / 4-row geometry from
// dmmv_q4k_qk_dual.metal and dmmv_q6k_llama.metal, but removes the separate
// V dispatch before the same QKV barrier on dense Gemma 31B decode.

struct QKVDensePush {
    uint M_q;
    uint M_k;
    uint M_v;
    uint K;
    uint a_q_offset;
    uint a_k_offset;
    uint a_v_offset;
    uint x_offset;
    uint y_q_offset;
    uint y_k_offset;
    uint y_v_offset;
};

#define NSG 2
#define NR0 2
#define QK_K 256
#define Q4_BLOCK_SIZE 144
#define Q6_BLOCK_SIZE 210
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

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

    const packed_uint4 hdr = *((device const packed_uint4*)block);
    const half2 dh = as_type<half2>(hdr.x);
    const uint sc_shift = uint(iq) * 16u;
    device const ushort* q1 = (device const ushort*)(block + 16) + 16 * iq + 4 * ir;

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
    constexpr ushort4 nibble_mask = ushort4(0x000F, 0x0F00, 0x00F0, 0xF000);

    FOR_UNROLL (short i = 0; i < 4; ++i) {
        const float4 yl4 = yl4_arr[i];
        const float4 yh4 = yh4_arr[i];
        const ushort q1i = q1v[i];
        const ushort q2i = q2v[i];
        const float4 q1m = float4(ushort4(q1i) & nibble_mask);
        const float4 q2m = float4(ushort4(q2i) & nibble_mask);
        acc1 = fma(yl4, q1m, acc1);
        acc2 = fma(yh4, q2m, acc2);
    }

    const float4 head_pair = fma(
        float4(acc1[1], acc1[3], acc2[1], acc2[3]),
        float4(1.f / 256.f),
        float4(acc1[0], acc1[2], acc2[0], acc2[2]));
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
    device const uchar* W_v [[buffer(2)]],
    constant QKVDensePush& p [[buffer(3)]],
    device const float* X [[buffer(4)]],
    device float* Y_q [[buffer(5)]],
    device float* Y_k [[buffer(6)]],
    device float* Y_v [[buffer(7)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int first_linear_row = (int(tgpig.x) * NSG + int(sgitg)) * NR0;
    const int qk_rows = int(p.M_q + p.M_k);

    if (first_linear_row < qk_rows) {
        const short ix = tiisg / 8;
        const short it = tiisg % 8;
        const short iq = it / 4;
        const short ir = it % 4;

        const int nb = int(p.K) / QK_K;
        const int row_bytes = nb * Q4_BLOCK_SIZE;
        const bool is_k = first_linear_row >= int(p.M_q);
        device const uchar* src = is_k ? (W_k + p.a_k_offset) : (W_q + p.a_q_offset);
        device float* out = is_k ? (Y_k + (p.y_k_offset / 4)) : (Y_q + (p.y_q_offset / 4));
        const int dst_row_base = is_k ? (first_linear_row - int(p.M_q)) : first_linear_row;

        device const float* x = X + (p.x_offset / 4);
        // Dense Gemma dispatches Q/K row counts as multiples of NSG*NR0, so
        // keep the reference implementation NR0=2 row pair in one vector accumulator instead
        // of a thread-private scalar array. This matches the Q6_K V half below
        // and gives the compiler the same 2-wide reduction shape for the Q4_K
        // Q/K half of the fused attention projection.
        float2 sumf = float2(0.0f);
        device const float* y4 = x + ix * QK_K + 64 * iq + 8 * ir;

        for (int ib = ix; ib < nb; ib += 4) {
            const device float4* y4v = (const device float4*)y4;
            const float4 a0 = y4v[0];   const float4 a1 = y4v[1];
            const float4 b0 = y4v[8];   const float4 b1 = y4v[9];
            const float4 c0 = y4v[32];  const float4 c1 = y4v[33];
            const float4 d0 = y4v[40];  const float4 d1 = y4v[41];

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

            const float4 yl_tot = (yl4_arr[0] + yl4_arr[1]) + (yl4_arr[2] + yl4_arr[3]);
            const float4 yh_tot = (yh4_arr[0] + yh4_arr[1]) + (yh4_arr[2] + yh4_arr[3]);
            float4 sumy;
            sumy[0] = yl_tot.x + yl_tot.y;
            sumy[1] = yl_tot.z + yl_tot.w;
            sumy[2] = yh_tot.x + yh_tot.y;
            sumy[3] = yh_tot.z + yh_tot.w;

            // Match the base dmmv_q4k row-interleaved body: load both adjacent
            // rows' headers/quants, then advance their accumulators together.
            // The dispatch guard requires M_q/M_k multiples of 4, so an NR0=2
            // simdgroup never straddles Q/K or a tail row.
            device const uchar* blk_0 = src + ulong(dst_row_base) * ulong(row_bytes) + ulong(ib) * Q4_BLOCK_SIZE;
            device const uchar* blk_1 = blk_0 + row_bytes;
            const packed_uint4 hdr_0 = *((device const packed_uint4*)blk_0);
            const packed_uint4 hdr_1 = *((device const packed_uint4*)blk_1);
            const half2 dh_pair_0 = as_type<half2>(hdr_0.x);
            const half2 dh_pair_1 = as_type<half2>(hdr_1.x);
            const uint sc_shift = uint(iq) * 16u;

            constexpr ushort kmask1 = 0x3f3f;
            constexpr ushort kmask2 = 0x0f0f;
            constexpr ushort kmask3 = 0xc0c0;
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

            device const ushort* q1_0 = (device const ushort*)(blk_0 + 16) + 16 * iq + 4 * ir;
            device const ushort* q1_1 = (device const ushort*)(blk_1 + 16) + 16 * iq + 4 * ir;
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
                const float4 yl4 = yl4_arr[i];
                const float4 yh4 = yh4_arr[i];
                const ushort q1_0i = q1v_0[i];
                const ushort q1_1i = q1v_1[i];
                const ushort q2_0i = q2v_0[i];
                const ushort q2_1i = q2v_1[i];
                const float4 q1m_0 = float4(ushort4(q1_0i) & nibble_mask);
                const float4 q1m_1 = float4(ushort4(q1_1i) & nibble_mask);
                const float4 q2m_0 = float4(ushort4(q2_0i) & nibble_mask);
                const float4 q2m_1 = float4(ushort4(q2_1i) & nibble_mask);
                acc1_0 = fma(yl4, q1m_0, acc1_0);
                acc1_1 = fma(yl4, q1m_1, acc1_1);
                acc2_0 = fma(yh4, q2m_0, acc2_0);
                acc2_1 = fma(yh4, q2m_1, acc2_1);
            }

            const float4 head_pair_0 = fma(
                float4(acc1_0[1], acc1_0[3], acc2_0[1], acc2_0[3]),
                float4(1.f / 256.f),
                float4(acc1_0[0], acc1_0[2], acc2_0[0], acc2_0[2]));
            const float4 head_pair_1 = fma(
                float4(acc1_1[1], acc1_1[3], acc2_1[1], acc2_1[3]),
                float4(1.f / 256.f),
                float4(acc1_1[0], acc1_1[2], acc2_1[0], acc2_1[2]));
            constexpr ushort4 lo_mask = ushort4(0x00FFu);
            constexpr float4 sc_pos_scale = float4(1.f, 1.f / 16.f, 1.f, 1.f / 16.f);
            const float4 sc_pos_0 = float4(ushort4(sc16_0.x, sc16_0.x >> 8, sc16_0.z, sc16_0.z >> 8) & lo_mask) * sc_pos_scale;
            const float4 sc_pos_1 = float4(ushort4(sc16_1.x, sc16_1.x >> 8, sc16_1.z, sc16_1.z >> 8) & lo_mask) * sc_pos_scale;
            const float4 sc_neg_0 = float4(ushort4(sc16_0.y, sc16_0.y >> 8, sc16_0.w, sc16_0.w >> 8) & lo_mask);
            const float4 sc_neg_1 = float4(ushort4(sc16_1.y, sc16_1.y >> 8, sc16_1.w, sc16_1.w >> 8) & lo_mask);
            const float2 dh_d = float2(float(dh_pair_0.x), float(dh_pair_1.x));
            const float2 dh_dmin = float2(float(dh_pair_0.y), float(dh_pair_1.y));
            const float2 head_dots = float2(dot(head_pair_0, sc_pos_0), dot(head_pair_1, sc_pos_1));
            const float2 tail_dots = float2(dot(sumy, sc_neg_0), dot(sumy, sc_neg_1));
            const float2 delta = fma(dh_d, head_dots, -dh_dmin * tail_dots);
            sumf += delta;

            y4 += 4 * QK_K;
        }

        const float total0 = simd_sum(sumf.x);
        const float total1 = simd_sum(sumf.y);
        if (tiisg == 0) {
            out[dst_row_base + 0] = total0;
            out[dst_row_base + 1] = total1;
        }
        return;
    }

    const uint first_v_row = uint(first_linear_row - qk_rows);
    const uint nb = p.K / QK_K;
    const uint row_bytes = nb * Q6_BLOCK_SIZE;
    device const uchar* src0 = W_v + p.a_v_offset;
    device const float* src1 = X + (p.x_offset / 4u);

    const ushort tid = tiisg / 2u;
    const ushort ix = tiisg % 2u;
    const ushort ip = tid / 8u;
    const ushort il = tid % 8u;
    const ushort l0 = 4u * il;
    const ushort is = 8u * ip + l0 / 16u;

    const uint y_offset = 128u * uint(ip) + uint(l0);
    const uint q_offset_l = 64u * uint(ip) + uint(l0);
    const uint q_offset_h = 32u * uint(ip) + uint(l0);

    // Keep the Q6_K NR0=2 row pair as a vector accumulator, matching the
    // standalone dmmv_q6k_llama path this fused Q/K/V shader replaces.
    float2 sumf = float2(0.0f);

    for (uint bi = ix; bi < nb; bi += 2u) {
        device const float* y = src1 + bi * QK_K + y_offset;

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
        const float4 yl_sum4 = (yl4_arr[0] + yl4_arr[1]) + (yl4_arr[2] + yl4_arr[3]);

        device const uchar* block0 = src0 + ulong(first_v_row) * ulong(row_bytes) + ulong(bi) * Q6_BLOCK_SIZE;
        device const uchar* block1 = block0 + row_bytes;

        const uchar4 q1v4_0 = *((device const uchar4*)(block0 + q_offset_l));
        const uchar4 q2v4_0 = *((device const uchar4*)(block0 + q_offset_l + 32u));
        const uchar4 qhv4_0 = *((device const uchar4*)(block0 + 128u + q_offset_h));
        device const char* sc_0 = (device const char*)(block0 + 192u + uint(is));
        const float d_0 = float(*((device const half*)(block0 + 208)));

        const uchar4 q1v4_1 = *((device const uchar4*)(block1 + q_offset_l));
        const uchar4 q2v4_1 = *((device const uchar4*)(block1 + q_offset_l + 32u));
        const uchar4 qhv4_1 = *((device const uchar4*)(block1 + 128u + q_offset_h));
        device const char* sc_1 = (device const char*)(block1 + 192u + uint(is));
        const float d_1 = float(*((device const half*)(block1 + 208)));

        float4 sums_0 = float4(0.0f);
        float4 sums_1 = float4(0.0f);
        FOR_UNROLL (ushort l = 0u; l < 4u; ++l) {
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

            const float4 yl4 = yl4_arr[l];
            const float4 q4_0 = float4(q_base_0 | h_part_0);
            const float4 q4_1 = float4(q_base_1 | h_part_1);
            sums_0 = fma(yl4, q4_0, sums_0);
            sums_1 = fma(yl4, q4_1, sums_1);
        }

        const packed_char4 sc_lo_0 = *((device const packed_char4*)sc_0);
        const packed_char4 sc_hi_0 = *((device const packed_char4*)(sc_0 + 4));
        const packed_char4 sc_lo_1 = *((device const packed_char4*)sc_1);
        const packed_char4 sc_hi_1 = *((device const packed_char4*)(sc_1 + 4));
        const float4 sc4_0 = float4(float(sc_lo_0.x), float(sc_lo_0.z), float(sc_hi_0.x), float(sc_hi_0.z));
        const float4 sc4_1 = float4(float(sc_lo_1.x), float(sc_lo_1.z), float(sc_hi_1.x), float(sc_hi_1.z));
        const float2 dh_d = float2(d_0, d_1);
        const float2 head_dots = float2(dot(sums_0, sc4_0), dot(sums_1, sc4_1));
        const float2 tail_dots = float2(dot(yl_sum4, sc4_0), dot(yl_sum4, sc4_1));
        const float2 delta = dh_d * fma(float2(-32.0f), tail_dots, head_dots);
        sumf += delta;
    }

    device float* out = Y_v + (p.y_v_offset / 4u);
    const float total0 = simd_sum(sumf.x);
    const float total1 = simd_sum(sumf.y);
    if (tiisg == 0u) {
        out[first_v_row + 0u] = total0;
        out[first_v_row + 1u] = total1;
    }
}
