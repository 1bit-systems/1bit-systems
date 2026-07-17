#include <metal_stdlib>
using namespace metal;

// Dense Gemma single-token gate/up Q4_K matvec fused with GeGLU.
//
// The row layout follows the reference implementation's kernel_mul_mv_q4_K_f32:
// 64 threads per threadgroup, 2 simdgroups, 2 rows per simdgroup.
// Unlike the older K=5376-specialized kernels, this does not stage the input
// vector in threadgroup memory. It only fuses the two same-shape projections
// and the activation so the input lanes are loaded once for gate+up.

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

inline float q4k_block_dot(
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
    constexpr float4 sc_pos_scale = float4(1.f, 1.f / 16.f, 1.f, 1.f / 16.f);
    const float4 sc_pos = float4(ushort4(sc16.x, sc16.x >> 8, sc16.z, sc16.z >> 8) & lo_mask) * sc_pos_scale;
    const float4 sc_neg = float4(ushort4(sc16.y, sc16.y >> 8, sc16.w, sc16.w >> 8) & lo_mask);
    return fma(float(dh.x), dot(head_pair, sc_pos), -float(dh.y) * dot(sumy, sc_neg));
}

inline float2 q4k_block_dot_pair(
    device const uchar* block0,
    int row_bytes,
    thread const float4* yl4_arr,
    thread const float4* yh4_arr,
    float4 sumy,
    ushort iq,
    ushort ir
) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    device const uchar* block1 = block0 + row_bytes;

    const packed_uint4 hdr_0 = *((device const packed_uint4*)block0);
    const packed_uint4 hdr_1 = *((device const packed_uint4*)block1);
    const half2 dh_0 = as_type<half2>(hdr_0.x);
    const half2 dh_1 = as_type<half2>(hdr_1.x);
    const uint sc_shift = uint(iq) * 16u;

    device const ushort* q1_0 = (device const ushort*)(block0 + 16) + 16 * iq + 4 * ir;
    device const ushort* q1_1 = (device const ushort*)(block1 + 16) + 16 * iq + 4 * ir;

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

    const float2 dh_d = float2(float(dh_0.x), float(dh_1.x));
    const float2 dh_dmin = float2(float(dh_0.y), float(dh_1.y));
    const float2 head_dots = float2(dot(head_pair_0, sc_pos_0), dot(head_pair_1, sc_pos_1));
    const float2 tail_dots = float2(dot(sumy, sc_neg_0), dot(sumy, sc_neg_1));
    return fma(dh_d, head_dots, -dh_dmin * tail_dots);
}

inline float geglu(float gate, float up) {
    const float g3 = gate * gate * gate;
    float inner = 0.7978845608f * fma(0.044715f, g3, gate);
#if defined(ZINC_DENSE_GEGLU_UNCHECKED_FAST)
    const float tanh_inner = fast::tanh(inner);
#else
    // `fast::tanh` can return NaN for extreme inputs on Apple GPUs. Saturate
    // the GeLU tails instead of clamping every value before the tanh call.
    const float tanh_inner = inner > 15.0f ? 1.0f : (inner < -15.0f ? -1.0f : fast::tanh(inner));
#endif
    const float gelu_gate = 0.5f * gate * (1.0f + tanh_inner);
    return gelu_gate * up;
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

    const int nb = p.K / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int row_bytes = nb * BLOCK_SIZE;

    device const uchar* gate_src = W0 + p.a0_offset;
    device const uchar* up_src = W1 + p.a1_offset;
    device const float* x = X + (p.x_offset / 4);
    device float* out = activatedY + (p.y0_offset / 4);

    // Keep the two row accumulators in vector registers. The failed inline
    // row/projection interleave for this GeGLU shader was too register-heavy;
    // this preserves the helper boundary while avoiding tiny thread arrays.
    float2 gate_sum = float2(0.0f);
    float2 up_sum = float2(0.0f);

    device const float* y4 = x + ix * QK_K + 64 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy;

        // Cycle 81: port cycle 23's explicit float4 y-loads pattern from
        // dmmv_q4k_dense_gate_up_swiglu.metal and dmmv_q4k.metal. Replace
        // 32 scalar y4[i] reads inside an 8-iter FOR_UNROLL with 8 explicit
        // float4 loads of the four 8-float slices (offsets 0,32,128,160 from
        // y4 — all 32-byte aligned by construction of ix=tiisg/8, iq=it/4,
        // ir=it%4). The dot(slice, ones) chains fold the sumy partials into 4
        // fused vector reductions instead of 32 scalar +=. Matches the y-load
        // shape of the Qwen3 swiglu sibling kernel, modernizing the Gemma
        // GeGLU dense FFN path that's still using the legacy scalar form.
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
        sumy[0] = yl_tot.x + yl_tot.y;
        sumy[1] = yl_tot.z + yl_tot.w;
        sumy[2] = yh_tot.x + yh_tot.y;
        sumy[3] = yh_tot.z + yh_tot.w;

        // Keep the NR0=2 row pair interleaved inside each projection, matching
        // the base Q4_K row-pair accumulator without repeating the earlier
        // four-way gate/up x row interleave that raised register pressure.
        const ulong row_off = ulong(first_row) * ulong(row_bytes) + ulong(ib) * BLOCK_SIZE;
        gate_sum += q4k_block_dot_pair(gate_src + row_off, row_bytes, yl4_arr, yh4_arr, sumy, iq, ir);
        up_sum += q4k_block_dot_pair(up_src + row_off, row_bytes, yl4_arr, yh4_arr, sumy, iq, ir);

        y4 += 4 * QK_K;
    }

    const float gate_total0 = simd_sum(gate_sum.x);
    const float gate_total1 = simd_sum(gate_sum.y);
    const float up_total0 = simd_sum(up_sum.x);
    const float up_total1 = simd_sum(up_sum.y);
    if (tiisg == 0) {
        out[first_row + 0] = geglu(gate_total0, up_total0);
        out[first_row + 1] = geglu(gate_total1, up_total1);
    }
}
