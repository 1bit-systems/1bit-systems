#include <metal_stdlib>
using namespace metal;

struct DualQ8DmmvConv1dPush {
    uint M0;
    uint M1;
    uint K;
    uint a0_offset;
    uint a1_offset;
    uint x_offset;
    uint y0_offset;
    uint y1_offset;
    uint conv_channels;   // for SSM qkv path: 8192 (== M0)
    uint d_conv;          // hard-gated to 4 by the dispatch site
};

// Fused RMSNorm + Dual-output Q8_0 DMMV + SSM conv1d postlude.
//
// Folds the standalone SSM `ssm_conv1d_qwen_d4` dispatch + its `qkv`-buffer
// barrier into the SSM qkv+z fused-norm-dual: after each simdgroup's lane 0
// writes its 4 row results to (attn_out_buf, gate_buf), the same lane runs
// the per-channel conv1d (silu(dot(w, [s0,s1,s2,new]))) for each of those
// rows that lands in attn_out_buf (Y0), writing swiglu_buf and shifting the
// 3-element conv state in place. The conv1d state is per-channel and each
// simdgroup owns disjoint channel indices, so there is no cross-thread race.
//
// Mirrors the reference implementation `ggml_metal_op_concurrency_check/reset` single-consumer
// fusion (ggml-metal-ops.cpp:159, 175) and extends cycle-13/22/23/25's
// fusion discipline to the SSM qkv → conv1d edge — the only consumer of
// the qkv slice of `attn_out_buf` in the decode token-major path is the
// conv1d that immediately follows, so the qkv-only barrier becomes
// redundant once both ops live in the same dispatch. Saves one dispatch
// + one barrier per SSM layer per decode token (≈30/decode token on
// Qwen3.6-35B). The trailing barrier on `swiglu_buf + gate_buf + alpha/beta
// outputs` is still needed (delta-net reads swiglu_buf) but now subsumes
// what used to be two separate barriers.
//
// Layout invariants (held by the dispatch site):
//   - M0 == conv_channels (the qkv slice of fused-norm-dual goes through
//     conv1d; the z slice in M1 does not).
//   - M0 is a multiple of `simdgroups_per_tg * 4` so the qkv→z boundary
//     never splits a simdgroup (each simdgroup has either all 4 rows in
//     attn_out_buf or all 4 in gate_buf; `first0` is uniform within the
//     simdgroup).
//   - d_conv == 4 (matches `ssm_conv1d_qwen_d4.metal`'s float4 layout).
//   - input_offset implicitly 0 (decode path; the prefill path uses a
//     different conv input buffer and is not eligible for this fusion).

kernel void main0(
    constant DualQ8DmmvConv1dPush& p [[buffer(0)]],
    device const uchar* W0 [[buffer(1)]],
    device const uchar* W1 [[buffer(2)]],
    device const float* hidden [[buffer(3)]],
    device float* Y0 [[buffer(4)]],
    device float* Y1 [[buffer(5)]],
    device const float* norm_weight [[buffer(6)]],
    device const float* conv_kernel [[buffer(7)]],
    device float* conv_state [[buffer(8)]],
    device float* conv_out [[buffer(9)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    device const float* h = hidden + (p.x_offset >> 2);

    // Step 1: Compute RMS normalization factor from raw hidden state.
    float sq_sum = 0.0f;
    for (uint i = lane; i < p.K; i += 32u) {
        const float v = h[i];
        sq_sum += v * v;
    }
    sq_sum = simd_sum(sq_sum);
    const float rms_inv = rsqrt(sq_sum / float(p.K) + 1e-6f);

    const uint linear_row = (tg_id * simdgroups_per_tg + sg_idx) * 4u;
    const uint total_rows = p.M0 + p.M1;
    if (linear_row >= total_rows) return;

    const bool first0 = linear_row < p.M0;
    const uint row0 = first0 ? linear_row : (linear_row - p.M0);
    device const uchar* weights0 = first0 ? W0 : W1;
    device float* output0 = first0 ? (Y0 + (p.y0_offset >> 2)) : (Y1 + (p.y1_offset >> 2));
    const uint a_offset0 = first0 ? p.a0_offset : p.a1_offset;

    const bool has1 = linear_row + 1u < total_rows;
    const uint linear1 = linear_row + 1u;
    const bool first1 = has1 ? (linear1 < p.M0) : first0;
    const uint row1 = has1 ? (first1 ? linear1 : (linear1 - p.M0)) : row0;
    device const uchar* weights1 = has1 ? (first1 ? W0 : W1) : weights0;
    device float* output1 = has1 ? (first1 ? (Y0 + (p.y0_offset >> 2)) : (Y1 + (p.y1_offset >> 2))) : output0;
    const uint a_offset1 = has1 ? (first1 ? p.a0_offset : p.a1_offset) : a_offset0;

    const bool has2 = linear_row + 2u < total_rows;
    const uint linear2 = linear_row + 2u;
    const bool first2 = has2 ? (linear2 < p.M0) : first0;
    const uint row2 = has2 ? (first2 ? linear2 : (linear2 - p.M0)) : row0;
    device const uchar* weights2 = has2 ? (first2 ? W0 : W1) : weights0;
    device float* output2 = has2 ? (first2 ? (Y0 + (p.y0_offset >> 2)) : (Y1 + (p.y1_offset >> 2))) : output0;
    const uint a_offset2 = has2 ? (first2 ? p.a0_offset : p.a1_offset) : a_offset0;

    const bool has3 = linear_row + 3u < total_rows;
    const uint linear3 = linear_row + 3u;
    const bool first3 = has3 ? (linear3 < p.M0) : first0;
    const uint row3 = has3 ? (first3 ? linear3 : (linear3 - p.M0)) : row0;
    device const uchar* weights3 = has3 ? (first3 ? W0 : W1) : weights0;
    device float* output3 = has3 ? (first3 ? (Y0 + (p.y0_offset >> 2)) : (Y1 + (p.y1_offset >> 2))) : output0;
    const uint a_offset3 = has3 ? (first3 ? p.a0_offset : p.a1_offset) : a_offset0;

    // Step 2: DMMV with inline-normalized input.
    const uint blocks_per_row = p.K >> 5;
    const ulong row_bytes = ulong(blocks_per_row) * 34ull;
    device const uchar* row_ptr0 = weights0 + a_offset0 + ulong(row0) * row_bytes;
    device const uchar* row_ptr1 = weights1 + a_offset1 + ulong(row1) * row_bytes;
    device const uchar* row_ptr2 = weights2 + a_offset2 + ulong(row2) * row_bytes;
    device const uchar* row_ptr3 = weights3 + a_offset3 + ulong(row3) * row_bytes;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (uint bi = lane; bi < blocks_per_row; bi += 32u) {
        device const uchar* block0 = row_ptr0 + bi * 34u;
        device const uchar* block1 = row_ptr1 + bi * 34u;
        device const uchar* block2 = row_ptr2 + bi * 34u;
        device const uchar* block3 = row_ptr3 + bi * 34u;
        const float scale0 = float(as_type<half>(*(device const ushort*)(block0)));
        const float scale1 = has1 ? float(as_type<half>(*(device const ushort*)(block1))) : 0.0f;
        const float scale2 = has2 ? float(as_type<half>(*(device const ushort*)(block2))) : 0.0f;
        const float scale3 = has3 ? float(as_type<half>(*(device const ushort*)(block3))) : 0.0f;
        device const packed_char4* quants0 = (device const packed_char4*)(block0 + 2u);
        device const packed_char4* quants1 = (device const packed_char4*)(block1 + 2u);
        device const packed_char4* quants2 = (device const packed_char4*)(block2 + 2u);
        device const packed_char4* quants3 = (device const packed_char4*)(block3 + 2u);
        const uint x_base = bi << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const uint idx = x_base + (vi << 2);
            const float4 h4 = *(device const float4*)(h + idx);
            const float4 nw4 = *(device const float4*)(norm_weight + idx);
            const float4 x = nw4 * (h4 * rms_inv);
            acc0 = fma(scale0, dot(float4(char4(quants0[vi])), x), acc0);
            if (has1) {
                acc1 = fma(scale1, dot(float4(char4(quants1[vi])), x), acc1);
            }
            if (has2) {
                acc2 = fma(scale2, dot(float4(char4(quants2[vi])), x), acc2);
            }
            if (has3) {
                acc3 = fma(scale3, dot(float4(char4(quants3[vi])), x), acc3);
            }
        }
    }

    // Cycle ~69: pack the four final-reduction `simd_sum` calls into one
    // `simd_sum(float4)` — Apple9 lowers vector `simd_sum` to a single
    // log2(32)=5-level butterfly that transfers 128-bit packed lanes per
    // shuffle_xor instead of four independent 32-bit trees, cutting per-
    // simdgroup tail shuffle traffic ~4×. Same proven pattern as cycle ~68
    // (the non-conv1d fused-norm dual sibling `dmmv_q8_0_dual_fused_norm.metal`)
    // applied symmetrically to this conv1d-fused variant. Production hot use:
    // SSM qkv+z fused-norm+conv1d path (M0=8192 M1=4096, ≈30 SSM layers ×
    // every decode token via the layer-0 SSM and the layer-N `prev_fused_attn_norm`
    // path that selects this fused-conv1d kernel when conv_channels==8192). The
    // downstream lane<4 paired writeback + conv1d postlude consume the four
    // sums as simdgroup-uniform scalars, so picking float4 components by lane
    // is bit-equivalent to the prior four independent `simd_sum` calls.
    const float4 sums = simd_sum(float4(acc0, acc1, acc2, acc3));
    // Parallelize the 4-row writeback + conv1d postlude across lanes 0..3.
    // After simd_sum all four sums are present on every lane, and row0..row3
    // are disjoint conv channels owned by this simdgroup, so the lanes touch
    // non-overlapping output*/conv_state/conv_out addresses (offsets r+0..r+3).
    // Lanes 0..3 reading conv_state[r..r+3] (also c+r..c+r+3 and 2c+r..2c+r+3)
    // issue each quad as a coalesced transaction instead of the previous four
    // serial lane-0 loads — mirrors cycle-27's lane 0/1 parallelization in the
    // sibling `dmmv_q8_0_repacked_k2048_dual_nr2_qwen_conv1d.metal` and
    // completes the symmetric postlude pattern across both Qwen SSM
    // qkv+conv1d-fused decode paths.
    if (lane < 4u) {
        const bool has = (lane == 0u) ? true
                       : (lane == 1u) ? has1
                       : (lane == 2u) ? has2
                       : has3;
        const bool first = (lane == 0u) ? first0
                         : (lane == 1u) ? first1
                         : (lane == 2u) ? first2
                         : first3;
        const uint local_row = (lane == 0u) ? row0
                             : (lane == 1u) ? row1
                             : (lane == 2u) ? row2
                             : row3;
        const float local_sum = (lane == 0u) ? sums.x
                              : (lane == 1u) ? sums.y
                              : (lane == 2u) ? sums.z
                              : sums.w;
        device float* local_output = (lane == 0u) ? output0
                                   : (lane == 1u) ? output1
                                   : (lane == 2u) ? output2
                                   : output3;

        if (has) {
            local_output[local_row] = local_sum;

            if (first) {
                const uint c = p.conv_channels;
                const float x0 = conv_state[local_row];
                const float x1 = conv_state[c + local_row];
                const float x2 = conv_state[2u * c + local_row];
                const float4 w = *(device const float4*)(conv_kernel + local_row * 4u);
                const float s = dot(w, float4(x0, x1, x2, local_sum));
                conv_out[local_row] = s * fast::divide(1.0f, 1.0f + fast::exp(-s));
                conv_state[local_row] = x1;
                conv_state[c + local_row] = x2;
                conv_state[2u * c + local_row] = local_sum;
            }
        }
    }
}
