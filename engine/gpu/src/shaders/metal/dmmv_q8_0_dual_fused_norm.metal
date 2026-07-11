#include <metal_stdlib>
using namespace metal;

struct DualQ8DmmvPush {
    uint M0;
    uint M1;
    uint K;
    uint a0_offset;
    uint a1_offset;
    uint x_offset;
    uint y0_offset;
    uint y1_offset;
};

// Fused RMSNorm + Dual-output Q8_0 DMMV — eliminates separate norm dispatch + barrier.
//
// Combines two Q8_0 matrix-vector multiplies that share the same input vector
// (e.g. SSM qkv 8192x2048 + gate 4096x2048) with inline RMSNorm computation.
// Each simdgroup handles four output rows, applying the reference implementation's adjacent-row
// matvec discipline more aggressively for the exact Qwen3.6 SSM shape while
// avoiding the intermediate norm_buf write and barrier.

kernel void main0(
    constant DualQ8DmmvPush& p [[buffer(0)]],
    device const uchar* W0 [[buffer(1)]],
    device const uchar* W1 [[buffer(2)]],
    device const float* hidden [[buffer(3)]],
    device float* Y0 [[buffer(4)]],
    device float* Y1 [[buffer(5)]],
    device const float* norm_weight [[buffer(6)]],
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

    // Cycle ~68: pack the four final-reduction `simd_sum` calls into one
    // `simd_sum(float4)` — Apple9 lowers vector `simd_sum` to a single
    // log2(32)=5-level butterfly that transfers 128-bit packed lanes per
    // shuffle_xor instead of four independent 32-bit trees, cutting per-
    // simdgroup tail shuffle traffic ~4×. Same pattern as cycle ~62
    // (`dmmv_q5k_moe_k512_quad`), cycle ~64 (`dmmv_q8_0_pair_swiglu`), and
    // cycle ~67 (the conv1d-fused sibling), now extended to this fused-norm
    // dual kernel. Hot uses: (a) SSM alpha+beta tail at M0=dt_rank=32 M1=32
    // (~1080 calls/req across 30 SSM layers × every decode token, 16 SGs/call
    // ⇒ ~52K shuffle savings/req); (b) production SSM qkv+z fused-norm path
    // (M0=8192 M1=4096, layer-0 SSM + fallback); (c) KV fused-norm at full-
    // attn layers. The downstream lane<4 paired writeback consumes the four
    // sums as simdgroup-uniform scalars, so picking float4 components by lane
    // is bit-equivalent.
    const float4 sums = simd_sum(float4(acc0, acc1, acc2, acc3));
    // Parallelize the 4-row writeback across lanes 0..3 (lane 0 serial 4
    // stores → lanes 0..3 parallel stores). After simd_sum all four sums are
    // broadcast to every lane, and per-row uniforms (row*/output*/has*) are
    // already computed by all lanes uniformly above. The production alpha+beta
    // tail (M0=dt_rank=32, M1=dt_rank=32, ~1080 calls/req across 30 SSM layers
    // × every decode token, block=32 → 1 SG/TG with 4 rows/SG) writes 4
    // contiguous floats per simdgroup into a single output buffer (Y0 or Y1,
    // since M0=32 is 4-aligned ⇒ no per-WG cross-buffer split) — issues a
    // coalesced 16-byte store instead of four serial lane-0 stores. The
    // has1/has2/has3 predicates remain to handle the generic-validation tail
    // (in-tree shader test uses M0=4, M1=3). Mirrors the sibling cycle-28
    // pattern in `dmmv_q8_0_dual_fused_norm_conv1d.metal` and cycle-27/32/38/
    // 39/40/41/43/45/46/47 lane-parallel writeback discipline across the Q8
    // family, completing it on the non-conv1d fused-norm dual kernel.
    if (lane < 4u) {
        const bool has = (lane == 0u) ? true
                       : (lane == 1u) ? has1
                       : (lane == 2u) ? has2
                       : has3;
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
        }
    }
}
