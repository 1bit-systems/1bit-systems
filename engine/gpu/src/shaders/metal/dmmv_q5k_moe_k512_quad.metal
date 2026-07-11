#include <metal_stdlib>
using namespace metal;

// Qwen3.6 exact-shape Q5_K routed MoE down projection for K=512.
//
// Four output rows share one simdgroup. This extends the paired-row K=512
// shader so the same 512-wide SwiGLU vector is reused across four adjacent
// down rows, halving workgroups versus dmmv_q5k_moe_k512.

struct MoeDmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint expert_stride;
    uint x_expert_stride;
    uint x_offset;
    uint y_offset;
};

inline float2 get_scale_min_k5(uint j, device const uchar* scales) {
    if (j < 4u) {
        return float2(float(scales[j] & 63u), float(scales[4u + j] & 63u));
    }
    return float2(
        float((scales[4u + j] & 0x0Fu) | ((scales[j - 4u] >> 6u) << 4u)),
        float((scales[4u + j] >> 4u) | ((scales[j] >> 6u) << 4u))
    );
}

inline void accumulate_q5k_block_quad(
    device const uchar* block0,
    device const uchar* block1,
    device const uchar* block2,
    device const uchar* block3,
    threadgroup const float* input,
    uint col_base,
    uint lane,
    thread float& sum0,
    thread float& sum1,
    thread float& sum2,
    thread float& sum3
) {
    const float d0 = float(as_type<half>(*(device const ushort*)(block0)));
    const float dmin0 = float(as_type<half>(*(device const ushort*)(block0 + 2u)));
    device const uchar* scales0 = block0 + 4u;
    device const uchar* high_bits0 = block0 + 16u;
    device const uchar* quants0 = block0 + 48u;

    const float d1 = float(as_type<half>(*(device const ushort*)(block1)));
    const float dmin1 = float(as_type<half>(*(device const ushort*)(block1 + 2u)));
    device const uchar* scales1 = block1 + 4u;
    device const uchar* high_bits1 = block1 + 16u;
    device const uchar* quants1 = block1 + 48u;

    const float d2 = float(as_type<half>(*(device const ushort*)(block2)));
    const float dmin2 = float(as_type<half>(*(device const ushort*)(block2 + 2u)));
    device const uchar* scales2 = block2 + 4u;
    device const uchar* high_bits2 = block2 + 16u;
    device const uchar* quants2 = block2 + 48u;

    const float d3 = float(as_type<half>(*(device const ushort*)(block3)));
    const float dmin3 = float(as_type<half>(*(device const ushort*)(block3 + 2u)));
    device const uchar* scales3 = block3 + 4u;
    device const uchar* high_bits3 = block3 + 16u;
    device const uchar* quants3 = block3 + 48u;

    const uint qh_val0 = uint(high_bits0[lane]);
    const uint qh_val1 = uint(high_bits1[lane]);
    const uint qh_val2 = uint(high_bits2[lane]);
    const uint qh_val3 = uint(high_bits3[lane]);

    #pragma unroll
    for (uint g = 0u; g < 4u; g++) {
        const uint sb_lo = g * 2u;
        const uint sb_hi = sb_lo + 1u;
        const uint col_lo = col_base + g * 64u + lane;
        const uint col_hi = col_lo + 32u;
        const float x_lo = input[col_lo];
        const float x_hi = input[col_hi];

        const float2 sm0_lo = get_scale_min_k5(sb_lo, scales0);
        const float2 sm0_hi = get_scale_min_k5(sb_hi, scales0);
        const float2 sm1_lo = get_scale_min_k5(sb_lo, scales1);
        const float2 sm1_hi = get_scale_min_k5(sb_hi, scales1);
        const float2 sm2_lo = get_scale_min_k5(sb_lo, scales2);
        const float2 sm2_hi = get_scale_min_k5(sb_hi, scales2);
        const float2 sm3_lo = get_scale_min_k5(sb_lo, scales3);
        const float2 sm3_hi = get_scale_min_k5(sb_hi, scales3);

        const uint q0_byte = uint(quants0[g * 32u + lane]);
        const uint q1_byte = uint(quants1[g * 32u + lane]);
        const uint q2_byte = uint(quants2[g * 32u + lane]);
        const uint q3_byte = uint(quants3[g * 32u + lane]);

        sum0 += (d0 * sm0_lo.x * float((q0_byte & 0x0Fu) | (((qh_val0 >> sb_lo) & 1u) << 4u)) - dmin0 * sm0_lo.y) * x_lo;
        sum0 += (d0 * sm0_hi.x * float((q0_byte >> 4u) | (((qh_val0 >> sb_hi) & 1u) << 4u)) - dmin0 * sm0_hi.y) * x_hi;
        sum1 += (d1 * sm1_lo.x * float((q1_byte & 0x0Fu) | (((qh_val1 >> sb_lo) & 1u) << 4u)) - dmin1 * sm1_lo.y) * x_lo;
        sum1 += (d1 * sm1_hi.x * float((q1_byte >> 4u) | (((qh_val1 >> sb_hi) & 1u) << 4u)) - dmin1 * sm1_hi.y) * x_hi;
        sum2 += (d2 * sm2_lo.x * float((q2_byte & 0x0Fu) | (((qh_val2 >> sb_lo) & 1u) << 4u)) - dmin2 * sm2_lo.y) * x_lo;
        sum2 += (d2 * sm2_hi.x * float((q2_byte >> 4u) | (((qh_val2 >> sb_hi) & 1u) << 4u)) - dmin2 * sm2_hi.y) * x_hi;
        sum3 += (d3 * sm3_lo.x * float((q3_byte & 0x0Fu) | (((qh_val3 >> sb_lo) & 1u) << 4u)) - dmin3 * sm3_lo.y) * x_lo;
        sum3 += (d3 * sm3_hi.x * float((q3_byte >> 4u) | (((qh_val3 >> sb_hi) & 1u) << 4u)) - dmin3 * sm3_hi.y) * x_hi;
    }
}

#define TG_SIZE 512
#define ROWS_PER_TG ((TG_SIZE / 32) * 4)

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant MoeDmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    device const uint* expert_ids [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint expert_slot = tg_pos.y;
    const uint expert_id = expert_ids[expert_slot];

    // Cooperative X cache: all 16 simdgroups in this TG share the same
    // `expert_slot = tg_pos.y` and therefore the same 512-float input vector
    // (the SwiGLU output for one active expert). Previously each simdgroup
    // re-read the same 512 floats from L1, 16×512=8192 redundant reads per TG.
    // Stage once into TG memory: TG_SIZE=512 threads ⇒ 1:1 mapping, each
    // thread loads exactly one float from contiguous DRAM offsets (perfect
    // coalescing). Same x_cache discipline as cycle ~56 in
    // `dmmv_q4k_moe_gate_up_swiglu_k2048.metal` (hot kernel #2). Load+barrier
    // happen BEFORE the `row0 >= p.M` early-return so partial-TG tails
    // (test M=69 ⇒ second TG has 59 idle rows) still satisfy barrier liveness.
    threadgroup float x_cache[512];
    {
        device const float* input_src = X + (p.x_offset / 4u) + expert_slot * p.x_expert_stride;
        const uint local_id = simdgroup * 32u + lane;
        x_cache[local_id] = input_src[local_id];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint row0 = tg_pos.x * ROWS_PER_TG + simdgroup * 4u;
    if (row0 >= p.M) return;
    const uint row1 = row0 + 1u;
    const uint row2 = row0 + 2u;
    const uint row3 = row0 + 3u;
    const bool has_row1 = row1 < p.M;
    const bool has_row2 = row2 < p.M;
    const bool has_row3 = row3 < p.M;

    const ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    device const uchar* row0_ptr = W + expert_base + ulong(row0) * 352ul;
    device const uchar* row1_ptr = has_row1 ? (row0_ptr + 352ul) : row0_ptr;
    device const uchar* row2_ptr = has_row2 ? (row0_ptr + 704ul) : row0_ptr;
    device const uchar* row3_ptr = has_row3 ? (row0_ptr + 1056ul) : row0_ptr;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    accumulate_q5k_block_quad(row0_ptr, row1_ptr, row2_ptr, row3_ptr, x_cache, 0u, lane, sum0, sum1, sum2, sum3);
    accumulate_q5k_block_quad(row0_ptr + 176u, row1_ptr + 176u, row2_ptr + 176u, row3_ptr + 176u, x_cache, 256u, lane, sum0, sum1, sum2, sum3);

    // Cycle ~62: pack four scalar simd_sums into one float4 simd_sum. Metal's
    // simd_sum is overloaded for vector types on Apple9 (M4) — a float4
    // reduction emits one log2(32)=5-level shuffle_xor tree that transfers
    // 128-bit packed lanes per shuffle instead of four independent 32-bit
    // trees. Cuts the per-simdgroup reduction shuffle count ~4× on the hot
    // kernel #3 critical path (270 ms/req across 1296 dispatches, 16
    // simdgroups/TG × 1024 TGs/dispatch ⇒ ~21M simd_sums replaced per req).
    // The downstream lane<4 quad writeback already consumed total0/1/2/3 as
    // simdgroup-uniform scalars, so picking from the float4 components is
    // bit-equivalent. Mirrors cycle-43/45/47/52 lane-parallel-writeback
    // discipline and extends the cycle-51 simd_sum reduction pattern to the
    // simultaneous-multi-row case.
    const float4 totals = simd_sum(float4(sum0, sum1, sum2, sum3));
    // Parallelize the 4-row writeback across lanes 0..3.
    // After simd_sum all four totals are present on every lane; output[row0..row3]
    // are four contiguous floats so the Qwen3.6-35B production MoE-down case
    // (M=2048, M%4==0 ⇒ every simdgroup writes a full quad) issues a single
    // coalesced 16-byte store instead of four serial lane-0 stores. The
    // has_row1/2/3 predicates remain to handle the generic-validation tail
    // (the in-tree shader test uses M=69). Mirrors cycle-27/32/38/39/40
    // lane-parallel writeback discipline across the Q8 family and extends it
    // to the hottest Q5_K MoE-down kernel (~271ms/req of kernel timing, hot
    // kernel #3, ~7.7K TGs/decode token across 30 SSM layers × 8 experts).
    if (lane < 4u) {
        const bool has = (lane == 0u) ? true
                       : (lane == 1u) ? has_row1
                       : (lane == 2u) ? has_row2
                       : has_row3;
        if (has) {
            device float* output = Y + (p.y_offset / 4u) + expert_slot * p.M;
            const float local_sum = (lane == 0u) ? totals.x
                                  : (lane == 1u) ? totals.y
                                  : (lane == 2u) ? totals.z
                                  : totals.w;
            output[row0 + lane] = local_sum;
        }
    }
}
