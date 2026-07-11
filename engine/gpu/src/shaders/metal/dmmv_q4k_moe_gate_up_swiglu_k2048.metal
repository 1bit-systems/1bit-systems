#include <metal_stdlib>
using namespace metal;

// Qwen3.6 exact-shape MoE gate/up dual Q4_K DMMV for K=2048 with SwiGLU fused.
//
// This is the same direct-input, one-simdgroup-per-row shape as
// dmmv_q4k_moe_gate_up_dual_k2048, but writes silu(gate) * up directly to the
// routed expert activation buffer. The token-major Qwen path only consumes the
// gate/up projections through the following SwiGLU, so this removes one small
// dispatch from every routed MoE layer without changing routing semantics.
//
// Like the reference implementation's fixed-shape Metal mul_mv specializations, this bakes the
// Qwen3.6 K=2048 block count and row stride into the hot loop.

struct MoeGateUpDualDmmvPush {
    uint M;
    uint K;
    uint gate_a_offset;
    uint up_a_offset;
    uint gate_expert_stride;
    uint up_expert_stride;
    uint x_expert_stride;
    uint x_offset;
    uint gate_y_offset;
    uint up_y_offset;
};

inline float2 get_scale_min_k4(uint j, device const uchar* sc) {
    if (j < 4u) {
        return float2(float(sc[j] & 63u), float(sc[j + 4u] & 63u));
    }
    return float2(
        float((sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6u) << 4u)),
        float(((sc[j + 4u] >> 4u) & 0x0Fu) | ((sc[j] >> 6u) << 4u))
    );
}

// Paired gate+up Q4_K accumulation: the gate and up blocks at index `bi`
// dequantize against the SAME X half-blocks. Compute col_lo/col_hi and the
// two float4 X loads once, then fold them into both accumulators. Sharing
// the X loads explicitly avoids relying on the inliner to CSE across device
// pointer function calls — mirrors the X-reuse discipline in the Q8_0
// dmmv_q8_0_pair_swiglu fused kernel.
inline void accumulate_q4k_pair_direct(
    device const uchar* gate_block,
    device const uchar* up_block,
    threadgroup const float* input,
    uint bi,
    uint lane,
    thread float& gate_acc,
    thread float& up_acc
) {
    const uint byte_off = lane * 4u;
    const uint j = byte_off / 32u;
    const uint local_off = byte_off % 32u;

    const uint col_lo = bi * 256u + j * 64u + local_off;
    const uint col_hi = col_lo + 32u;

    const float4 x_lo = *(threadgroup const float4*)(input + col_lo);
    const float4 x_hi = *(threadgroup const float4*)(input + col_hi);

    {
        const float d = float(as_type<half>(*(device const ushort*)(gate_block)));
        const float dmin = float(as_type<half>(*(device const ushort*)(gate_block + 2u)));
        device const uchar* scales = gate_block + 4u;
        device const uchar* quants = gate_block + 16u;

        const uchar4 qbytes = *(device const uchar4*)(quants + byte_off);
        const float2 sm_lo = get_scale_min_k4(j * 2u, scales);
        const float2 sm_hi = get_scale_min_k4(j * 2u + 1u, scales);

        const float d_sc_lo = d * sm_lo.x;
        const float d_m_lo = dmin * sm_lo.y;
        const float d_sc_hi = d * sm_hi.x;
        const float d_m_hi = dmin * sm_hi.y;

        const uchar4 q_lo = uchar4(
            qbytes.x & 0x0F,
            qbytes.y & 0x0F,
            qbytes.z & 0x0F,
            qbytes.w & 0x0F
        );
        const uchar4 q_hi = uchar4(
            qbytes.x >> 4,
            qbytes.y >> 4,
            qbytes.z >> 4,
            qbytes.w >> 4
        );

        const float4 lo_vals = fma(float4(q_lo), float4(d_sc_lo), float4(-d_m_lo));
        const float4 hi_vals = fma(float4(q_hi), float4(d_sc_hi), float4(-d_m_hi));

        gate_acc += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
    }

    {
        const float d = float(as_type<half>(*(device const ushort*)(up_block)));
        const float dmin = float(as_type<half>(*(device const ushort*)(up_block + 2u)));
        device const uchar* scales = up_block + 4u;
        device const uchar* quants = up_block + 16u;

        const uchar4 qbytes = *(device const uchar4*)(quants + byte_off);
        const float2 sm_lo = get_scale_min_k4(j * 2u, scales);
        const float2 sm_hi = get_scale_min_k4(j * 2u + 1u, scales);

        const float d_sc_lo = d * sm_lo.x;
        const float d_m_lo = dmin * sm_lo.y;
        const float d_sc_hi = d * sm_hi.x;
        const float d_m_hi = dmin * sm_hi.y;

        const uchar4 q_lo = uchar4(
            qbytes.x & 0x0F,
            qbytes.y & 0x0F,
            qbytes.z & 0x0F,
            qbytes.w & 0x0F
        );
        const uchar4 q_hi = uchar4(
            qbytes.x >> 4,
            qbytes.y >> 4,
            qbytes.z >> 4,
            qbytes.w >> 4
        );

        const float4 lo_vals = fma(float4(q_lo), float4(d_sc_lo), float4(-d_m_lo));
        const float4 hi_vals = fma(float4(q_hi), float4(d_sc_hi), float4(-d_m_hi));

        up_acc += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
    }
}

#define TG_SIZE 512
#define ROWS_PER_TG (TG_SIZE / 32)

kernel void main0(
    device const uchar* gateW                     [[buffer(0)]],
    constant MoeGateUpDualDmmvPush& p             [[buffer(1)]],
    device const uchar* upW                       [[buffer(2)]],
    device const float* X                         [[buffer(3)]],
    device float* swigluY                         [[buffer(4)]],
    device const uint* expert_ids                 [[buffer(5)]],
    uint3 tg_pos                                  [[threadgroup_position_in_grid]],
    ushort lane                                   [[thread_index_in_simdgroup]],
    ushort simdgroup                              [[simdgroup_index_in_threadgroup]]
) {
    const uint expert_slot = tg_pos.y;
    const uint row = tg_pos.x * ROWS_PER_TG + uint(simdgroup);

    // Cycle ~56: cooperative X load into threadgroup memory. All 16 simdgroups
    // in this TG handle the SAME expert_slot (`expert_slot = tg_pos.y`) and the
    // dispatch site (forward_metal.zig:9157 `dispatchDmmvMoeGateUpSwiGLUQ4kOnCmd`)
    // passes `x_expert_stride=0`, so EVERY simdgroup reads identical 2048-float
    // X. Previously each simdgroup re-issued ~256 device float4 loads against
    // the same 8 KB X buffer (16 sg × 256 reads = 4096 redundant L1-cached reads
    // per TG). Stage X into 8 KB of threadgroup memory once (TG_SIZE=512 threads
    // each load 1 float4 from contiguous offsets — perfect DRAM coalescing),
    // then read x_lo/x_hi from TG mem inside the matvec (1-2 cycle latency vs
    // 10+ cycle L1 cache latency on Apple9). Same x_cache discipline as the
    // hot kernel #1 sibling `residual_rms_norm_router_f32_topk` (uses
    // `x_cache4`) and `residual_rms_norm_router_q8_0_topk_repacked_k2048`
    // (uses `norm_cache`). 1024 TGs × ~217 µs/dispatch on the #2 hot
    // decode-token kernel (304 ms/req across 1400 calls, 20% of timed kernel
    // time). Also frees L1 capacity for the 36 KB of W reads per TG.
    threadgroup float x_cache[2048];
    device const float* input = X + (p.x_offset / 4u) + expert_slot * p.x_expert_stride;
    const uint local_id = uint(simdgroup) * 32u + uint(lane);
    *(threadgroup float4*)(&x_cache[local_id * 4u]) = *(device const float4*)(input + local_id * 4u);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (row >= p.M) {
        return;
    }

    const uint expert_id = expert_ids[expert_slot];
    constexpr uint blocks_per_row = 8u;
    constexpr ulong row_bytes = 1152ul;
    const ulong gate_expert_base = ulong(p.gate_a_offset) + ulong(expert_id) * ulong(p.gate_expert_stride);
    const ulong up_expert_base = ulong(p.up_a_offset) + ulong(expert_id) * ulong(p.up_expert_stride);

    device const uchar* gate_row = gateW + gate_expert_base + ulong(row) * row_bytes;
    device const uchar* up_row = upW + up_expert_base + ulong(row) * row_bytes;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    #pragma unroll
    for (uint bi = 0u; bi < blocks_per_row; bi++) {
        accumulate_q4k_pair_direct(
            gate_row + ulong(bi) * 144ul,
            up_row + ulong(bi) * 144ul,
            x_cache, bi, uint(lane), gate_acc, up_acc);
    }

    // Cycle ~63: pack the two final-reduction simd_sum calls into a single
    // simd_sum(float2) — Apple's `simd_sum` on float2/float4 issues one fused
    // butterfly per lane width (5 shuffle_xor pairs total for SIMD32) instead
    // of N sequential reductions, halving cross-lane shuffle traffic on the
    // per-simdgroup row tail. Same pattern as cycle ~62 on hot kernel #3
    // (`dmmv_q5k_moe_k512_quad`'s float4 pack). Hot kernel #2 (303 ms/req
    // across 1400 calls), so the per-row tail is amplified by ROWS_PER_TG=16.
    const float2 sums = simd_sum(float2(gate_acc, up_acc));
    if (lane == 0u) {
        const float gate_sum = sums.x;
        const float up_sum = sums.y;
        swigluY[p.gate_y_offset / 4u + expert_slot * p.M + row] =
            gate_sum * fast::divide(1.0f, 1.0f + fast::exp(-gate_sum)) * up_sum;
    }
}
