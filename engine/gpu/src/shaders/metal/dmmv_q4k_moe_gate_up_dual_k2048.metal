#include <metal_stdlib>
using namespace metal;

// Qwen3.6 exact-shape MoE gate/up dual Q4_K DMMV for K=2048.
//
// This mirrors dmmv_q4k_moe_k2048's L1-cached input discipline: each
// simdgroup reads the <=8 KiB route input directly instead of staging it
// through threadgroup memory and synchronizing all simdgroups.

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

inline void accumulate_q4k_direct(
    device const uchar* block,
    device const float* input,
    uint bi,
    uint lane,
    thread float& acc
) {
    const float d = float(as_type<half>(*(device const ushort*)(block)));
    const float dmin = float(as_type<half>(*(device const ushort*)(block + 2u)));
    device const uchar* scales = block + 4u;
    device const uchar* quants = block + 16u;

    const uint byte_off = lane * 4u;
    const uint j = byte_off / 32u;
    const uint local_off = byte_off % 32u;

    const uchar4 qbytes = *(device const uchar4*)(quants + byte_off);
    const float2 sm_lo = get_scale_min_k4(j * 2u, scales);
    const float2 sm_hi = get_scale_min_k4(j * 2u + 1u, scales);

    const float d_sc_lo = d * sm_lo.x;
    const float d_m_lo = dmin * sm_lo.y;
    const float d_sc_hi = d * sm_hi.x;
    const float d_m_hi = dmin * sm_hi.y;

    const uint col_lo = bi * 256u + j * 64u + local_off;
    const uint col_hi = col_lo + 32u;

    const float4 x_lo = *(device const float4*)(input + col_lo);
    const float4 x_hi = *(device const float4*)(input + col_hi);

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

    acc += dot(lo_vals, x_lo) + dot(hi_vals, x_hi);
}

#define TG_SIZE 512
#define ROWS_PER_TG (TG_SIZE / 32)

kernel void main0(
    device const uchar* gateW                     [[buffer(0)]],
    constant MoeGateUpDualDmmvPush& p             [[buffer(1)]],
    device const uchar* upW                       [[buffer(2)]],
    device const float* X                         [[buffer(3)]],
    device float* gateY                           [[buffer(4)]],
    device float* upY                             [[buffer(5)]],
    device const uint* expert_ids                 [[buffer(6)]],
    uint3 tg_pos                                  [[threadgroup_position_in_grid]],
    ushort lane                                   [[thread_index_in_simdgroup]],
    ushort simdgroup                              [[simdgroup_index_in_threadgroup]]
) {
    const uint expert_slot = tg_pos.y;
    const uint expert_id = expert_ids[expert_slot];
    const uint row = tg_pos.x * ROWS_PER_TG + uint(simdgroup);
    if (row >= p.M) {
        return;
    }

    device const float* input = X + (p.x_offset / 4u) + expert_slot * p.x_expert_stride;
    const uint blocks_per_row = p.K / 256u;
    const ulong row_bytes = ulong(blocks_per_row) * 144ul;
    const ulong gate_expert_base = ulong(p.gate_a_offset) + ulong(expert_id) * ulong(p.gate_expert_stride);
    const ulong up_expert_base = ulong(p.up_a_offset) + ulong(expert_id) * ulong(p.up_expert_stride);

    device const uchar* gate_row = gateW + gate_expert_base + ulong(row) * row_bytes;
    device const uchar* up_row = upW + up_expert_base + ulong(row) * row_bytes;

    float gate_acc = 0.0f;
    float up_acc = 0.0f;

    for (uint bi = 0u; bi < blocks_per_row; bi++) {
        accumulate_q4k_direct(gate_row + ulong(bi) * 144ul, input, bi, uint(lane), gate_acc);
        accumulate_q4k_direct(up_row + ulong(bi) * 144ul, input, bi, uint(lane), up_acc);
    }

    // Pack the two per-row simdgroup reductions into one `simd_sum(float2)`
    // — Apple9's vector `simd_sum` lowers to a single log2(32)=5-level
    // butterfly that transfers a 64-bit packed lane per `shuffle_xor`
    // instead of two independent 32-bit trees, cutting cross-lane shuffle
    // traffic ~2× on the per-simdgroup tail. Same proven pattern as
    // cycle ~63 (`dmmv_q4k_moe_gate_up_swiglu_k2048.metal` — the SwiGLU-
    // fused sibling that production decode takes via the Qwen3.6 MoE
    // gate-up path) applied here to the non-fused validation/fallback
    // dual variant (qwen_prefill_validation / debug / gemma_moe_validation
    // paths and the M=17 partial-WG test). The downstream lane-0
    // writeback consumes both sums as simdgroup-uniform scalars
    // (component access on the simdgroup-uniform float2), so picking
    // float2 components is bit-equivalent to the prior two scalar
    // `simd_sum` calls. Writebacks stay lane-0 serial since gateY and
    // upY are disjoint buffers (no coalesce opportunity between them).
    const float2 sums = simd_sum(float2(gate_acc, up_acc));
    if (lane == 0u) {
        gateY[p.gate_y_offset / 4u + expert_slot * p.M + row] = sums.x;
        upY[p.up_y_offset / 4u + expert_slot * p.M + row] = sums.y;
    }
}
