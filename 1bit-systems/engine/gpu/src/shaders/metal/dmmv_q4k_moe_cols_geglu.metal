#include <metal_stdlib>
using namespace metal;

// Route-packed Gemma MoE Q4_K gate/up DMMV fused with GeGLU.
//
// This consumes the same expert-major packed route IDs as dmmv_q4k_moe_cols,
// but reads Gemma's fused [gate, up] expert tensor and writes activated route
// outputs directly. It saves the separate gate column dispatch, up column
// dispatch, and GeGLU dispatch from the active-block batched prefill path.

struct MoeColsGateUpDmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint expert_stride;
    uint gate_base_offset;
    uint up_base_offset;
    uint x_offset;
    uint y_offset;
    uint ids_stride;
    uint x_route_divisor;
    uint use_active_blocks;
    uint enable_exact5;
};

inline float2 get_scale_min_k4(uint j, device const uchar* sc) {
    if (j < 4) {
        return float2(float(sc[j] & 63), float(sc[j + 4] & 63));
    }
    return float2(
        float((sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4)),
        float(((sc[j + 4] >> 4) & 0x0F) | ((sc[j] >> 6) << 4))
    );
}

inline float geglu(float gate, float up) {
    const float g3 = gate * gate * gate;
    float inner = 0.7978845608f * (gate + 0.044715f * g3);
    inner = clamp(inner, -15.0f, 15.0f);
    const float gelu_gate = 0.5f * gate * (1.0f + precise::tanh(inner));
    return gelu_gate * up;
}

#define NUM_COLS 8u
#define ROWS_PER_TG 8u

kernel void main0(
    device const uchar* W                     [[buffer(0)]],
    constant MoeColsGateUpDmmvPush& p         [[buffer(1)]],
    device const float* X                     [[buffer(2)]],
    device float* activatedY                  [[buffer(3)]],
    device const uint* counts                 [[buffer(4)]],
    device const uint* packed_ids             [[buffer(5)]],
    device const uint* active_blocks          [[buffer(6)]],
    device const uint* active_block_count     [[buffer(7)]],
    uint3 tg_pos                              [[threadgroup_position_in_grid]],
    uint tid                                  [[thread_index_in_simdgroup]],
    uint sgid                                 [[simdgroup_index_in_threadgroup]]
) {
    if (p.use_active_blocks != 0u && tg_pos.y >= active_block_count[0]) {
        return;
    }

    const uint block_entry = (p.use_active_blocks != 0u) ? active_blocks[tg_pos.y] : 0u;
    const uint expert_id = (p.use_active_blocks != 0u) ? (block_entry & 0xFFFFu) : tg_pos.y;
    const uint row = tg_pos.x * ROWS_PER_TG + sgid;
    if (row >= p.M) {
        return;
    }

    const uint packed_base = (p.use_active_blocks != 0u) ? ((block_entry >> 16u) * NUM_COLS) : (tg_pos.z * NUM_COLS);
    const uint count = counts[expert_id];
    if (packed_base >= count) {
        return;
    }

    device const uint* expert_ids = packed_ids + expert_id * p.ids_stride;
    const uint remaining = count - packed_base;
    const bool active0 = true;
    const uint route0 = expert_ids[packed_base + 0u];

    const uint x_div = max(p.x_route_divisor, 1u);
    device const float* x_base = X + (p.x_offset / 4u);
    device const float* x0 = x_base + (route0 / x_div) * p.K;

    const uint nb = p.K / 256u;
    const uint bpb = 144u;
    const ulong row_bytes = ulong(nb) * ulong(bpb);
    const ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    device const uchar* gate_src = W + expert_base + ulong(p.gate_base_offset) + ulong(row) * row_bytes;
    device const uchar* up_src = W + expert_base + ulong(p.up_base_offset) + ulong(row) * row_bytes;

    if (remaining == 1u) {
        float gate_acc = 0.0f;
        float up_acc = 0.0f;

        for (uint b = 0u; b < nb; b++) {
            device const uchar* gate_block = gate_src + b * bpb;
            device const uchar* up_block = up_src + b * bpb;

            const float gate_d = float(as_type<half>(*(device const ushort*)(gate_block)));
            const float gate_dmin = float(as_type<half>(*(device const ushort*)(gate_block + 2)));
            device const uchar* gate_scales = gate_block + 4;
            device const uchar* gate_quants = gate_block + 16;

            const float up_d = float(as_type<half>(*(device const ushort*)(up_block)));
            const float up_dmin = float(as_type<half>(*(device const ushort*)(up_block + 2)));
            device const uchar* up_scales = up_block + 4;
            device const uchar* up_quants = up_block + 16;

            const uint byte_off = tid * 4u;
            const uint j = byte_off / 32u;
            const uint local_off = byte_off % 32u;

            const uchar4 gate_qbytes = *(device const uchar4*)(gate_quants + byte_off);
            const float2 gate_sm_lo = get_scale_min_k4(j * 2u, gate_scales);
            const float2 gate_sm_hi = get_scale_min_k4(j * 2u + 1u, gate_scales);

            const float gate_d_sc_lo = gate_d * gate_sm_lo.x;
            const float gate_d_m_lo = gate_dmin * gate_sm_lo.y;
            const float gate_d_sc_hi = gate_d * gate_sm_hi.x;
            const float gate_d_m_hi = gate_dmin * gate_sm_hi.y;

            const uchar4 up_qbytes = *(device const uchar4*)(up_quants + byte_off);
            const float2 up_sm_lo = get_scale_min_k4(j * 2u, up_scales);
            const float2 up_sm_hi = get_scale_min_k4(j * 2u + 1u, up_scales);

            const float up_d_sc_lo = up_d * up_sm_lo.x;
            const float up_d_m_lo = up_dmin * up_sm_lo.y;
            const float up_d_sc_hi = up_d * up_sm_hi.x;
            const float up_d_m_hi = up_dmin * up_sm_hi.y;

            const uint col_lo = b * 256u + j * 64u + local_off;
            const uint col_hi = col_lo + 32u;

            const uchar4 gate_q_lo = uchar4(
                gate_qbytes.x & 0x0F,
                gate_qbytes.y & 0x0F,
                gate_qbytes.z & 0x0F,
                gate_qbytes.w & 0x0F
            );
            const uchar4 gate_q_hi = uchar4(
                gate_qbytes.x >> 4,
                gate_qbytes.y >> 4,
                gate_qbytes.z >> 4,
                gate_qbytes.w >> 4
            );
            const float4 gate_lo_vals = fma(float4(gate_q_lo), float4(gate_d_sc_lo), float4(-gate_d_m_lo));
            const float4 gate_hi_vals = fma(float4(gate_q_hi), float4(gate_d_sc_hi), float4(-gate_d_m_hi));

            const uchar4 up_q_lo = uchar4(
                up_qbytes.x & 0x0F,
                up_qbytes.y & 0x0F,
                up_qbytes.z & 0x0F,
                up_qbytes.w & 0x0F
            );
            const uchar4 up_q_hi = uchar4(
                up_qbytes.x >> 4,
                up_qbytes.y >> 4,
                up_qbytes.z >> 4,
                up_qbytes.w >> 4
            );
            const float4 up_lo_vals = fma(float4(up_q_lo), float4(up_d_sc_lo), float4(-up_d_m_lo));
            const float4 up_hi_vals = fma(float4(up_q_hi), float4(up_d_sc_hi), float4(-up_d_m_hi));

            const float4 x_lo = *(device const float4*)(x0 + col_lo);
            const float4 x_hi = *(device const float4*)(x0 + col_hi);
            gate_acc += dot(gate_lo_vals, x_lo) + dot(gate_hi_vals, x_hi);
            up_acc += dot(up_lo_vals, x_lo) + dot(up_hi_vals, x_hi);
        }

        const float gate0 = simd_sum(gate_acc);
        const float up0 = simd_sum(up_acc);

        device float* y_base = activatedY + (p.y_offset / 4u);
        if (tid == 0u) {
            y_base[route0 * p.M + row] = geglu(gate0, up0);
        }
        return;
    }

    if (remaining == 2u) {
        const uint route1 = expert_ids[packed_base + 1u];
        device const float* x1 = x_base + (route1 / x_div) * p.K;

        float2 gate_acc = float2(0.0f);
        float2 up_acc = float2(0.0f);

        for (uint b = 0u; b < nb; b++) {
            device const uchar* gate_block = gate_src + b * bpb;
            device const uchar* up_block = up_src + b * bpb;

            const float gate_d = float(as_type<half>(*(device const ushort*)(gate_block)));
            const float gate_dmin = float(as_type<half>(*(device const ushort*)(gate_block + 2)));
            device const uchar* gate_scales = gate_block + 4;
            device const uchar* gate_quants = gate_block + 16;

            const float up_d = float(as_type<half>(*(device const ushort*)(up_block)));
            const float up_dmin = float(as_type<half>(*(device const ushort*)(up_block + 2)));
            device const uchar* up_scales = up_block + 4;
            device const uchar* up_quants = up_block + 16;

            const uint byte_off = tid * 4u;
            const uint j = byte_off / 32u;
            const uint local_off = byte_off % 32u;

            const uchar4 gate_qbytes = *(device const uchar4*)(gate_quants + byte_off);
            const float2 gate_sm_lo = get_scale_min_k4(j * 2u, gate_scales);
            const float2 gate_sm_hi = get_scale_min_k4(j * 2u + 1u, gate_scales);

            const float gate_d_sc_lo = gate_d * gate_sm_lo.x;
            const float gate_d_m_lo = gate_dmin * gate_sm_lo.y;
            const float gate_d_sc_hi = gate_d * gate_sm_hi.x;
            const float gate_d_m_hi = gate_dmin * gate_sm_hi.y;

            const uchar4 up_qbytes = *(device const uchar4*)(up_quants + byte_off);
            const float2 up_sm_lo = get_scale_min_k4(j * 2u, up_scales);
            const float2 up_sm_hi = get_scale_min_k4(j * 2u + 1u, up_scales);

            const float up_d_sc_lo = up_d * up_sm_lo.x;
            const float up_d_m_lo = up_dmin * up_sm_lo.y;
            const float up_d_sc_hi = up_d * up_sm_hi.x;
            const float up_d_m_hi = up_dmin * up_sm_hi.y;

            const uint col_lo = b * 256u + j * 64u + local_off;
            const uint col_hi = col_lo + 32u;

            const uchar4 gate_q_lo = uchar4(
                gate_qbytes.x & 0x0F,
                gate_qbytes.y & 0x0F,
                gate_qbytes.z & 0x0F,
                gate_qbytes.w & 0x0F
            );
            const uchar4 gate_q_hi = uchar4(
                gate_qbytes.x >> 4,
                gate_qbytes.y >> 4,
                gate_qbytes.z >> 4,
                gate_qbytes.w >> 4
            );
            const float4 gate_lo_vals = fma(float4(gate_q_lo), float4(gate_d_sc_lo), float4(-gate_d_m_lo));
            const float4 gate_hi_vals = fma(float4(gate_q_hi), float4(gate_d_sc_hi), float4(-gate_d_m_hi));

            const uchar4 up_q_lo = uchar4(
                up_qbytes.x & 0x0F,
                up_qbytes.y & 0x0F,
                up_qbytes.z & 0x0F,
                up_qbytes.w & 0x0F
            );
            const uchar4 up_q_hi = uchar4(
                up_qbytes.x >> 4,
                up_qbytes.y >> 4,
                up_qbytes.z >> 4,
                up_qbytes.w >> 4
            );
            const float4 up_lo_vals = fma(float4(up_q_lo), float4(up_d_sc_lo), float4(-up_d_m_lo));
            const float4 up_hi_vals = fma(float4(up_q_hi), float4(up_d_sc_hi), float4(-up_d_m_hi));

            const float4 x0_lo = *(device const float4*)(x0 + col_lo);
            const float4 x0_hi = *(device const float4*)(x0 + col_hi);
            const float4 x1_lo = *(device const float4*)(x1 + col_lo);
            const float4 x1_hi = *(device const float4*)(x1 + col_hi);

            gate_acc.x += dot(gate_lo_vals, x0_lo) + dot(gate_hi_vals, x0_hi);
            gate_acc.y += dot(gate_lo_vals, x1_lo) + dot(gate_hi_vals, x1_hi);
            up_acc.x += dot(up_lo_vals, x0_lo) + dot(up_hi_vals, x0_hi);
            up_acc.y += dot(up_lo_vals, x1_lo) + dot(up_hi_vals, x1_hi);
        }

        const float gate0 = simd_sum(gate_acc.x);
        const float gate1 = simd_sum(gate_acc.y);
        const float up0 = simd_sum(up_acc.x);
        const float up1 = simd_sum(up_acc.y);

        device float* y_base = activatedY + (p.y_offset / 4u);
        if (tid == 0u) {
            y_base[route0 * p.M + row] = geglu(gate0, up0);
            y_base[route1 * p.M + row] = geglu(gate1, up1);
        }
        return;
    }

    if (remaining == 3u) {
        const uint route1 = expert_ids[packed_base + 1u];
        const uint route2 = expert_ids[packed_base + 2u];
        device const float* x1 = x_base + (route1 / x_div) * p.K;
        device const float* x2 = x_base + (route2 / x_div) * p.K;

        float3 gate_acc = float3(0.0f);
        float3 up_acc = float3(0.0f);

        for (uint b = 0u; b < nb; b++) {
            device const uchar* gate_block = gate_src + b * bpb;
            device const uchar* up_block = up_src + b * bpb;

            const float gate_d = float(as_type<half>(*(device const ushort*)(gate_block)));
            const float gate_dmin = float(as_type<half>(*(device const ushort*)(gate_block + 2)));
            device const uchar* gate_scales = gate_block + 4;
            device const uchar* gate_quants = gate_block + 16;

            const float up_d = float(as_type<half>(*(device const ushort*)(up_block)));
            const float up_dmin = float(as_type<half>(*(device const ushort*)(up_block + 2)));
            device const uchar* up_scales = up_block + 4;
            device const uchar* up_quants = up_block + 16;

            const uint byte_off = tid * 4u;
            const uint j = byte_off / 32u;
            const uint local_off = byte_off % 32u;

            const uchar4 gate_qbytes = *(device const uchar4*)(gate_quants + byte_off);
            const float2 gate_sm_lo = get_scale_min_k4(j * 2u, gate_scales);
            const float2 gate_sm_hi = get_scale_min_k4(j * 2u + 1u, gate_scales);

            const float gate_d_sc_lo = gate_d * gate_sm_lo.x;
            const float gate_d_m_lo = gate_dmin * gate_sm_lo.y;
            const float gate_d_sc_hi = gate_d * gate_sm_hi.x;
            const float gate_d_m_hi = gate_dmin * gate_sm_hi.y;

            const uchar4 up_qbytes = *(device const uchar4*)(up_quants + byte_off);
            const float2 up_sm_lo = get_scale_min_k4(j * 2u, up_scales);
            const float2 up_sm_hi = get_scale_min_k4(j * 2u + 1u, up_scales);

            const float up_d_sc_lo = up_d * up_sm_lo.x;
            const float up_d_m_lo = up_dmin * up_sm_lo.y;
            const float up_d_sc_hi = up_d * up_sm_hi.x;
            const float up_d_m_hi = up_dmin * up_sm_hi.y;

            const uint col_lo = b * 256u + j * 64u + local_off;
            const uint col_hi = col_lo + 32u;

            const uchar4 gate_q_lo = uchar4(
                gate_qbytes.x & 0x0F,
                gate_qbytes.y & 0x0F,
                gate_qbytes.z & 0x0F,
                gate_qbytes.w & 0x0F
            );
            const uchar4 gate_q_hi = uchar4(
                gate_qbytes.x >> 4,
                gate_qbytes.y >> 4,
                gate_qbytes.z >> 4,
                gate_qbytes.w >> 4
            );
            const float4 gate_lo_vals = fma(float4(gate_q_lo), float4(gate_d_sc_lo), float4(-gate_d_m_lo));
            const float4 gate_hi_vals = fma(float4(gate_q_hi), float4(gate_d_sc_hi), float4(-gate_d_m_hi));

            const uchar4 up_q_lo = uchar4(
                up_qbytes.x & 0x0F,
                up_qbytes.y & 0x0F,
                up_qbytes.z & 0x0F,
                up_qbytes.w & 0x0F
            );
            const uchar4 up_q_hi = uchar4(
                up_qbytes.x >> 4,
                up_qbytes.y >> 4,
                up_qbytes.z >> 4,
                up_qbytes.w >> 4
            );
            const float4 up_lo_vals = fma(float4(up_q_lo), float4(up_d_sc_lo), float4(-up_d_m_lo));
            const float4 up_hi_vals = fma(float4(up_q_hi), float4(up_d_sc_hi), float4(-up_d_m_hi));

            const float4 x0_lo = *(device const float4*)(x0 + col_lo);
            const float4 x0_hi = *(device const float4*)(x0 + col_hi);
            const float4 x1_lo = *(device const float4*)(x1 + col_lo);
            const float4 x1_hi = *(device const float4*)(x1 + col_hi);
            const float4 x2_lo = *(device const float4*)(x2 + col_lo);
            const float4 x2_hi = *(device const float4*)(x2 + col_hi);

            gate_acc.x += dot(gate_lo_vals, x0_lo) + dot(gate_hi_vals, x0_hi);
            gate_acc.y += dot(gate_lo_vals, x1_lo) + dot(gate_hi_vals, x1_hi);
            gate_acc.z += dot(gate_lo_vals, x2_lo) + dot(gate_hi_vals, x2_hi);
            up_acc.x += dot(up_lo_vals, x0_lo) + dot(up_hi_vals, x0_hi);
            up_acc.y += dot(up_lo_vals, x1_lo) + dot(up_hi_vals, x1_hi);
            up_acc.z += dot(up_lo_vals, x2_lo) + dot(up_hi_vals, x2_hi);
        }

        const float gate0 = simd_sum(gate_acc.x);
        const float gate1 = simd_sum(gate_acc.y);
        const float gate2 = simd_sum(gate_acc.z);
        const float up0 = simd_sum(up_acc.x);
        const float up1 = simd_sum(up_acc.y);
        const float up2 = simd_sum(up_acc.z);

        device float* y_base = activatedY + (p.y_offset / 4u);
        if (tid == 0u) {
            y_base[route0 * p.M + row] = geglu(gate0, up0);
            y_base[route1 * p.M + row] = geglu(gate1, up1);
            y_base[route2 * p.M + row] = geglu(gate2, up2);
        }
        return;
    }

    if (p.enable_exact5 != 0u && remaining == 5u) {
        const uint route1 = expert_ids[packed_base + 1u];
        const uint route2 = expert_ids[packed_base + 2u];
        const uint route3 = expert_ids[packed_base + 3u];
        const uint route4 = expert_ids[packed_base + 4u];

        device const float* x1 = x_base + (route1 / x_div) * p.K;
        device const float* x2 = x_base + (route2 / x_div) * p.K;
        device const float* x3 = x_base + (route3 / x_div) * p.K;
        device const float* x4 = x_base + (route4 / x_div) * p.K;

        float4 gate_acc0 = float4(0.0f);
        float4 up_acc0 = float4(0.0f);
        float gate_acc4 = 0.0f;
        float up_acc4 = 0.0f;

        for (uint b = 0u; b < nb; b++) {
            device const uchar* gate_block = gate_src + b * bpb;
            device const uchar* up_block = up_src + b * bpb;

            const float gate_d = float(as_type<half>(*(device const ushort*)(gate_block)));
            const float gate_dmin = float(as_type<half>(*(device const ushort*)(gate_block + 2)));
            device const uchar* gate_scales = gate_block + 4;
            device const uchar* gate_quants = gate_block + 16;

            const float up_d = float(as_type<half>(*(device const ushort*)(up_block)));
            const float up_dmin = float(as_type<half>(*(device const ushort*)(up_block + 2)));
            device const uchar* up_scales = up_block + 4;
            device const uchar* up_quants = up_block + 16;

            const uint byte_off = tid * 4u;
            const uint j = byte_off / 32u;
            const uint local_off = byte_off % 32u;

            const uchar4 gate_qbytes = *(device const uchar4*)(gate_quants + byte_off);
            const float2 gate_sm_lo = get_scale_min_k4(j * 2u, gate_scales);
            const float2 gate_sm_hi = get_scale_min_k4(j * 2u + 1u, gate_scales);

            const float gate_d_sc_lo = gate_d * gate_sm_lo.x;
            const float gate_d_m_lo = gate_dmin * gate_sm_lo.y;
            const float gate_d_sc_hi = gate_d * gate_sm_hi.x;
            const float gate_d_m_hi = gate_dmin * gate_sm_hi.y;

            const uchar4 up_qbytes = *(device const uchar4*)(up_quants + byte_off);
            const float2 up_sm_lo = get_scale_min_k4(j * 2u, up_scales);
            const float2 up_sm_hi = get_scale_min_k4(j * 2u + 1u, up_scales);

            const float up_d_sc_lo = up_d * up_sm_lo.x;
            const float up_d_m_lo = up_dmin * up_sm_lo.y;
            const float up_d_sc_hi = up_d * up_sm_hi.x;
            const float up_d_m_hi = up_dmin * up_sm_hi.y;

            const uint col_lo = b * 256u + j * 64u + local_off;
            const uint col_hi = col_lo + 32u;

            const uchar4 gate_q_lo = uchar4(
                gate_qbytes.x & 0x0F,
                gate_qbytes.y & 0x0F,
                gate_qbytes.z & 0x0F,
                gate_qbytes.w & 0x0F
            );
            const uchar4 gate_q_hi = uchar4(
                gate_qbytes.x >> 4,
                gate_qbytes.y >> 4,
                gate_qbytes.z >> 4,
                gate_qbytes.w >> 4
            );
            const float4 gate_lo_vals = fma(float4(gate_q_lo), float4(gate_d_sc_lo), float4(-gate_d_m_lo));
            const float4 gate_hi_vals = fma(float4(gate_q_hi), float4(gate_d_sc_hi), float4(-gate_d_m_hi));

            const uchar4 up_q_lo = uchar4(
                up_qbytes.x & 0x0F,
                up_qbytes.y & 0x0F,
                up_qbytes.z & 0x0F,
                up_qbytes.w & 0x0F
            );
            const uchar4 up_q_hi = uchar4(
                up_qbytes.x >> 4,
                up_qbytes.y >> 4,
                up_qbytes.z >> 4,
                up_qbytes.w >> 4
            );
            const float4 up_lo_vals = fma(float4(up_q_lo), float4(up_d_sc_lo), float4(-up_d_m_lo));
            const float4 up_hi_vals = fma(float4(up_q_hi), float4(up_d_sc_hi), float4(-up_d_m_hi));

            const float4 x0_lo = *(device const float4*)(x0 + col_lo);
            const float4 x0_hi = *(device const float4*)(x0 + col_hi);
            const float4 x1_lo = *(device const float4*)(x1 + col_lo);
            const float4 x1_hi = *(device const float4*)(x1 + col_hi);
            const float4 x2_lo = *(device const float4*)(x2 + col_lo);
            const float4 x2_hi = *(device const float4*)(x2 + col_hi);
            const float4 x3_lo = *(device const float4*)(x3 + col_lo);
            const float4 x3_hi = *(device const float4*)(x3 + col_hi);
            const float4 x4_lo = *(device const float4*)(x4 + col_lo);
            const float4 x4_hi = *(device const float4*)(x4 + col_hi);

            gate_acc0.x += dot(gate_lo_vals, x0_lo) + dot(gate_hi_vals, x0_hi);
            gate_acc0.y += dot(gate_lo_vals, x1_lo) + dot(gate_hi_vals, x1_hi);
            gate_acc0.z += dot(gate_lo_vals, x2_lo) + dot(gate_hi_vals, x2_hi);
            gate_acc0.w += dot(gate_lo_vals, x3_lo) + dot(gate_hi_vals, x3_hi);
            gate_acc4 += dot(gate_lo_vals, x4_lo) + dot(gate_hi_vals, x4_hi);

            up_acc0.x += dot(up_lo_vals, x0_lo) + dot(up_hi_vals, x0_hi);
            up_acc0.y += dot(up_lo_vals, x1_lo) + dot(up_hi_vals, x1_hi);
            up_acc0.z += dot(up_lo_vals, x2_lo) + dot(up_hi_vals, x2_hi);
            up_acc0.w += dot(up_lo_vals, x3_lo) + dot(up_hi_vals, x3_hi);
            up_acc4 += dot(up_lo_vals, x4_lo) + dot(up_hi_vals, x4_hi);
        }

        const float gate0 = simd_sum(gate_acc0.x);
        const float gate1 = simd_sum(gate_acc0.y);
        const float gate2 = simd_sum(gate_acc0.z);
        const float gate3 = simd_sum(gate_acc0.w);
        const float gate4 = simd_sum(gate_acc4);

        const float up0 = simd_sum(up_acc0.x);
        const float up1 = simd_sum(up_acc0.y);
        const float up2 = simd_sum(up_acc0.z);
        const float up3 = simd_sum(up_acc0.w);
        const float up4 = simd_sum(up_acc4);

        device float* y_base = activatedY + (p.y_offset / 4u);
        if (tid == 0u) {
            y_base[route0 * p.M + row] = geglu(gate0, up0);
            y_base[route1 * p.M + row] = geglu(gate1, up1);
            y_base[route2 * p.M + row] = geglu(gate2, up2);
            y_base[route3 * p.M + row] = geglu(gate3, up3);
            y_base[route4 * p.M + row] = geglu(gate4, up4);
        }
        return;
    }

    if (remaining >= NUM_COLS) {
        const uint route1 = expert_ids[packed_base + 1u];
        const uint route2 = expert_ids[packed_base + 2u];
        const uint route3 = expert_ids[packed_base + 3u];
        const uint route4 = expert_ids[packed_base + 4u];
        const uint route5 = expert_ids[packed_base + 5u];
        const uint route6 = expert_ids[packed_base + 6u];
        const uint route7 = expert_ids[packed_base + 7u];

        device const float* x1 = x_base + (route1 / x_div) * p.K;
        device const float* x2 = x_base + (route2 / x_div) * p.K;
        device const float* x3 = x_base + (route3 / x_div) * p.K;
        device const float* x4 = x_base + (route4 / x_div) * p.K;
        device const float* x5 = x_base + (route5 / x_div) * p.K;
        device const float* x6 = x_base + (route6 / x_div) * p.K;
        device const float* x7 = x_base + (route7 / x_div) * p.K;

        float4 gate_acc0 = float4(0.0f);
        float4 gate_acc1 = float4(0.0f);
        float4 up_acc0 = float4(0.0f);
        float4 up_acc1 = float4(0.0f);

        for (uint b = 0u; b < nb; b++) {
            device const uchar* gate_block = gate_src + b * bpb;
            device const uchar* up_block = up_src + b * bpb;

            const float gate_d = float(as_type<half>(*(device const ushort*)(gate_block)));
            const float gate_dmin = float(as_type<half>(*(device const ushort*)(gate_block + 2)));
            device const uchar* gate_scales = gate_block + 4;
            device const uchar* gate_quants = gate_block + 16;

            const float up_d = float(as_type<half>(*(device const ushort*)(up_block)));
            const float up_dmin = float(as_type<half>(*(device const ushort*)(up_block + 2)));
            device const uchar* up_scales = up_block + 4;
            device const uchar* up_quants = up_block + 16;

            const uint byte_off = tid * 4u;
            const uint j = byte_off / 32u;
            const uint local_off = byte_off % 32u;

            const uchar4 gate_qbytes = *(device const uchar4*)(gate_quants + byte_off);
            const float2 gate_sm_lo = get_scale_min_k4(j * 2u, gate_scales);
            const float2 gate_sm_hi = get_scale_min_k4(j * 2u + 1u, gate_scales);

            const float gate_d_sc_lo = gate_d * gate_sm_lo.x;
            const float gate_d_m_lo = gate_dmin * gate_sm_lo.y;
            const float gate_d_sc_hi = gate_d * gate_sm_hi.x;
            const float gate_d_m_hi = gate_dmin * gate_sm_hi.y;

            const uchar4 up_qbytes = *(device const uchar4*)(up_quants + byte_off);
            const float2 up_sm_lo = get_scale_min_k4(j * 2u, up_scales);
            const float2 up_sm_hi = get_scale_min_k4(j * 2u + 1u, up_scales);

            const float up_d_sc_lo = up_d * up_sm_lo.x;
            const float up_d_m_lo = up_dmin * up_sm_lo.y;
            const float up_d_sc_hi = up_d * up_sm_hi.x;
            const float up_d_m_hi = up_dmin * up_sm_hi.y;

            const uint col_lo = b * 256u + j * 64u + local_off;
            const uint col_hi = col_lo + 32u;

            const uchar4 gate_q_lo = uchar4(
                gate_qbytes.x & 0x0F,
                gate_qbytes.y & 0x0F,
                gate_qbytes.z & 0x0F,
                gate_qbytes.w & 0x0F
            );
            const uchar4 gate_q_hi = uchar4(
                gate_qbytes.x >> 4,
                gate_qbytes.y >> 4,
                gate_qbytes.z >> 4,
                gate_qbytes.w >> 4
            );
            const float4 gate_lo_vals = fma(float4(gate_q_lo), float4(gate_d_sc_lo), float4(-gate_d_m_lo));
            const float4 gate_hi_vals = fma(float4(gate_q_hi), float4(gate_d_sc_hi), float4(-gate_d_m_hi));

            const uchar4 up_q_lo = uchar4(
                up_qbytes.x & 0x0F,
                up_qbytes.y & 0x0F,
                up_qbytes.z & 0x0F,
                up_qbytes.w & 0x0F
            );
            const uchar4 up_q_hi = uchar4(
                up_qbytes.x >> 4,
                up_qbytes.y >> 4,
                up_qbytes.z >> 4,
                up_qbytes.w >> 4
            );
            const float4 up_lo_vals = fma(float4(up_q_lo), float4(up_d_sc_lo), float4(-up_d_m_lo));
            const float4 up_hi_vals = fma(float4(up_q_hi), float4(up_d_sc_hi), float4(-up_d_m_hi));

            const float4 x0_lo = *(device const float4*)(x0 + col_lo);
            const float4 x0_hi = *(device const float4*)(x0 + col_hi);
            const float4 x1_lo = *(device const float4*)(x1 + col_lo);
            const float4 x1_hi = *(device const float4*)(x1 + col_hi);
            const float4 x2_lo = *(device const float4*)(x2 + col_lo);
            const float4 x2_hi = *(device const float4*)(x2 + col_hi);
            const float4 x3_lo = *(device const float4*)(x3 + col_lo);
            const float4 x3_hi = *(device const float4*)(x3 + col_hi);
            const float4 x4_lo = *(device const float4*)(x4 + col_lo);
            const float4 x4_hi = *(device const float4*)(x4 + col_hi);
            const float4 x5_lo = *(device const float4*)(x5 + col_lo);
            const float4 x5_hi = *(device const float4*)(x5 + col_hi);
            const float4 x6_lo = *(device const float4*)(x6 + col_lo);
            const float4 x6_hi = *(device const float4*)(x6 + col_hi);
            const float4 x7_lo = *(device const float4*)(x7 + col_lo);
            const float4 x7_hi = *(device const float4*)(x7 + col_hi);

            gate_acc0.x += dot(gate_lo_vals, x0_lo) + dot(gate_hi_vals, x0_hi);
            gate_acc0.y += dot(gate_lo_vals, x1_lo) + dot(gate_hi_vals, x1_hi);
            gate_acc0.z += dot(gate_lo_vals, x2_lo) + dot(gate_hi_vals, x2_hi);
            gate_acc0.w += dot(gate_lo_vals, x3_lo) + dot(gate_hi_vals, x3_hi);
            gate_acc1.x += dot(gate_lo_vals, x4_lo) + dot(gate_hi_vals, x4_hi);
            gate_acc1.y += dot(gate_lo_vals, x5_lo) + dot(gate_hi_vals, x5_hi);
            gate_acc1.z += dot(gate_lo_vals, x6_lo) + dot(gate_hi_vals, x6_hi);
            gate_acc1.w += dot(gate_lo_vals, x7_lo) + dot(gate_hi_vals, x7_hi);

            up_acc0.x += dot(up_lo_vals, x0_lo) + dot(up_hi_vals, x0_hi);
            up_acc0.y += dot(up_lo_vals, x1_lo) + dot(up_hi_vals, x1_hi);
            up_acc0.z += dot(up_lo_vals, x2_lo) + dot(up_hi_vals, x2_hi);
            up_acc0.w += dot(up_lo_vals, x3_lo) + dot(up_hi_vals, x3_hi);
            up_acc1.x += dot(up_lo_vals, x4_lo) + dot(up_hi_vals, x4_hi);
            up_acc1.y += dot(up_lo_vals, x5_lo) + dot(up_hi_vals, x5_hi);
            up_acc1.z += dot(up_lo_vals, x6_lo) + dot(up_hi_vals, x6_hi);
            up_acc1.w += dot(up_lo_vals, x7_lo) + dot(up_hi_vals, x7_hi);
        }

        const float gate0 = simd_sum(gate_acc0.x);
        const float gate1 = simd_sum(gate_acc0.y);
        const float gate2 = simd_sum(gate_acc0.z);
        const float gate3 = simd_sum(gate_acc0.w);
        const float gate4 = simd_sum(gate_acc1.x);
        const float gate5 = simd_sum(gate_acc1.y);
        const float gate6 = simd_sum(gate_acc1.z);
        const float gate7 = simd_sum(gate_acc1.w);

        const float up0 = simd_sum(up_acc0.x);
        const float up1 = simd_sum(up_acc0.y);
        const float up2 = simd_sum(up_acc0.z);
        const float up3 = simd_sum(up_acc0.w);
        const float up4 = simd_sum(up_acc1.x);
        const float up5 = simd_sum(up_acc1.y);
        const float up6 = simd_sum(up_acc1.z);
        const float up7 = simd_sum(up_acc1.w);

        device float* y_base = activatedY + (p.y_offset / 4u);
        if (tid == 0u) {
            y_base[route0 * p.M + row] = geglu(gate0, up0);
            y_base[route1 * p.M + row] = geglu(gate1, up1);
            y_base[route2 * p.M + row] = geglu(gate2, up2);
            y_base[route3 * p.M + row] = geglu(gate3, up3);
            y_base[route4 * p.M + row] = geglu(gate4, up4);
            y_base[route5 * p.M + row] = geglu(gate5, up5);
            y_base[route6 * p.M + row] = geglu(gate6, up6);
            y_base[route7 * p.M + row] = geglu(gate7, up7);
        }
        return;
    }

    const bool active1 = remaining > 1u;
    const bool active2 = remaining > 2u;
    const bool active3 = remaining > 3u;
    const bool active4 = remaining > 4u;
    const bool active5 = remaining > 5u;
    const bool active6 = remaining > 6u;
    const bool active7 = remaining > 7u;

    const uint route1 = active1 ? expert_ids[packed_base + 1u] : 0u;
    const uint route2 = active2 ? expert_ids[packed_base + 2u] : 0u;
    const uint route3 = active3 ? expert_ids[packed_base + 3u] : 0u;
    const uint route4 = active4 ? expert_ids[packed_base + 4u] : 0u;
    const uint route5 = active5 ? expert_ids[packed_base + 5u] : 0u;
    const uint route6 = active6 ? expert_ids[packed_base + 6u] : 0u;
    const uint route7 = active7 ? expert_ids[packed_base + 7u] : 0u;

    device const float* x1 = x_base + (route1 / x_div) * p.K;
    device const float* x2 = x_base + (route2 / x_div) * p.K;
    device const float* x3 = x_base + (route3 / x_div) * p.K;
    device const float* x4 = x_base + (route4 / x_div) * p.K;
    device const float* x5 = x_base + (route5 / x_div) * p.K;
    device const float* x6 = x_base + (route6 / x_div) * p.K;
    device const float* x7 = x_base + (route7 / x_div) * p.K;

    float4 gate_acc0 = float4(0.0f);
    float4 gate_acc1 = float4(0.0f);
    float4 up_acc0 = float4(0.0f);
    float4 up_acc1 = float4(0.0f);

    for (uint b = 0u; b < nb; b++) {
        device const uchar* gate_block = gate_src + b * bpb;
        device const uchar* up_block = up_src + b * bpb;

        const float gate_d = float(as_type<half>(*(device const ushort*)(gate_block)));
        const float gate_dmin = float(as_type<half>(*(device const ushort*)(gate_block + 2)));
        device const uchar* gate_scales = gate_block + 4;
        device const uchar* gate_quants = gate_block + 16;

        const float up_d = float(as_type<half>(*(device const ushort*)(up_block)));
        const float up_dmin = float(as_type<half>(*(device const ushort*)(up_block + 2)));
        device const uchar* up_scales = up_block + 4;
        device const uchar* up_quants = up_block + 16;

        const uint byte_off = tid * 4u;
        const uint j = byte_off / 32u;
        const uint local_off = byte_off % 32u;

        const uchar4 gate_qbytes = *(device const uchar4*)(gate_quants + byte_off);
        const float2 gate_sm_lo = get_scale_min_k4(j * 2u, gate_scales);
        const float2 gate_sm_hi = get_scale_min_k4(j * 2u + 1u, gate_scales);

        const float gate_d_sc_lo = gate_d * gate_sm_lo.x;
        const float gate_d_m_lo = gate_dmin * gate_sm_lo.y;
        const float gate_d_sc_hi = gate_d * gate_sm_hi.x;
        const float gate_d_m_hi = gate_dmin * gate_sm_hi.y;

        const uchar4 up_qbytes = *(device const uchar4*)(up_quants + byte_off);
        const float2 up_sm_lo = get_scale_min_k4(j * 2u, up_scales);
        const float2 up_sm_hi = get_scale_min_k4(j * 2u + 1u, up_scales);

        const float up_d_sc_lo = up_d * up_sm_lo.x;
        const float up_d_m_lo = up_dmin * up_sm_lo.y;
        const float up_d_sc_hi = up_d * up_sm_hi.x;
        const float up_d_m_hi = up_dmin * up_sm_hi.y;

        const uint col_lo = b * 256u + j * 64u + local_off;
        const uint col_hi = col_lo + 32u;

        const uchar4 gate_q_lo = uchar4(
            gate_qbytes.x & 0x0F,
            gate_qbytes.y & 0x0F,
            gate_qbytes.z & 0x0F,
            gate_qbytes.w & 0x0F
        );
        const uchar4 gate_q_hi = uchar4(
            gate_qbytes.x >> 4,
            gate_qbytes.y >> 4,
            gate_qbytes.z >> 4,
            gate_qbytes.w >> 4
        );
        const float4 gate_lo_vals = fma(float4(gate_q_lo), float4(gate_d_sc_lo), float4(-gate_d_m_lo));
        const float4 gate_hi_vals = fma(float4(gate_q_hi), float4(gate_d_sc_hi), float4(-gate_d_m_hi));

        const uchar4 up_q_lo = uchar4(
            up_qbytes.x & 0x0F,
            up_qbytes.y & 0x0F,
            up_qbytes.z & 0x0F,
            up_qbytes.w & 0x0F
        );
        const uchar4 up_q_hi = uchar4(
            up_qbytes.x >> 4,
            up_qbytes.y >> 4,
            up_qbytes.z >> 4,
            up_qbytes.w >> 4
        );
        const float4 up_lo_vals = fma(float4(up_q_lo), float4(up_d_sc_lo), float4(-up_d_m_lo));
        const float4 up_hi_vals = fma(float4(up_q_hi), float4(up_d_sc_hi), float4(-up_d_m_hi));

        if (active0) {
            const float4 x_lo = *(device const float4*)(x0 + col_lo);
            const float4 x_hi = *(device const float4*)(x0 + col_hi);
            gate_acc0.x += dot(gate_lo_vals, x_lo) + dot(gate_hi_vals, x_hi);
            up_acc0.x += dot(up_lo_vals, x_lo) + dot(up_hi_vals, x_hi);
        }
        if (active1) {
            const float4 x_lo = *(device const float4*)(x1 + col_lo);
            const float4 x_hi = *(device const float4*)(x1 + col_hi);
            gate_acc0.y += dot(gate_lo_vals, x_lo) + dot(gate_hi_vals, x_hi);
            up_acc0.y += dot(up_lo_vals, x_lo) + dot(up_hi_vals, x_hi);
        }
        if (active2) {
            const float4 x_lo = *(device const float4*)(x2 + col_lo);
            const float4 x_hi = *(device const float4*)(x2 + col_hi);
            gate_acc0.z += dot(gate_lo_vals, x_lo) + dot(gate_hi_vals, x_hi);
            up_acc0.z += dot(up_lo_vals, x_lo) + dot(up_hi_vals, x_hi);
        }
        if (active3) {
            const float4 x_lo = *(device const float4*)(x3 + col_lo);
            const float4 x_hi = *(device const float4*)(x3 + col_hi);
            gate_acc0.w += dot(gate_lo_vals, x_lo) + dot(gate_hi_vals, x_hi);
            up_acc0.w += dot(up_lo_vals, x_lo) + dot(up_hi_vals, x_hi);
        }
        if (active4) {
            const float4 x_lo = *(device const float4*)(x4 + col_lo);
            const float4 x_hi = *(device const float4*)(x4 + col_hi);
            gate_acc1.x += dot(gate_lo_vals, x_lo) + dot(gate_hi_vals, x_hi);
            up_acc1.x += dot(up_lo_vals, x_lo) + dot(up_hi_vals, x_hi);
        }
        if (active5) {
            const float4 x_lo = *(device const float4*)(x5 + col_lo);
            const float4 x_hi = *(device const float4*)(x5 + col_hi);
            gate_acc1.y += dot(gate_lo_vals, x_lo) + dot(gate_hi_vals, x_hi);
            up_acc1.y += dot(up_lo_vals, x_lo) + dot(up_hi_vals, x_hi);
        }
        if (active6) {
            const float4 x_lo = *(device const float4*)(x6 + col_lo);
            const float4 x_hi = *(device const float4*)(x6 + col_hi);
            gate_acc1.z += dot(gate_lo_vals, x_lo) + dot(gate_hi_vals, x_hi);
            up_acc1.z += dot(up_lo_vals, x_lo) + dot(up_hi_vals, x_hi);
        }
        if (active7) {
            const float4 x_lo = *(device const float4*)(x7 + col_lo);
            const float4 x_hi = *(device const float4*)(x7 + col_hi);
            gate_acc1.w += dot(gate_lo_vals, x_lo) + dot(gate_hi_vals, x_hi);
            up_acc1.w += dot(up_lo_vals, x_lo) + dot(up_hi_vals, x_hi);
        }
    }

    const float gate0 = simd_sum(gate_acc0.x);
    const float gate1 = simd_sum(gate_acc0.y);
    const float gate2 = simd_sum(gate_acc0.z);
    const float gate3 = simd_sum(gate_acc0.w);
    const float gate4 = simd_sum(gate_acc1.x);
    const float gate5 = simd_sum(gate_acc1.y);
    const float gate6 = simd_sum(gate_acc1.z);
    const float gate7 = simd_sum(gate_acc1.w);

    const float up0 = simd_sum(up_acc0.x);
    const float up1 = simd_sum(up_acc0.y);
    const float up2 = simd_sum(up_acc0.z);
    const float up3 = simd_sum(up_acc0.w);
    const float up4 = simd_sum(up_acc1.x);
    const float up5 = simd_sum(up_acc1.y);
    const float up6 = simd_sum(up_acc1.z);
    const float up7 = simd_sum(up_acc1.w);

    device float* y_base = activatedY + (p.y_offset / 4u);
    if (tid == 0u) {
        if (active0) y_base[route0 * p.M + row] = geglu(gate0, up0);
        if (active1) y_base[route1 * p.M + row] = geglu(gate1, up1);
        if (active2) y_base[route2 * p.M + row] = geglu(gate2, up2);
        if (active3) y_base[route3 * p.M + row] = geglu(gate3, up3);
        if (active4) y_base[route4 * p.M + row] = geglu(gate4, up4);
        if (active5) y_base[route5 * p.M + row] = geglu(gate5, up5);
        if (active6) y_base[route6 * p.M + row] = geglu(gate6, up6);
        if (active7) y_base[route7 * p.M + row] = geglu(gate7, up7);
    }
}
