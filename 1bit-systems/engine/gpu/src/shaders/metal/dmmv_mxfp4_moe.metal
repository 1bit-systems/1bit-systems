#include <metal_stdlib>
using namespace metal;

// Batched MXFP4 MoE DMMV. grid.y selects the route slot, and the route slot
// maps to a real expert id via the routing buffer.

struct MoeDmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint expert_stride;
    uint x_expert_stride;
    uint x_offset;
    uint y_offset;
};

constant float kvalues_mxfp4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline float e8m0_to_fp32(uchar x) {
    uint bits;
    if (x == 0u) {
        bits = 0x00400000u;
    } else {
        bits = uint(x) << 23;
    }
    return as_type<float>(bits);
}

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant MoeDmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    device const uint* expert_ids [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 local_pos [[thread_position_in_threadgroup]]
) {
    threadgroup float x_cache[4096];

    const uint local_id = local_pos.x;
    const uint expert_slot = tg_pos.y;
    const uint expert_id = expert_ids[expert_slot];
    device const float* input = X + (p.x_offset / 4) + expert_slot * p.x_expert_stride;

    for (uint i = local_id; i < p.K; i += 64u) {
        x_cache[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tg_pos.x * 64u + local_id;
    if (row >= p.M) return;

    const uint blocks_per_row = p.K / 32u;
    const ulong row_bytes = ulong(blocks_per_row) * 17ul;
    const ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    device const uchar* src = W + expert_base + ulong(row) * row_bytes;

    float sum = 0.0f;
    for (uint b = 0; b < blocks_per_row; b++) {
        device const uchar* block = src + ulong(b) * 17ul;
        const float d = e8m0_to_fp32(block[0]);
        device const uchar* qs = block + 1;
        const uint base = b * 32u;

        for (uint j = 0; j < 16u; j++) {
            const uchar q = qs[j];
            const float v_lo = d * kvalues_mxfp4[q & 0x0Fu];
            const float v_hi = d * kvalues_mxfp4[q >> 4];
            sum += v_lo * x_cache[base + j] + v_hi * x_cache[base + 16u + j];
        }
    }

    device float* out = Y + (p.y_offset / 4) + expert_slot * p.M;
    out[row] = sum;
}
