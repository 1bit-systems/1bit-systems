#include <metal_stdlib>
using namespace metal;

// Q8_0 routed MoE DMMV for the rare Gemma expert-down layer stored as Q8_0.
//
// This is the small-batch branch of the reference implementation's Metal `mul_mat_id` idea: keep
// selected expert ids compact on device, then have each expert-slot workgroup
// read `expert_ids[slot]` to choose the weight slice. The inner dot follows
// `kernel_mul_mv_q8_0_f32_impl` / this repo's `dmmv_q8_0.metal` two-row
// simdgroup shape rather than the grouped matrix-matrix path, because decode
// has one token and eight selected experts.

struct MoeDmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint expert_stride;
    uint x_expert_stride;
    uint x_offset;
    uint y_offset;
};

kernel void main0(
    device const uchar* W                     [[buffer(0)]],
    constant MoeDmmvPush& p                   [[buffer(1)]],
    device const float* X                     [[buffer(2)]],
    device float* Y                           [[buffer(3)]],
    device const uint* expert_ids             [[buffer(4)]],
    uint3 tg_pos                              [[threadgroup_position_in_grid]],
    uint sgid                                 [[simdgroup_index_in_threadgroup]],
    uint lane                                 [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg                    [[simdgroups_per_threadgroup]]
) {
    const uint expert_slot = tg_pos.y;
    const uint expert_id = expert_ids[expert_slot];
    const uint base_row = (tg_pos.x * simdgroups_per_tg + sgid) * 2u;
    if (base_row >= p.M) {
        return;
    }

    const uint blocks_per_row = p.K >> 5;
    const ulong row_bytes = ulong(blocks_per_row) * 34ull;
    const ulong expert_base = ulong(p.a_offset) + ulong(expert_id) * ulong(p.expert_stride);
    device const uchar* row0 = W + expert_base + ulong(base_row) * row_bytes;
    device const uchar* row1 = row0 + row_bytes;

    device const float* input = X + (p.x_offset >> 2) + expert_slot * p.x_expert_stride;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const bool has_next = base_row + 1u < p.M;

    for (uint bi = lane; bi < blocks_per_row; bi += 32u) {
        device const uchar* blk0 = row0 + bi * 34u;
        device const uchar* blk1 = has_next ? (row1 + bi * 34u) : blk0;
        const float s0 = float(as_type<half>(*(device const ushort*)(blk0)));
        const float s1 = has_next ? float(as_type<half>(*(device const ushort*)(blk1))) : 0.0f;
        device const packed_char4* q0 = (device const packed_char4*)(blk0 + 2u);
        device const packed_char4* q1 = (device const packed_char4*)(blk1 + 2u);
        const uint x_base = bi << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const float4 x = *(device const float4*)(input + x_base + (vi << 2));
            acc0 = fma(s0, dot(float4(char4(q0[vi])), x), acc0);
            if (has_next) {
                acc1 = fma(s1, dot(float4(char4(q1[vi])), x), acc1);
            }
        }
    }

    const float sum0 = simd_sum(acc0);
    const float sum1 = simd_sum(acc1);
    device float* output = Y + (p.y_offset >> 2) + expert_slot * p.M;
    if (lane < 2u && (lane == 0u || has_next)) {
        output[base_row + lane] = (lane == 0u) ? sum0 : sum1;
    }
}
