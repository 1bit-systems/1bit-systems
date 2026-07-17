#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Native Metal Q8_0 DMMV — nr=2 multi-row, barrier-free, L1-cached X reads.
//
// Q8_0 is the dominant path for SSM projections and lm_head on the target
// Qwen3.6-35B-A3B model (72.5% of all DMMV data).
//
// Adapted from the reference implementation kernel_mul_mv_q8_0_f32 (ggml-metal.metal) which uses
// N_R0_Q8_0 = 2: each simdgroup processes TWO output rows simultaneously,
// sharing the L1-cached X vector (at most 16 KiB for K<=4096).  This doubles
// useful compute per X fetch, improving pipeline utilization.
//
// Q8_0 block layout (34 bytes, 32 elements):
//   [0..1]   d  (float16) scale
//   [2..33]  qs (32 x int8)

kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    const uint base_row = (tg_id * simdgroups_per_tg + sg_idx) * 2u;
    if (base_row >= p.M) return;

    device const float* input = X + (p.x_offset >> 2);
    device float* output = Y + (p.y_offset >> 2);

    const uint blocks_per_row = p.K >> 5;
    const ulong row_bytes = ulong(blocks_per_row) * 34ull;
    device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
    device const uchar* row1 = row0 + row_bytes;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (uint bi = lane; bi < blocks_per_row; bi += 32u) {
        device const uchar* blk0 = row0 + bi * 34u;
        device const uchar* blk1 = row1 + bi * 34u;
        const float s0 = float(as_type<half>(*(device const ushort*)(blk0)));
        const float s1 = float(as_type<half>(*(device const ushort*)(blk1)));
        device const packed_char4* q0 = (device const packed_char4*)(blk0 + 2u);
        device const packed_char4* q1 = (device const packed_char4*)(blk1 + 2u);
        const uint x_base = bi << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const float4 x = *(device const float4*)(input + x_base + (vi << 2));
            acc0 = fma(s0, dot(float4(char4(q0[vi])), x), acc0);
            acc1 = fma(s1, dot(float4(char4(q1[vi])), x), acc1);
        }
    }

    // Distribute the two row writes across lanes 0 and 1 so the pair of
    // output stores at base_row..base_row+1 issues as one coalesced 8-byte
    // transaction (mirrors cycle-43 dmmv_q8_0_pair_swiglu, cycle-47
    // dmmv_q8_0_repacked_k4096_nr2_qwen, and cycle-49
    // dmmv_q8_0_repacked_k2048_dual_nr2_qwen). Hoist simd_sum(acc1) out of the
    // conditional — simd_sum needs uniform participation, and `has_next` is
    // uniform within a simdgroup (depends only on base_row), so this is
    // bit-equivalent. Lane 1's write is gated on has_next for the boundary
    // simdgroup.
    //
    // Applies to the production Qwen3.6 SSM qkv path: M=8192 K=2048 (17.93
    // GiB per request, 1080 calls — the single biggest Q8 traffic line, hit
    // every decode token across all 30 SSM layers via the tg128 nr=2 path
    // selected in cachedDmmvPipeline_q8 for preferLlamaQ8SmallThreadgroupForQwenSsm).
    const float sum0 = simd_sum(acc0);
    const float sum1 = simd_sum(acc1);
    const bool has_next = base_row + 1u < p.M;
    if (lane < 2u && (lane == 0u || has_next)) {
        output[base_row + lane] = (lane == 0u) ? sum0 : sum1;
    }
}
