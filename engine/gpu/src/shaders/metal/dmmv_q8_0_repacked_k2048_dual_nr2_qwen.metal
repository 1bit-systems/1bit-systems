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

// Exact Qwen3.6 SSM dual Q8_0 repacked DMMV for K=2048.
//
// This keeps the accepted reference-style two-row Q8 geometry from
// dmmv_q8_0_repacked_k2048_nr2_qwen.metal, but lets sibling SSM QKV and gate
// projections sharing the same norm row run as one encoder dispatch. The
// simdgroup count is now taken from the dispatcher (matches the K=2048 quad
// and Qwen-specific repacked kernels) so the dispatch may pick a larger
// threadgroup when occupancy per workgroup would otherwise be too low.
kernel void main0(
    constant DualQ8DmmvPush& p [[buffer(0)]],
    device const uchar* W0 [[buffer(1)]],
    device const uchar* W1 [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* Y0 [[buffer(4)]],
    device float* Y1 [[buffer(5)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    const uint base_pair = tg_id * simdgroups_per_tg + sg_idx;
    const uint base_row = base_pair * 2u;
    const uint total_rows = p.M0 + p.M1;
    if (base_row >= total_rows) {
        return;
    }

    device const float* input = X + (p.x_offset >> 2);

    constexpr ulong group_bytes = 1088ul;
    constexpr ulong row_bytes = 2176ul;

    const bool first = base_row < p.M0;
    const uint row = first ? base_row : (base_row - p.M0);
    device const uchar* weights = first ? W0 : W1;
    device float* output = first ? (Y0 + (p.y0_offset >> 2)) : (Y1 + (p.y1_offset >> 2));
    const uint a_offset = first ? p.a0_offset : p.a1_offset;

    device const uchar* row0 = weights + a_offset + ulong(row) * row_bytes;
    device const uchar* row1 = row0 + row_bytes;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    #pragma unroll
    for (uint gi = 0u; gi < 2u; ++gi) {
        device const uchar* g0 = row0 + ulong(gi) * group_bytes;
        device const uchar* g1 = row1 + ulong(gi) * group_bytes;

        const float s0 = float(as_type<half>(*(device const ushort*)(g0 + lane * 2u)));
        const float s1 = float(as_type<half>(*(device const ushort*)(g1 + lane * 2u)));
        const uint x_base = (gi * 32u + lane) << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const uint qo = 64u + vi * 128u + lane * 4u;
            const char4 q0 = as_type<char4>(*(device const int*)(g0 + qo));
            const char4 q1 = as_type<char4>(*(device const int*)(g1 + qo));
            const float4 x = *(device const float4*)(input + x_base + (vi << 2));

            acc0 = fma(s0, dot(float4(q0), x), acc0);
            acc1 = fma(s1, dot(float4(q1), x), acc1);
        }
    }

    // Cycle ~82: pack the two final-reduction `simd_sum` calls into one
    // `simd_sum(float2)` — Apple9 lowers vector `simd_sum` to a single
    // log2(32)=5-level butterfly that transfers 64-bit packed lanes per
    // `shuffle_xor` instead of two independent 32-bit trees, cutting
    // cross-lane shuffle traffic ~2× on the per-simdgroup tail of the
    // non-conv1d sibling of cycle-67's
    // `dmmv_q8_0_repacked_k2048_dual_nr2_qwen_conv1d.metal`. Production hot
    // user: the SSM qkv+gate dual projection when conv1d fusion isn't applied
    // (canUseQwenSsmDualRepackedQ8K2048 fallback path,
    // forward_metal.zig:18192/18247) — the same dual-matvec geometry as the
    // conv1d sibling, so the pack carries the same bit-equivalence: both
    // acc0/acc1 share an identical 5-level reduction tree, the downstream
    // lane<2 paired writeback consumes the two sums as simdgroup-uniform
    // scalars, and the existing M0=8192 (conv_channels) / M1=4096 (d_inner)
    // invariant keeps `row` and `row+1` inside one weight's output buffer.
    // Same proven pattern as cycles ~67 (conv1d dual sibling),
    // ~73/74/76/81 across the q8_0 hot kernel family.
    const float2 sums = simd_sum(float2(acc0, acc1));
    // Parallelize the 2-row writeback across lanes 0 and 1 — non-conv1d sibling
    // of cycle-27's `dmmv_q8_0_repacked_k2048_dual_nr2_qwen_conv1d.metal`
    // pattern, completing the lane-parallel writeback across both dual K=2048
    // nr2 Qwen shaders. After the packed simd_sum both float2 components are
    // broadcast on every lane; `row` and `row + 1` are two contiguous floats
    // in `output` (the production Qwen3.6 SSM dispatch passes M0=8192
    // (conv_channels) and M1=4096 (d_inner), both even, so `base_row =
    // base_pair * 2u` keeps every pair inside one weight's output buffer),
    // so lanes 0/1 issue a single coalesced 8-byte store instead of two
    // serial lane-0 stores. Same discipline as cycles
    // 27/32/38/39/40/41/43/45/46/47/48. Production hot user: the SSM qkv+gate
    // dual projection when conv1d fusion isn't applied
    // (canUseQwenSsmDualRepackedQ8K2048 fallback path, forward_metal.zig:18101).
    if (lane < 2u) {
        const float local_sum = (lane == 0u) ? sums.x : sums.y;
        output[row + lane] = local_sum;
    }
}
