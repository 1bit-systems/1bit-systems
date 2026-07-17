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

// Sibling of dmmv_q8_0_repacked_k2048_dual_nr2_qwen.metal that fuses the
// per-channel SSM conv1d postlude into the qkv slice of the dual matvec.
// Used by the `prev_fused_attn_norm` SSM path (most layers after layer 0):
// the MoE finalizer of the previous layer already materialized norm_buf
// with this layer's attn_norm weights, so this kernel reads norm_buf
// directly and skips the inline RMSNorm — but the qkv → conv1d join
// pattern is identical to dmmv_q8_0_dual_fused_norm_conv1d.metal.
//
// Saves one standalone `ssm_conv1d_qwen_d4` dispatch + one `.qkv` barrier
// per SSM layer per decode token (≈29/decode token after layer 0 on
// Qwen3.6-35B). The conv1d postlude is run by the same lane (lane 0) that
// produces the conv1d input (sum0, sum1), and each simdgroup owns disjoint
// channel indices, so state[] writes do not race.
//
// Layout invariants (held by the dispatch site):
//   - M0 == conv_channels (the qkv slice goes through conv1d; the z slice
//     in M1 does not).
//   - M0 % 2 == 0 so the qkv→z boundary never splits a simdgroup (`first`
//     is uniform within the simdgroup; each simdgroup writes 2 contiguous
//     rows).
//   - d_conv == 4 (matches `ssm_conv1d_qwen_d4.metal`'s float4 layout).
//   - input_offset implicitly 0 (decode path; the prefill path uses a
//     different conv input buffer and is not eligible for this fusion).

kernel void main0(
    constant DualQ8DmmvConv1dPush& p [[buffer(0)]],
    device const uchar* W0 [[buffer(1)]],
    device const uchar* W1 [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* Y0 [[buffer(4)]],
    device float* Y1 [[buffer(5)]],
    device const float* conv_kernel [[buffer(6)]],
    device float* conv_state [[buffer(7)]],
    device float* conv_out [[buffer(8)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    // Cycle ~59: cooperative X-cache (8 KB threadgroup memory) for hot kernel
    // #5 (SSM in-projection qkv+z dual matvec fused with conv1d postlude,
    // 218 ms/req across 1044 calls, 15% of timed kernel time, ≈29 calls per
    // decode token via the `prev_fused_attn_norm` path). All 8 simdgroups in
    // this TG read the SAME 2048-float X (a single SSM in-projection input —
    // `x_offset` is uniform, no per-expert stride — dispatched from
    // `dispatchQwenSsmDualRepackedQ8K2048Conv1dOnCmd` at forward_metal.zig:7714
    // with `{wgs,1,1}` grid × `{block_size,1,1}` TG, block_size=256 in
    // production). Previously each simdgroup re-issued ~16 device float4 loads
    // against the same 8 KB X buffer (8 SGs × 32 lanes × 16 vi-reads each →
    // 4096 device-side float4 loads per TG; the second-and-onward SGs hit L1
    // but pay ~10-cycle L1 latency per access). Stage X into 8 KB of
    // threadgroup memory once (256 threads × 2 float4 each via the generic
    // stride loop below — perfect DRAM coalescing on the first access, then
    // 7× redundant device-side traffic eliminated), then read the 4-float
    // slice from TG mem inside the matvec (1-2 cycle latency on Apple9). Also
    // frees L1 capacity for the 64 KB of W reads per TG (8 SGs × 2 rows ×
    // 2176 B = 34816 B + the per-lane scale loads). Same x_cache discipline
    // as cycle ~56 (#2 swiglu_k2048) and cycle ~57 (#3 q5k_moe_k512_quad).
    // The load loop is generic over simdgroups_per_tg so the block_size=128
    // fallback (`max_block < 256` at forward_metal.zig:7710) still loads the
    // full 2048 floats — 128 threads × 4 float4 each. Load happens BEFORE the
    // early-return check so partial-TG tails still satisfy barrier liveness,
    // though production M0+M1=10240 with rows_per_wg=16 never early-returns.
    threadgroup float x_cache[2048];
    device const float* input_dev = X + (p.x_offset >> 2);
    const uint local_id = sg_idx * 32u + lane;
    const uint local_count = simdgroups_per_tg * 32u;
    for (uint vi = local_id; vi < 512u; vi += local_count) {
        *(threadgroup float4*)(&x_cache[vi * 4u]) = *(device const float4*)(input_dev + vi * 4u);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base_pair = tg_id * simdgroups_per_tg + sg_idx;
    const uint base_row = base_pair * 2u;
    const uint total_rows = p.M0 + p.M1;
    if (base_row >= total_rows) {
        return;
    }

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
            const float4 x = *(threadgroup const float4*)(&x_cache[x_base + (vi << 2)]);

            acc0 = fma(s0, dot(float4(q0), x), acc0);
            acc1 = fma(s1, dot(float4(q1), x), acc1);
        }
    }

    // Cycle ~67: pack the two final-reduction `simd_sum` calls into one
    // `simd_sum(float2)` — Apple9 lowers vector `simd_sum` to a single
    // log2(32)=5-level butterfly that transfers 64-bit packed lanes per
    // shuffle_xor instead of two independent 32-bit trees, cutting cross-lane
    // shuffle traffic ~2× on the per-simdgroup tail of hot kernel #5
    // (220 ms/req across 1044 calls, 15% of timed kernel time — the `prev_fused_attn_norm`
    // SSM in-proj qkv+z dual matvec fused with conv1d postlude). The
    // downstream lane<2 paired writeback consumes both sums as
    // simdgroup-uniform scalars, so picking float2 components by lane is
    // bit-equivalent. Same proven pattern as cycle ~62/63/64/65.
    const float2 sums = simd_sum(float2(acc0, acc1));
    // Distribute the two row writes + conv1d postludes across lanes 0 and 1.
    // After simd_sum both components are present on every lane, and row/row+1
    // are disjoint conv channels owned by this simdgroup, so the two lanes
    // touch non-overlapping output/state addresses. Lanes 0 and 1 reading
    // conv_state[row..row+1] (also c+row..c+row+1, 2c+row..2c+row+1) issue
    // each pair as a coalesced 8-byte transaction instead of the previous
    // two serial lane-0 loads — halves the postlude latency on the 29 SSM
    // layers per decode token that take the `prev_fused_attn_norm` path.
    if (lane < 2u) {
        const uint local_row = row + lane;
        const float local_sum = (lane == 0u) ? sums.x : sums.y;
        output[local_row] = local_sum;

        if (first) {
            const uint c = p.conv_channels;
            const float s0 = conv_state[local_row];
            const float s1 = conv_state[c + local_row];
            const float s2 = conv_state[2u * c + local_row];
            const float4 w = *(device const float4*)(conv_kernel + local_row * 4u);
            const float ss = dot(w, float4(s0, s1, s2, local_sum));
            conv_out[local_row] = ss * fast::divide(1.0f, 1.0f + fast::exp(-ss));
            conv_state[local_row] = s1;
            conv_state[c + local_row] = s2;
            conv_state[2u * c + local_row] = local_sum;
        }
    }
}
