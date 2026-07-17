#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
    uint g_offset;
};

// Qwen3.6 exact repacked Q8_0 DMMV for K=4096 and M%4==0, fused with the
// attn_gate sigmoid-multiply on the input vector.
//
// Adapted from `dmmv_q8_0_repacked_k4096_qwen.metal` (the standalone SSM-out /
// full-attn-out kernel). The full-attn path's sigmoid_mul kernel was an extra
// dispatch + barrier between flash-attn and the output projection — mirroring
// the reference implementation `ggml_metal_op_concurrency_check/reset` discipline, fuse it into
// the output DMMV's input load so the read of `X * sigmoid(G)` happens inline
// while we already have the X cache line resident. Removes one dispatch +
// barrier per full-attn layer with attn_gate (10/token on Qwen3.6-35B), at no
// extra memory traffic — gate_buf was already going to be read once anyway.
kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device const float* G [[buffer(3)]],
    device float* Y [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    const uint base_row = (tg_id * simdgroups_per_tg + sg_idx) * 4u;
    if (base_row >= p.M) return;

    device const float* input = X + (p.x_offset >> 2);
    device const float* gate = G + (p.g_offset >> 2);
    device float* output = Y + (p.y_offset >> 2);

    constexpr ulong group_bytes = 1088ul;
    constexpr ulong row_bytes = 4352ul;

    device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
    device const uchar* row1 = row0 + row_bytes;
    device const uchar* row2 = row0 + 2ul * row_bytes;
    device const uchar* row3 = row0 + 3ul * row_bytes;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    #pragma unroll
    for (uint gi = 0u; gi < 4u; ++gi) {
        device const uchar* g0 = row0 + ulong(gi) * group_bytes;
        device const uchar* g1 = row1 + ulong(gi) * group_bytes;
        device const uchar* g2 = row2 + ulong(gi) * group_bytes;
        device const uchar* g3 = row3 + ulong(gi) * group_bytes;

        const float s0 = float(as_type<half>(*(device const ushort*)(g0 + lane * 2u)));
        const float s1 = float(as_type<half>(*(device const ushort*)(g1 + lane * 2u)));
        const float s2 = float(as_type<half>(*(device const ushort*)(g2 + lane * 2u)));
        const float s3 = float(as_type<half>(*(device const ushort*)(g3 + lane * 2u)));
        const uint x_base = (gi * 32u + lane) << 5;

        #pragma unroll
        for (uint vi = 0u; vi < 8u; ++vi) {
            const uint qo = 64u + vi * 128u + lane * 4u;
            const char4 q0 = as_type<char4>(*(device const int*)(g0 + qo));
            const char4 q1 = as_type<char4>(*(device const int*)(g1 + qo));
            const char4 q2 = as_type<char4>(*(device const int*)(g2 + qo));
            const char4 q3 = as_type<char4>(*(device const int*)(g3 + qo));
            const float4 gate_vals = *(device const float4*)(gate + x_base + (vi << 2));
            const float4 sig_gate = float4(1.0f) / (float4(1.0f) + exp(-gate_vals));
            const float4 x = (*(device const float4*)(input + x_base + (vi << 2))) * sig_gate;

            acc0 = fma(s0, dot(float4(q0), x), acc0);
            acc1 = fma(s1, dot(float4(q1), x), acc1);
            acc2 = fma(s2, dot(float4(q2), x), acc2);
            acc3 = fma(s3, dot(float4(q3), x), acc3);
        }
    }

    // Cycle ~76: pack the four final-reduction `simd_sum` calls into one
    // `simd_sum(float4)` — Apple9 lowers vector `simd_sum` to a single
    // log2(32)=5-level butterfly that transfers 128-bit packed lanes per
    // shuffle_xor instead of four independent 32-bit trees, cutting cross-lane
    // shuffle traffic ~4× on the per-simdgroup tail of the Qwen3.6 full-attn
    // output projection fused with attn_gate sigmoid-multiply (M=2048,
    // K=4096, ~10 calls/token × 256 WGs ≈ 2.5K TGs/token across the 10
    // full-attn layers per decode token). Gated sibling of cycle 74's
    // `dmmv_q8_0_repacked_k4096_qwen.metal` pack (which covers the
    // ungated SSM-out / full-attn-out at 30/token) — same K=4096
    // exact-repacked Q8 4-row geometry, same float4-packed reduction.
    // Also mirrors cycle 73 (`dmmv_q8_0_repacked_k2048_qwen` K=2048 sibling)
    // and cycle 71 (`dmmv_q8_0_k4096_quad` non-repacked K=4096 sibling).
    // Downstream lane<4 writeback consumes the four sums as simdgroup-uniform
    // scalars (component access on the simdgroup-uniform float4), so picking
    // float4 components by lane is bit-equivalent. The dispatch route only
    // fires for production M=2048 (multiple of 4) via
    // `canUseQwenGatedAttnOutDmmv` + the `base_row >= p.M` early-return at
    // line 36 means all four rows owned by this simdgroup are always valid
    // — no aliasing safety dance needed (matches the existing unconditional
    // 4-row write below).
    const float4 sums = simd_sum(float4(acc0, acc1, acc2, acc3));
    // Parallelize the 4-row writeback across lanes 0..3 (cycle-40 sibling of
    // cycle-39's standalone K=4096 lane-parallel writeback). After simd_sum
    // all four sums are present on every lane; base_row+0..+3 are four
    // contiguous floats so lanes 0..3 of this simdgroup issue a single
    // coalesced 16-byte store instead of four serial lane-0 stores. The
    // dispatch route `.exact_qwen_k4096_gated` gates on `M % 4 == 0` and
    // the `base_row >= p.M` early-return at the top means all four rows
    // owned by this simdgroup are always valid (no per-row `has` check).
    // Production hot user: full-attn out projection with attn_gate
    // (M=2048, K=4096, 10/token × 256 WGs ≈ 2.5K TGs/token). Same
    // lane-parallel pattern as cycle-32/38/39 across the standalone
    // Q8_0 exact-repacked kernel family.
    if (lane < 4u) {
        const float local_sum = (lane == 0u) ? sums.x
                              : (lane == 1u) ? sums.y
                              : (lane == 2u) ? sums.z
                              : sums.w;
        output[base_row + lane] = local_sum;
    }
}
