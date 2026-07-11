#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Qwen3.6 exact repacked Q8_0 DMMV for K=4096 and M%4==0.
//
// Tail-free fixed row groups match the production SSM-out/full-attn-out shapes.
// The generic K=4096 quad kernel remains the safety path for tests and odd M.
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
    const uint base_row = (tg_id * simdgroups_per_tg + sg_idx) * 4u;
    if (base_row >= p.M) return;

    device const float* input = X + (p.x_offset >> 2);
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
            const float4 x = *(device const float4*)(input + x_base + (vi << 2));

            acc0 = fma(s0, dot(float4(q0), x), acc0);
            acc1 = fma(s1, dot(float4(q1), x), acc1);
            acc2 = fma(s2, dot(float4(q2), x), acc2);
            acc3 = fma(s3, dot(float4(q3), x), acc3);
        }
    }

    // Pack the four per-row simdgroup reductions into a single `simd_sum(float4)`
    // — Apple9's vector `simd_sum` lowers to one 5-level butterfly that
    // transfers a 128-bit packed lane per `shuffle_xor`, cutting cross-lane
    // shuffle traffic ~4× on the per-simdgroup tail vs. four scalar `simd_sum`
    // calls. K=4096 sibling of cycle-73's K=2048 repacked-qwen pack and cycle-71's
    // non-repacked K=4096 quad pack (`dmmv_q8_0_k4096_quad.metal`). Production
    // hot user: SSM out projection (M=2048, K=4096, 1080 calls/req × 256 TGs ≈
    // 276K simdgroup-tail reductions/req — hot kernel #4 by streamed bytes,
    // 8.96 GiB/req across both SSM-out and full-attn-out which share this exact
    // shape). The lane-parallel writeback below remains: lanes 0..3 each write
    // one of the float4 components, issuing a coalesced 16-byte store.
    const float4 sums = simd_sum(float4(acc0, acc1, acc2, acc3));
    if (lane < 4u) {
        output[base_row + lane] = sums[lane];
    }
}
