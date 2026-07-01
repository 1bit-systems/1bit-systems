#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Exact even-row Qwen3.6 SSM Q8_0 repacked DMMV for K=4096.
//
// Keeps the accepted reference-style TG128/two-row geometry for ssm_out while
// removing the generic tail-row branch. Production Qwen3.6 ssm_out rows are
// even, so the guarded dispatch below keeps tail cases on the older shader.
kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    const uint base_row = (tg_id * 4u + sg_idx) * 2u;

    device const float* input = X + (p.x_offset >> 2);
    device float* output = Y + (p.y_offset >> 2);

    constexpr ulong group_bytes = 1088ul;
    constexpr ulong row_bytes = 4352ul;

    device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
    device const uchar* row1 = row0 + row_bytes;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    #pragma unroll
    for (uint gi = 0u; gi < 4u; ++gi) {
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

    // Pack the two per-row simdgroup reductions into a single
    // `simd_sum(float2)` — Apple9's vector `simd_sum` lowers to one
    // log2(32)=5-level butterfly that transfers a 64-bit packed lane per
    // `shuffle_xor` instead of two independent 32-bit trees, cutting cross-
    // lane shuffle traffic ~2× on the per-simdgroup tail of hot kernel #4
    // by streamed bytes (SSM `ssm_out` M=2048 K=4096, 8.96 GiB/req across
    // 1080 calls — every SSM layer per decode token). K=4096 sibling of the
    // cycle ~75 K=2048 nr2 qwen pack. Same proven pattern as cycle ~82
    // (`dmmv_q8_0_repacked_k2048_dual_nr2_qwen`) and cycle ~83
    // (`dmmv_q8_0_pair`). Downstream lane<2 writeback consumes the sums
    // as simdgroup-uniform scalars (component access on the simdgroup-
    // uniform float2), so picking float2 components by lane is bit-
    // equivalent.
    const float2 sums = simd_sum(float2(acc0, acc1));
    // Distribute the two row writes across lanes 0 and 1 so the pair of
    // output stores at base_row..base_row+1 issues as one coalesced 8-byte
    // transaction (sibling of cycle-45 k2048_nr2_qwen, cycle-43 pair-swiglu,
    // cycle-32 conv1d, cycle-38/39 4-row writeback pattern). M is guaranteed
    // to be a multiple of `rows_per_wg=8` by the dispatcher (see
    // forward_metal.zig M % rows_per_wg == 0 guard before `tg128_k4096_qwen`
    // selection), so base_row + 1 is always in range.
    if (lane < 2u) {
        output[base_row + lane] = (lane == 0u) ? sums.x : sums.y;
    }
}
