#include <metal_stdlib>
using namespace metal;

struct DualF32DmmvPush {
    uint M0;
    uint M1;
    uint K;
    uint a0_offset;
    uint a1_offset;
    uint x_offset;
    uint y0_offset;
    uint y1_offset;
};

// Two small F32 matvecs in one launch.
//
// Qwen3.6 SSM alpha/beta tails are both 32x2048 F32 projections. The previous
// path launched dmmv_f32 twice; this keeps the same combined alpha/beta
// dispatch, but gives each row a full simdgroup so the K reduction is parallel
// instead of one thread walking all 2048 columns. grid.y selects alpha vs beta.
// grid.z is the prompt-token row for layer-major prefill; single-token decode
// dispatches z=1 and keeps the original layout.
kernel void main0(
    constant DualF32DmmvPush& p [[buffer(0)]],
    device const char* W0 [[buffer(1)]],
    device const char* W1 [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* Y0 [[buffer(4)]],
    device float* Y1 [[buffer(5)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    const bool second = tg_pos.y != 0u;
    const uint M = second ? p.M1 : p.M0;
    const uint row = tg_pos.x * simdgroups_per_tg + simd_idx;
    if (row >= M || tg_pos.y > 1u) {
        return;
    }

    device const char* Wc = second ? W1 : W0;
    const uint a_offset = second ? p.a1_offset : p.a0_offset;
    const uint y_offset = second ? p.y1_offset : p.y0_offset;
    device float* Y = second ? Y1 : Y0;

    device const float* W = (device const float*)(Wc + a_offset);
    const uint token = tg_pos.z;
    device const float* x = X + (p.x_offset >> 2) + token * p.K;
    device const float* w = W + row * p.K;

    float acc = 0.0f;
    for (uint k = lane; k < p.K; k += 32u) {
        acc = fma(w[k], x[k], acc);
    }
    acc = simd_sum(acc);

    if (lane == 0u) {
        Y[(y_offset >> 2) + token * M + row] = acc;
    }
}
