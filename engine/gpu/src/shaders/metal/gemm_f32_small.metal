#include <metal_stdlib>
using namespace metal;

// Small f32 batched GEMV for one-row auxiliary gates.
//
// Computes dst[N, M] = W[M, K] x src[N, K] with f32 weights and f32 input.
// The Qwen route-packed prefill path uses this only for M=1
// ffn_gate_inp_shexp.weight, keeping the main shared expert matrices on the
// faster Q8/Q4/Q6 simdgroup-matrix kernels.

struct GemmF32SmallPush {
    uint M;
    uint K;
    uint N;
    uint src0_off;
};

kernel void main0(
    constant GemmF32SmallPush& args [[buffer(0)]],
    device const char* src0 [[buffer(1)]],
    device const float* src1 [[buffer(2)]],
    device float* dst [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const uint row = tgpig.x;
    const uint token = tgpig.y;
    if (row >= args.M || token >= args.N) {
        return;
    }

    device const float* weights = (device const float*)(src0 + args.src0_off);
    device const float* w = weights + row * args.K;
    device const float* x = src1 + token * args.K;

    float acc = 0.0f;
    for (uint k = tid; k < args.K; k += 32u) {
        acc = fma(w[k], x[k], acc);
    }

    const float sum = simd_sum(acc);
    if (tid == 0u) {
        dst[token * args.M + row] = sum;
    }
}
