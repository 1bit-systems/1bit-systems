#include <metal_stdlib>
using namespace metal;

struct GeGLUParams {
    uint N;
};

kernel void main0(
    constant GeGLUParams& params [[buffer(0)]],
    device const float* gate      [[buffer(1)]],
    device float* y               [[buffer(2)]],
    device const float* up        [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= params.N) return;

    const float g = gate[idx];
    const float g3 = g * g * g;
    const float inner = 0.7978845608f * fma(0.044715f, g3, g);
    const float gelu_g = 0.5f * g * (1.0f + fast::tanh(inner));
    y[idx] = gelu_g * up[idx];
}
