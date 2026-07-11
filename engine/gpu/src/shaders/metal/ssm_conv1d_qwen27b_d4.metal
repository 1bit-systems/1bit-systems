#include <metal_stdlib>

using namespace metal;

struct Params {
    uint conv_channels;
    uint d_conv;
    uint kernel_is_f16;
    uint input_offset;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* kernel_w [[buffer(1)]],
    device float* state [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint ch [[thread_position_in_grid]]
) {
    if (ch >= 10240u || p.conv_channels != 10240u || p.d_conv != 4u || p.kernel_is_f16 != 0u) {
        return;
    }

    constexpr uint c = 10240u;
    const float x0 = state[ch];
    const float x1 = state[c + ch];
    const float x2 = state[2u * c + ch];
    const float x3 = input[p.input_offset + ch];
    const float4 w = *(device const float4*)(kernel_w + ch * 4u);
    const float sum = dot(w, float4(x0, x1, x2, x3));

    output[ch] = sum * fast::divide(1.0f, 1.0f + fast::exp(-sum));
    state[ch] = x1;
    state[c + ch] = x2;
    state[2u * c + ch] = x3;
}
