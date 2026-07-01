#include <metal_stdlib>

using namespace metal;

struct Params {
    uint conv_channels;
    uint d_conv;
    uint n_tokens;
    uint input_stride;
    uint input_offset;
    uint output_offset;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* kernel_w [[buffer(1)]],
    device float* state [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint ch [[thread_position_in_grid]]
) {
    if (ch >= 8192u || p.conv_channels != 8192u || p.d_conv != 4u) {
        return;
    }

    const uint c = 8192u;
    const uint k = ch * 4u;
    const float w0 = kernel_w[k];
    const float w1 = kernel_w[k + 1u];
    const float w2 = kernel_w[k + 2u];
    const float w3 = kernel_w[k + 3u];

    float x0 = state[ch];
    float x1 = state[c + ch];
    float x2 = state[2u * c + ch];

    for (uint token = 0u; token < p.n_tokens; ++token) {
        const uint token_base = token * p.input_stride + ch;
        const float x3 = input[p.input_offset + token_base];
        const float sum = fma(w3, x3, fma(w2, x2, fma(w1, x1, w0 * x0)));
        output[p.output_offset + token_base] = sum * fast::divide(1.0f, 1.0f + fast::exp(-sum));
        x0 = x1;
        x1 = x2;
        x2 = x3;
    }

    state[ch] = x0;
    state[c + ch] = x1;
    state[2u * c + ch] = x2;
}
