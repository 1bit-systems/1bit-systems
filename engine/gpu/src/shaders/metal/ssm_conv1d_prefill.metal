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
    if (ch >= p.conv_channels || p.d_conv < 1u || p.d_conv > 8u) {
        return;
    }

    float hist[7];
    const uint hist_len = p.d_conv - 1u;
    for (uint i = 0u; i < hist_len; ++i) {
        hist[i] = state[i * p.conv_channels + ch];
    }

    for (uint token = 0u; token < p.n_tokens; ++token) {
        const uint token_base = token * p.input_stride + ch;
        const float current = input[p.input_offset + token_base];

        float sum = 0.0f;
        for (uint ki = 0u; ki < p.d_conv; ++ki) {
            const float x = (ki < hist_len) ? hist[ki] : current;
            sum = fma(kernel_w[ch * p.d_conv + ki], x, sum);
        }

        output[p.output_offset + token_base] = sum / (1.0f + exp(-sum));

        for (uint i = 0u; i + 1u < hist_len; ++i) {
            hist[i] = hist[i + 1u];
        }
        if (hist_len > 0u) {
            hist[hist_len - 1u] = current;
        }
    }

    for (uint i = 0u; i < hist_len; ++i) {
        state[i * p.conv_channels + ch] = hist[i];
    }
}
