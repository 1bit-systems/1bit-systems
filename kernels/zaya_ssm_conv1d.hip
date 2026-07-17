// Zaya SSM conv1d — depthwise 1D convolution for the State Space Model path.
//
// Zaya's hybrid architecture combines CCA (Cross-Computer Attention) with a
// State Space Model conv1d. This kernel computes the SSM convolution:
//
//   ssm_state[t] = conv1d(ssm_state[t-1], ssm_conv_weight) + input[t]
//
// The conv is depthwise (per-channel), with a kernel width of 2 and 1280
// channels. Each channel has its own learned filter: [w0, w1].
//
// Grid: 1 block, 1280 threads (one per channel)
// Block: 128 threads, single wave
//
// Build:
//   /opt/rocm-7.2.4/bin/hipcc -O3 --offload-arch=gfx1151 zaya_ssm_conv1d.hip

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ── SSM conv1d: depthwise 1D convolution, kernel width 2 ─────────────────
//
// state_in     [channels] FP16 — previous SSM state (or zero-initialized)
// conv_weight  [2, channels] FP16 — per-channel learned weights [w0, w1]
// conv_bias    [channels] FP16 — per-channel bias (or nullptr)
// input        [channels] FP16 — current input to the SSM path
// state_out    [channels] FP16 — updated SSM state (write-back)
// output       [channels] FP16 — conv output = state[t]
// channels     — number of SSM channels (1280 for ZAYA1-8B)
//
// Formula: output[c] = state_in[c] * w0[c] + input[c] * w1[c] + bias[c]
//
// Note: This is the Zaya "shallow SSM" variant where the conv acts as a
// gated recurrence. The full SSM (with Hippo matrix) is not used here.
__global__ void zaya_ssm_conv1d_kernel(
    const __half* __restrict__ state_in,
    const __half* __restrict__ conv_weight,
    const __half* __restrict__ conv_bias,
    const __half* __restrict__ input,
    __half* __restrict__ state_out,
    __half* __restrict__ output,
    int channels)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    // Load weights: conv_weight is [2, channels] = [2 * channels]
    // w0 = conv_weight[c]         (coefficient for previous state)
    // w1 = conv_weight[channels + c]  (coefficient for current input)
    float w0 = __half2float(conv_weight[c]);
    float w1 = __half2float(conv_weight[channels + c]);

    // Load state and input
    float s = __half2float(state_in ? state_in[c] : __float2half(0.0f));
    float x = __half2float(input[c]);

    // Compute: new_state = w0 * prev_state + w1 * input
    float new_s = fmaf(w0, s, w1 * x);

    // Add bias
    float bias = conv_bias ? __half2float(conv_bias[c]) : 0.0f;
    float out = new_s + bias;

    // Write back
    state_out[c] = __float2half(new_s);
    output[c]    = __float2half(out);
}

// ── C API (extern "C" for linking with bitnet_decode) ────────────────────
extern "C" void zaya_ssm_conv1d_launch(
    const void* state_in,
    const void* conv_weight,
    const void* conv_bias,
    const void* input,
    void* state_out,
    void* output,
    int channels,
    void* stream)
{
    dim3 grid((channels + 127) / 128, 1, 1);
    dim3 block(128, 1, 1);
    hipStream_t s = static_cast<hipStream_t>(stream);

    hipLaunchKernelGGL(zaya_ssm_conv1d_kernel,
        grid, block, 0, s,
        static_cast<const __half*>(state_in),
        static_cast<const __half*>(conv_weight),
        static_cast<const __half*>(conv_bias),
        static_cast<const __half*>(input),
        static_cast<__half*>(state_out),
        static_cast<__half*>(output),
        channels);
}
