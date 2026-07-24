// ternary_gemv_q1_0.hip — 1-bit binary GEMV kernel (Q1_0 format)
//
// Q1_0 format: groups of 128 elements
//   - 1 fp16 scale (2 bytes)
//   - 16 bytes of packed bits (128/8)
//   - bit = 0 → -scale, bit = 1 → +scale
//   - Total per group: 18 bytes (1.125 bpw)
//
// Activation: int8_t[K], dynamically quantized per call.
// Output: float16[M]. Per-row fp16 post-quant scales optional.

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q1_BLOCK_K 128
#define Q1_BLOCK_BYTES 18
#define Q1_BITS_OFFSET 2   // skip fp16 scale

// ─── Kernel declaration ───────────────────────────────────────────
__global__ void q1_0_gemv_kernel(
    const uint8_t* packed,
    const int8_t*  x_i8,
    float          x_scale,
    const __half*  row_scales,
    __half*        y_f16,
    int            M,
    int            K);

// ─── Launch wrapper ───────────────────────────────────────────────
extern "C" void ternary_gemv_q1_0_f16_launch(
    const void* packed,
    const void* x_i8,
    float x_scale,
    const void* row_scales,
    void* y_f16,
    int M,
    int K,
    void* stream_void)
{
    if (M <= 0 || K <= 0) return;
    hipStream_t stream = stream_void ? *(hipStream_t*)stream_void : hipStreamPerThread;

    dim3 block(64);
    dim3 grid((M + 63) / 64);

    hipLaunchKernelGGL(
        q1_0_gemv_kernel,
        grid, block, 0, stream,
        (const uint8_t*)packed, (const int8_t*)x_i8, x_scale,
        (const __half*)row_scales, (__half*)y_f16, M, K);
}

// ─── Kernel body ──────────────────────────────────────────────────
__global__ void q1_0_gemv_kernel(
    const uint8_t* __restrict__ packed,
    const int8_t*  __restrict__ x_i8,
    float          x_scale,
    const __half*  __restrict__ row_scales,
    __half*        __restrict__ y_f16,
    int            M,
    int            K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    int n_blocks = (K + Q1_BLOCK_K - 1) / Q1_BLOCK_K;
    int row_stride = n_blocks * Q1_BLOCK_BYTES;

    float sum = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        int block_start = b * Q1_BLOCK_K;
        int block_end = min(block_start + Q1_BLOCK_K, K);
        int cols_here = block_end - block_start;

        const uint8_t* block = packed + (size_t)row * row_stride + b * Q1_BLOCK_BYTES;

        // Read fp16 scale
        __half scale_h;
        memcpy(&scale_h, block, 2);
        float scale = (float)scale_h;

        // Read 16 bytes of packed bits
        uint8_t bits[16];
        memcpy(bits, block + Q1_BITS_OFFSET, 16);

        // Accumulate: for each of the 128 bits, ternary = ±scale
        for (int i = 0; i < cols_here; i++) {
            int col = block_start + i;
            int byte_i = i >> 3;       // i / 8
            int bit_i = i & 7;          // i % 8
            float w = (bits[byte_i] & (1 << bit_i)) ? scale : -scale;
            sum += w * (float)x_i8[col];
        }
    }

    sum *= x_scale;
    if (row_scales) sum *= (float)row_scales[row];
    y_f16[row] = (__half)sum;
}
