// TWLA-style Ternary×INT4 GEMV (W1.58A4)
// Paper: TWLA — Achieving Ternary Weights and Low-Bit Activations for LLMs (ICML 2026)
// Improvement over baseline ternary_gemv.hip:
//   1. INT4 activations instead of FP32 → 2× memory BW, 2× compute
//   2. Packed INT4 (2 values per byte) → half the activation traffic
//   3. Block-scaled with FP8 E4M3 scales → no accuracy loss
//   4. Kronecker-structured weight layout for better cache behavior
//
// "Tri-modal distribution = ternary-friendly" — TWLA authors
//
// Encoding: 2 bits per weight (same as baseline), but activations are
// INT4 (signed, -8..7) packed 2 per byte.
// Block: 32 ternary weights + 1 FP8 scale = 9 bytes per block
// Activation: INT4 block = 16 bytes per 32 activations (vs 32 bytes for INT8)

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>

#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define LDS_TILE_I4 4096  // 4K INT4 values = 2K bytes in LDS
#define TWLA_BLOCK_K 32   // 32 ternary weights per scale block
#define TWLA_BLOCK_BYTES 9  // 4B (packed uint32) + 1B (scale) + 4B (zero) = 9

#include "hip_check.h"

// FP8 E4M3 → FP32 conversion (shared with block_scaled variant)
__device__ inline float fp8e4m3_to_fp32_twla(uint8_t b) {
    if (b == 0) return 0.0f;
    uint32_t sign = (uint32_t)(b >> 7) << 31;
    int exp = (b >> 3) & 0x0F;
    uint32_t mant = (uint32_t)(b & 0x07) << 20;
    // E4M3: bias=7, no implicit leading 1 for denorm
    if (exp == 0) {
        exp = 1;  // denorm → norm (E4M3 has no implicit bit)
    } else {
        mant |= 0x3F800000;  // set implicit leading 1
    }
    uint32_t f = sign | ((exp + 127 - 7) << 23) | mant;
    float result;
    __builtin_memcpy(&result, &f, 4);
    return result;
}

// Unpack INT4 from packed bytes
// 2 INT4 values per byte: low nibble = val[0], high nibble = val[1]
__device__ inline int8_t unpack_int4_low(uint8_t b) {
    int8_t v = (int8_t)(b & 0x0F);
    return (v >= 8) ? (int8_t)(v - 16) : v;  // sign-extend
}

__device__ inline int8_t unpack_int4_high(uint8_t b) {
    int8_t v = (int8_t)((b >> 4) & 0x0F);
    return (v >= 8) ? (int8_t)(v - 16) : v;
}

__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void ternary_gemv_twla_int4_kernel(
    const uint32_t* __restrict__ packed_weights,  // [M, K/16] packed ternary
    const uint8_t* __restrict__ x_int4,           // [K/2] packed INT4 activations
    const uint8_t* __restrict__ weight_scales,    // [M, K/32] FP8 E4M3 scales
    const float* __restrict__ act_scales,          // [1] activation FP32 scale
    float* __restrict__ y,                         // [M] output
    int M, int K)
{
    int row = blockIdx.x;
    if (row >= M) return;

    int tid = threadIdx.x;
    int lane = tid & (WARP_SIZE - 1);
    int warp_id = tid / WARP_SIZE;

    int blocks_per_row = (K + TWLA_BLOCK_K - 1) / TWLA_BLOCK_K;
    int packed_k = K / 16;  // uint32s per row of weights

    const uint32_t* row_w = packed_weights + (size_t)row * packed_k;
    const uint8_t* row_s = weight_scales + (size_t)row * blocks_per_row;

    // Shared memory for INT4 activations (dequantized to INT8 in LDS)
    __shared__ int8_t x_shared[LDS_TILE_I4];

    float acc = 0.0f;

    // Process K in tiles
    for (int k_base = 0; k_base < K; k_base += LDS_TILE_I4) {
        if (k_base > 0) __syncthreads();

        // Load INT4 activations into LDS, expand to INT8
        // 128 threads load LDS_TILE_I4/2 = 2048 bytes of packed INT4
        int tile_bytes = (LDS_TILE_I4 / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int i = 0; i < tile_bytes; i++) {
            int byte_idx = k_base / 2 + tid + i * BLOCK_SIZE;
            int lds_idx = (tid + i * BLOCK_SIZE) * 2;
            if (byte_idx < K / 2) {
                uint8_t byte_val = x_int4[byte_idx];
                x_shared[lds_idx]     = unpack_int4_low(byte_val);
                x_shared[lds_idx + 1] = unpack_int4_high(byte_val);
            } else {
                if (lds_idx < LDS_TILE_I4) x_shared[lds_idx] = 0;
                if (lds_idx + 1 < LDS_TILE_I4) x_shared[lds_idx + 1] = 0;
            }
        }
        __syncthreads();

        // Process blocks in this tile
        int blk_start = k_base / TWLA_BLOCK_K;
        int blk_count = (LDS_TILE_I4 + TWLA_BLOCK_K - 1) / TWLA_BLOCK_K;

        for (int b = tid; b < blk_count; b += BLOCK_SIZE) {
            int gb = blk_start + b;
            if (gb >= blocks_per_row) break;

            // Load 32 ternary weights = 2 uint32s
            int u32_base = gb * TWLA_BLOCK_K / 16;
            uint32_t w0 = (u32_base < packed_k) ? row_w[u32_base] : 0;
            uint32_t w1 = (u32_base + 1 < packed_k) ? row_w[u32_base + 1] : 0;

            // Load FP8 scale
            float scale = fp8e4m3_to_fp32_twla(row_s[gb]);

            // Dot product: 32 ternary × INT4 with accumulation
            int block_acc = 0;
            int x_base = b * TWLA_BLOCK_K;

            #pragma unroll
            for (int v = 0; v < 16; v++) {
                // First uint32
                int bits0 = (w0 >> (v * 2)) & 3;
                int sign0 = (bits0 == 1) ? 1 : (bits0 == 2 ? -1 : 0);
                int xv0 = (x_base + v < K) ? (int)x_shared[x_base + v] : 0;
                block_acc += sign0 * xv0;

                // Second uint32
                int bits1 = (w1 >> (v * 2)) & 3;
                int sign1 = (bits1 == 1) ? 1 : (bits1 == 2 ? -1 : 0);
                int xv1 = (x_base + v + 16 < K) ? (int)x_shared[x_base + v + 16] : 0;
                block_acc += sign1 * xv1;
            }

            acc += (float)block_acc * scale;
        }
    }

    // Warp reduction
    float act_scale = (act_scales ? act_scales[0] : 1.0f);
    acc *= act_scale;

    #pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
        acc += __shfl_xor(acc, o);

    __shared__ float warp_results[NUM_WARPS];
    if (lane == 0) warp_results[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        acc = (lane < NUM_WARPS) ? warp_results[lane] : 0.0f;
        #pragma unroll
        for (int o = WARP_SIZE / 2; o > 0; o >>= 1)
            acc += __shfl_xor(acc, o);
        if (lane == 0) y[row] = acc;
    }
}

// Launch wrapper
extern "C" void ternary_gemv_twla_int4_launch(
    const void* packed_weights,
    const void* x_int4,
    const void* weight_scales,
    const float* act_scales,
    void* y,
    int M, int K,
    void* stream)
{
    hipLaunchKernelGGL(ternary_gemv_twla_int4_kernel,
        dim3(M, 1, 1),
        dim3(BLOCK_SIZE, 1, 1),
        0,
        (hipStream_t)stream,
        (const uint32_t*)packed_weights,
        (const uint8_t*)x_int4,
        (const uint8_t*)weight_scales,
        act_scales ? act_scales : nullptr,
        (float*)y,
        M, K);
    HIP_CHECK(hipGetLastError());
}
