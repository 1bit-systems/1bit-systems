// FP8 WMMA-based GEMV for gfx1151 decode path — LDS-tiled.
//
// Uses AMD's fp8 WMMA intrinsics available on RDNA 3.5 (gfx1151) with
// ROCm 6.2+. If the compiler doesn't support the FP8 WMMA builtins,
// this file compiles to an empty translation unit (the kernels are
// skipped and the C API wrappers return -1 at runtime).
//
// Two variants:
//   fp8_wmma_gemv_fp8  — weights in FP8, activations in FP8
//   fp8_wmma_gemv_bf8  — weights in BF8, activations in BF8

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#ifndef WMMA_THREADS
#define WMMA_THREADS 32
#endif

#include "hip_check.h"

// Check whether the compiler supports AMD FP8 WMMA intrinsics.
// These are available starting with ROCm 6.2 / LLVM 18+ for gfx1151.
#if __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32)

using fp8x16_t = __attribute__((ext_vector_type(16))) uint8_t;
using float8_t = __attribute__((ext_vector_type(8))) float;

// Number of K-tiles to batch into one LDS prefetch.
// Each tile = 16 K values per row. With FP8 (1 byte/element),
// 64 tiles = 1024 K values prefetched per batch.
// 16 rows x 64 tiles x 1 byte = 1KB in LDS per wave.
#define FP8_TILES_PER_BATCH 64

// ─── FP8 WMMA GEMV kernel ─────────────────────────────────────────────────
// out[M] = in[K] @ W[M, K]^T, where W is stored in FP8 format.
// Each block handles 16 rows. Grid = (M/16).
__attribute__((amdgpu_flat_work_group_size(32, 32)))
__launch_bounds__(32) __global__ void fp8_wmma_gemv_kernel(
    const uint8_t* __restrict__ W_fp8,     // [M, K/16, 16] row-major fp8
    const uint8_t* __restrict__ x_fp8,     // [K/16, 16] fp8 activations
    const float*   __restrict__ W_scales,  // [M] per-row fp32 scale factor
    float          x_scale,                // fp32 activation scale factor
    float*         __restrict__ y,         // [M] fp32 output
    int M, int K)
{
    const int row_base = blockIdx.x * 16;
    if (row_base >= M) return;

    const int lane = threadIdx.x;
    const int u16_per_row = K / 16;
    float8_t acc = {};

    __shared__ uint8_t W_lds[16 * FP8_TILES_PER_BATCH * 16];

    for (int tile_base = 0; tile_base < u16_per_row; tile_base += FP8_TILES_PER_BATCH) {
        const int n_tiles = (tile_base + FP8_TILES_PER_BATCH <= u16_per_row)
            ? FP8_TILES_PER_BATCH : (u16_per_row - tile_base);

        __syncthreads();
        for (int t = 0; t < n_tiles; t++) {
            int tile_id = tile_base + t;
            for (int r = 0; r < 16; r++) {
                if (row_base + r < M) {
                    int lds_idx = r * FP8_TILES_PER_BATCH * 16 + t * 16;
                    int global_idx = (size_t)(row_base + r) * u16_per_row * 16 + tile_id * 16;
                    if (lane < 16)
                        W_lds[lds_idx + lane] = W_fp8[global_idx + lane];
                }
            }
        }
        // Activation tile broadcast from x_fp8 (same for all output rows)
        fp8x16_t x_tile;
        int x_base = tile_base * 16;
        if (lane == 0)
            for (int i = 0; i < 16; i++) x_tile[i] = x_fp8[x_base + i];
        for (int i = 0; i < 16; i++) x_tile[i] = __shfl(x_tile[i], 0);
        __syncthreads();

        for (int t = 0; t < n_tiles; t++) {
            fp8x16_t W_frag;
            for (int i = 0; i < 16; i++)
                W_frag[i] = W_lds[lane * FP8_TILES_PER_BATCH * 16 + t * 16 + i];
            // WMMA: fp8 x fp8 -> fp32
            acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32(W_frag, x_tile, acc);
        }
    }

    if (lane < 16 && row_base + lane < M) {
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) sum += acc[i];
        y[row_base + lane] = sum * x_scale * W_scales[row_base + lane];
    }
}

// ─── BF8 WMMA GEMV kernel ────────────────────────────────────────────────
__attribute__((amdgpu_flat_work_group_size(32, 32)))
__launch_bounds__(32) __global__ void bf8_wmma_gemv_kernel(
    const uint8_t* __restrict__ W_bf8,
    const uint8_t* __restrict__ x_bf8,
    const float*   __restrict__ W_scales,
    float          x_scale,
    float*         __restrict__ y,
    int M, int K)
{
    const int row_base = blockIdx.x * 16;
    if (row_base >= M) return;

    const int lane = threadIdx.x;
    const int u16_per_row = K / 16;
    float8_t acc = {};

    __shared__ uint8_t W_lds[16 * FP8_TILES_PER_BATCH * 16];

    for (int tile_base = 0; tile_base < u16_per_row; tile_base += FP8_TILES_PER_BATCH) {
        const int n_tiles = (tile_base + FP8_TILES_PER_BATCH <= u16_per_row)
            ? FP8_TILES_PER_BATCH : (u16_per_row - tile_base);

        __syncthreads();
        for (int t = 0; t < n_tiles; t++) {
            int tile_id = tile_base + t;
            for (int r = 0; r < 16; r++) {
                if (row_base + r < M) {
                    int lds_idx = r * FP8_TILES_PER_BATCH * 16 + t * 16;
                    int global_idx = (size_t)(row_base + r) * u16_per_row * 16 + tile_id * 16;
                    if (lane < 16)
                        W_lds[lds_idx + lane] = W_bf8[global_idx + lane];
                }
            }
        }
        fp8x16_t x_tile;
        int x_base = tile_base * 16;
        if (lane == 0)
            for (int i = 0; i < 16; i++) x_tile[i] = x_bf8[x_base + i];
        for (int i = 0; i < 16; i++) x_tile[i] = __shfl(x_tile[i], 0);
        __syncthreads();

        for (int t = 0; t < n_tiles; t++) {
            fp8x16_t W_frag;
            for (int i = 0; i < 16; i++)
                W_frag[i] = W_lds[lane * FP8_TILES_PER_BATCH * 16 + t * 16 + i];
            // BF8 WMMA: bf8 x bf8 -> fp32
            acc = __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32(W_frag, x_tile, acc);
        }
    }

    if (lane < 16 && row_base + lane < M) {
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) sum += acc[i];
        y[row_base + lane] = sum * x_scale * W_scales[row_base + lane];
    }
}

// ─── Host launch wrappers (compiled only when FP8 WMMA is available) ─────

extern "C" int rcpp_fp8_wmma_gemv(
    const void* W_dev, const void* x_dev,
    const float* scales_dev, float x_scale,
    float* y_dev, int M, int K, void* stream)
{
    if (!W_dev || !x_dev || !scales_dev || !y_dev) return -1;
    if (M <= 0 || K <= 0 || (K % 16) != 0) return -1;

    int grid = (M + 15) / 16;
    hipLaunchKernelGGL(fp8_wmma_gemv_kernel,
        dim3(grid), dim3(32), 0, (hipStream_t)stream,
        (const uint8_t*)W_dev, (const uint8_t*)x_dev,
        scales_dev, x_scale, y_dev, M, K);
    HIP_CHECK(hipGetLastError());
    return 0;
}

extern "C" int rcpp_bf8_wmma_gemv(
    const void* W_dev, const void* x_dev,
    const float* scales_dev, float x_scale,
    float* y_dev, int M, int K, void* stream)
{
    if (!W_dev || !x_dev || !scales_dev || !y_dev) return -1;
    if (M <= 0 || K <= 0 || (K % 16) != 0) return -1;

    int grid = (M + 15) / 16;
    hipLaunchKernelGGL(bf8_wmma_gemv_kernel,
        dim3(grid), dim3(32), 0, (hipStream_t)stream,
        (const uint8_t*)W_dev, (const uint8_t*)x_dev,
        scales_dev, x_scale, y_dev, M, K);
    HIP_CHECK(hipGetLastError());
    return 0;
}

#else  // FP8 WMMA not supported by this compiler

// Stub: return -1 at runtime (caller should check return value and fall back)
extern "C" int rcpp_fp8_wmma_gemv(
    const void*, const void*, const float*, float, float*, int, int, void*)
{
    fprintf(stderr, "  [fp8_wmma] FP8 WMMA not supported by this ROCm version (need 6.2+)\n");
    return -1;
}

extern "C" int rcpp_bf8_wmma_gemv(
    const void*, const void*, const float*, float, float*, int, int, void*)
{
    fprintf(stderr, "  [bf8_wmma] BF8 WMMA not supported by this ROCm version (need 6.2+)\n");
    return -1;
}

#endif
