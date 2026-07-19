// INT8 GEMV kernel using sudot4 — Hadamard-rotated decode for gfx1151.
//
// Design:
//   128 threads per block, 128 rows per block, 1 row per thread.
//   Cooperative LDS load of activations. Each thread independently
//   computes its row's dot product. NO warp reduction — each thread
//   owns its row exclusively.
//
//   K/4 sudot4 operations per row, each covering 4 int8 elements.
//   Activation shared via LDS; weights loaded per-thread from global.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include "rocm_cpp/ck_gemm.h"

#define BLOCK_SIZE     128
#define LDS_TILE_I8    2560    // 128 threads × 20 bytes = 2560
#include "hip_check.h"

__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void wmma_i8_gemv_kernel(
    const int8_t*  __restrict__ W_i8,         // [M, K] row-major
    const int8_t*  __restrict__ x_i8,         // [K]
    float                      x_scale,
    const float*   __restrict__ row_scales,   // [M] per-row fp32 scale
    __half*        __restrict__ y_fp16,        // [M]
    int M, int K)
{
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    __shared__ int8_t x_shared[LDS_TILE_I8];

    int32_t acc = 0;
    const int8_t* w_row = W_i8 + (size_t)row * K;
    const int K_u32 = K / 4;

    // Stream through K in LDS tiles
    for (int k_base = 0; k_base < K; k_base += LDS_TILE_I8) {
        // Cooperative load of activation tile into LDS
        __syncthreads();
        const int lds_off = tid * 20;
        const int k_start = k_base + lds_off;
        if (k_start + 20 <= K) {
            uint32_t* lds_u32 = reinterpret_cast<uint32_t*>(&x_shared[lds_off]);
            const uint32_t* src_u32 = reinterpret_cast<const uint32_t*>(x_i8 + k_start);
            #pragma unroll
            for (int u = 0; u < 5; ++u) lds_u32[u] = src_u32[u];
        } else {
            #pragma unroll
            for (int b = 0; b < 20; ++b) {
                int k = k_start + b;
                x_shared[lds_off + b] = (k < K) ? x_i8[k] : (int8_t)0;
            }
        }
        __syncthreads();

        const int tile_k = (k_base + LDS_TILE_I8 <= K) ? LDS_TILE_I8 : (K - k_base);
        const int tile_ku = tile_k / 4;
        const int tile_ku_base = k_base / 4;

        // Each thread processes ALL K/4 u32s for its row (no warp sharing).
        for (int ku = 0; ku < tile_ku; ++ku) {
            uint32_t w_packed = *reinterpret_cast<const uint32_t*>(w_row + k_base + ku * 4);
            uint32_t x_packed = *reinterpret_cast<const uint32_t*>(&x_shared[ku * 4]);
            acc = __builtin_amdgcn_sudot4(true, w_packed, true, x_packed, acc, false);
        }
    }

    constexpr float FP16_MAX = 65504.0f;
    float out = (float)acc * x_scale * row_scales[row];
    out = fminf(fmaxf(out, -FP16_MAX), FP16_MAX);
    y_fp16[row] = __float2half(out);
}

extern "C" rcpp_status_t rcpp_wmma_i8_gemv(
    const void* W_i8_dev, const void* x_i8_dev,
    float x_scale, const void* row_scales_dev,
    void* y_fp16_dev, int M, int K, void* stream)
{
    if (!W_i8_dev || !x_i8_dev || !row_scales_dev || !y_fp16_dev)
        return RCPP_INVALID_ARG;
    if (M <= 0 || K <= 0 || (K % 4) != 0)
        return RCPP_INVALID_ARG;
    if (x_scale <= 0.0f)
        return RCPP_INVALID_ARG;

    const int grid = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 g(grid, 1, 1);
    dim3 b(BLOCK_SIZE, 1, 1);

    hipLaunchKernelGGL(wmma_i8_gemv_kernel, g, b, 0, (hipStream_t)stream,
        static_cast<const int8_t*>(W_i8_dev),
        static_cast<const int8_t*>(x_i8_dev),
        x_scale,
        static_cast<const float*>(row_scales_dev),
        static_cast<__half*>(y_fp16_dev), M, K);
    HIP_CHECK(hipGetLastError());
    return RCPP_OK;
}
