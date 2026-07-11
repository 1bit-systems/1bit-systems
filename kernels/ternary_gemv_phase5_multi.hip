// Phase 5 multi-row GEMV — same INT8 ternary kernel as Phase 5, but 4 output
// rows per block. All 4 rows share the A LDS load, so A is read from global
// 1/4 as often. Each warp (Wave32) handles one row.
//
// Grid:  ((M+3)/4, 1, 1)
// Block: (128, 1, 1) — 4 warps, wave_id 0..3 -> row (block_row_base + wave_id)

#include <hip/hip_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define ROWS_PER_BLOCK 4
#define VALS_PER_U32 16
#define LDS_TILE_I8 2048

__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void ternary_gemv_phase5_multi_kernel(
    const uint32_t* __restrict__ packed,
    const int8_t*   __restrict__ x_i8,
    float                       x_scale,
    const float*    __restrict__ scales,
    float*          __restrict__ y,
    int M, int K)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wave = tid >> 5;  // 0..3

    const int row_base = blockIdx.x * ROWS_PER_BLOCK;
    const int my_row   = row_base + wave;
    const bool active  = (my_row < M);

    const int packed_k = K / VALS_PER_U32;
    const uint32_t* row_weights = active ? (packed + (size_t)my_row * packed_k) : nullptr;
    const float scale = active ? scales[my_row] : 0.0f;

    __shared__ int8_t x_shared[LDS_TILE_I8];

    int32_t acc = 0;

    for (int k_base = 0; k_base < K; k_base += LDS_TILE_I8) {
        __syncthreads();

        // 128 threads load 2048 bytes = 16 bytes/thread, vectorized 128-bit.
        using v16 = __attribute__((ext_vector_type(16))) int8_t;
        const int k_start = k_base + tid * 16;
        v16* __restrict__ lds_v = reinterpret_cast<v16*>(x_shared) + tid;
        if (k_start + 16 <= K) {
            const v16* __restrict__ src_v = reinterpret_cast<const v16*>(x_i8 + k_start);
            *lds_v = *src_v;
        } else {
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                int k = k_start + b;
                x_shared[tid * 16 + b] = (k < K) ? x_i8[k] : (int8_t)0;
            }
        }
        __syncthreads();

        // Each WARP covers K1 = LDS_TILE_I8/VALS_PER_U32 = 128 uint32s for its row.
        // 32 lanes within a warp, 128 u32s -> 4 uint32s per lane per tile.
        const int tile_u32_start = k_base / VALS_PER_U32;
        const int tile_u32_count = LDS_TILE_I8 / VALS_PER_U32;   // 128

        if (active) {
            for (int u = lane; u < tile_u32_count; u += WARP_SIZE) {
                int global_u32 = tile_u32_start + u;
                if (global_u32 >= packed_k) break;
                const uint32_t word = row_weights[global_u32];
                const int x_offset  = u * VALS_PER_U32;

                #pragma unroll
                for (int v = 0; v < VALS_PER_U32; v++) {
                    const uint32_t bits = (word >> (v * 2)) & 0x3;
                    const int xv = (int)x_shared[x_offset + v];
                    const int sign = (int)(bits == 1) - (int)(bits == 2);
                    acc += sign * xv;
                }
            }
        }
    }

    // Wave32 reduction (lane-local)
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor(acc, offset);
    }

    if (active && lane == 0) {
        y[my_row] = (float)acc * x_scale * scale;
    }
}

extern "C" void ternary_gemv_phase5_multi_launch(
    const void* packed, const void* x_i8, float x_scale,
    const void* scales, void* y,
    int M, int K, void* stream)
{
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL(ternary_gemv_phase5_multi_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const uint32_t*>(packed),
                       static_cast<const int8_t*>(x_i8),
                       x_scale,
                       static_cast<const float*>(scales),
                       static_cast<float*>(y),
                       M, K);
}
