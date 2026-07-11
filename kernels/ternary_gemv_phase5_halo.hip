// Phase 5 dot4 — halo-1bit encoding variant.
//
// Halo-1bit's TernaryLinear packs weights with this 2-bit code per value:
//   0 -> -1
//   1 ->  0 (skip)
//   2 -> +1
//   3 -> unused
//
// Stored as uint8 [rows, (cols+3)/4] — 4 values per byte, K-contiguous within
// byte. Reinterpret-cast to uint32: 4 bytes = one uint32 = 16 values, same
// K ordering as our Phase 5 kernel expects.
//
// Same 8 rows/block + dot4 pipeline as the phase5_dot4 kernel, but the
// signed-unpack step matches halo's code assignment.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define ROWS_PER_BLOCK 8
#define ROWS_PER_WARP  2
#define VALS_PER_U32 16
#define LDS_TILE_I8 2048

static __device__ __forceinline__ uint32_t pack4(int8_t a, int8_t b, int8_t c, int8_t d) {
    return ((uint32_t)(uint8_t)a)
         | ((uint32_t)(uint8_t)b << 8)
         | ((uint32_t)(uint8_t)c << 16)
         | ((uint32_t)(uint8_t)d << 24);
}

// Halo encoding: sign = (bits == 2) - (bits == 0).
static __device__ __forceinline__ uint32_t unpack_ternary_halo_4(uint32_t word, int base_v) {
    const uint32_t bits0 = (word >> ((base_v + 0) * 2)) & 0x3;
    const uint32_t bits1 = (word >> ((base_v + 1) * 2)) & 0x3;
    const uint32_t bits2 = (word >> ((base_v + 2) * 2)) & 0x3;
    const uint32_t bits3 = (word >> ((base_v + 3) * 2)) & 0x3;
    const int8_t s0 = (int8_t)((int)(bits0 == 2) - (int)(bits0 == 0));
    const int8_t s1 = (int8_t)((int)(bits1 == 2) - (int)(bits1 == 0));
    const int8_t s2 = (int8_t)((int)(bits2 == 2) - (int)(bits2 == 0));
    const int8_t s3 = (int8_t)((int)(bits3 == 2) - (int)(bits3 == 0));
    return pack4(s0, s1, s2, s3);
}

__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void ternary_gemv_phase5_halo_kernel(
    const uint32_t* __restrict__ packed,
    const int8_t*   __restrict__ x_i8,
    float                       x_scale,
    const float*    __restrict__ scales,
    float*          __restrict__ y,
    int M, int K)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wave = tid >> 5;

    const int row_base = blockIdx.x * ROWS_PER_BLOCK;
    const int warp_row_base = row_base + wave * ROWS_PER_WARP;

    const int packed_k = K / VALS_PER_U32;

    __shared__ int8_t x_shared[LDS_TILE_I8];

    int32_t acc[ROWS_PER_WARP] = {};

    for (int k_base = 0; k_base < K; k_base += LDS_TILE_I8) {
        __syncthreads();
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

        const int tile_u32_start = k_base / VALS_PER_U32;
        const int tile_u32_count = LDS_TILE_I8 / VALS_PER_U32;

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int my_row = warp_row_base + r;
            if (my_row >= M) continue;
            const uint32_t* row_weights = packed + (size_t)my_row * packed_k;

            for (int u = lane; u < tile_u32_count; u += WARP_SIZE) {
                int global_u32 = tile_u32_start + u;
                if (global_u32 >= packed_k) break;
                const uint32_t word = row_weights[global_u32];
                const int x_offset  = u * VALS_PER_U32;

                const uint32_t* xpack = reinterpret_cast<const uint32_t*>(&x_shared[x_offset]);
                const uint32_t xw0 = xpack[0];
                const uint32_t xw1 = xpack[1];
                const uint32_t xw2 = xpack[2];
                const uint32_t xw3 = xpack[3];

                const uint32_t tw0 = unpack_ternary_halo_4(word, 0);
                const uint32_t tw1 = unpack_ternary_halo_4(word, 4);
                const uint32_t tw2 = unpack_ternary_halo_4(word, 8);
                const uint32_t tw3 = unpack_ternary_halo_4(word, 12);

                int32_t local = 0;
                local = __builtin_amdgcn_sudot4(true, tw0, true, xw0, local, false);
                local = __builtin_amdgcn_sudot4(true, tw1, true, xw1, local, false);
                local = __builtin_amdgcn_sudot4(true, tw2, true, xw2, local, false);
                local = __builtin_amdgcn_sudot4(true, tw3, true, xw3, local, false);
                acc[r] += local;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int32_t v = acc[r];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v += __shfl_xor(v, offset);
        }
        const int my_row = warp_row_base + r;
        if (lane == 0 && my_row < M) {
            y[my_row] = (float)v * x_scale * scales[my_row];
        }
    }
}

extern "C" void ternary_gemv_phase5_halo_launch(
    const void* packed, const void* x_i8, float x_scale,
    const void* scales, void* y,
    int M, int K, void* stream)
{
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL(ternary_gemv_phase5_halo_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const uint32_t*>(packed),
                       static_cast<const int8_t*>(x_i8),
                       x_scale,
                       static_cast<const float*>(scales),
                       static_cast<float*>(y),
                       M, K);
}

// ─── FP16-output fused variant (halo-ai Lane A) ──────────────────────────────
// Identical inner loop; final output cast to __half inline, eliminating the
// separate fp32_to_fp16_kernel pass that previously fired 1:1 with this GEMV.
// Saves ~1.5% of GPU time + O(53k) kernel launches per 256-token decode.
__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void ternary_gemv_phase5_halo_f16_kernel(
    const uint32_t* __restrict__ packed,
    const int8_t*   __restrict__ x_i8,
    float                       x_scale,
    const float*    __restrict__ scales,
    __half*         __restrict__ y_f16,
    int M, int K)
{
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int wave = tid >> 5;

    const int row_base = blockIdx.x * ROWS_PER_BLOCK;
    const int warp_row_base = row_base + wave * ROWS_PER_WARP;

    const int packed_k = K / VALS_PER_U32;

    __shared__ int8_t x_shared[LDS_TILE_I8];

    int32_t acc[ROWS_PER_WARP] = {};

    for (int k_base = 0; k_base < K; k_base += LDS_TILE_I8) {
        __syncthreads();
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

        const int tile_u32_start = k_base / VALS_PER_U32;
        const int tile_u32_count = LDS_TILE_I8 / VALS_PER_U32;

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; ++r) {
            const int my_row = warp_row_base + r;
            if (my_row >= M) continue;
            const uint32_t* row_weights = packed + (size_t)my_row * packed_k;

            for (int u = lane; u < tile_u32_count; u += WARP_SIZE) {
                int global_u32 = tile_u32_start + u;
                if (global_u32 >= packed_k) break;
                const uint32_t word = row_weights[global_u32];
                const int x_offset  = u * VALS_PER_U32;

                const uint32_t* xpack = reinterpret_cast<const uint32_t*>(&x_shared[x_offset]);
                const uint32_t xw0 = xpack[0];
                const uint32_t xw1 = xpack[1];
                const uint32_t xw2 = xpack[2];
                const uint32_t xw3 = xpack[3];

                const uint32_t tw0 = unpack_ternary_halo_4(word, 0);
                const uint32_t tw1 = unpack_ternary_halo_4(word, 4);
                const uint32_t tw2 = unpack_ternary_halo_4(word, 8);
                const uint32_t tw3 = unpack_ternary_halo_4(word, 12);

                int32_t local = 0;
                local = __builtin_amdgcn_sudot4(true, tw0, true, xw0, local, false);
                local = __builtin_amdgcn_sudot4(true, tw1, true, xw1, local, false);
                local = __builtin_amdgcn_sudot4(true, tw2, true, xw2, local, false);
                local = __builtin_amdgcn_sudot4(true, tw3, true, xw3, local, false);
                acc[r] += local;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int32_t v = acc[r];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v += __shfl_xor(v, offset);
        }
        const int my_row = warp_row_base + r;
        if (lane == 0 && my_row < M) {
            y_f16[my_row] = __float2half((float)v * x_scale * scales[my_row]);
        }
    }
}

extern "C" void ternary_gemv_phase5_halo_f16_launch(
    const void* packed, const void* x_i8, float x_scale,
    const void* scales, void* y_f16,
    int M, int K, void* stream)
{
    dim3 grid((M + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL(ternary_gemv_phase5_halo_f16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const uint32_t*>(packed),
                       static_cast<const int8_t*>(x_i8),
                       x_scale,
                       static_cast<const float*>(scales),
                       static_cast<__half*>(y_f16),
                       M, K);
}
