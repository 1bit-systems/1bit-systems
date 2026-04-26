// Phase 5 decode — ternary × INT8 activation GEMV on gfx1151.
//
// v1 GEMV reads FP32 activations (4 bytes each). This variant reads INT8
// activations (1 byte each), giving 4x less A memory traffic on the same
// ternary encoding. For memory-bound decode (batch=1), this is where the
// real ceiling moves.
//
// Inputs:
//   packed  : [M, K/16] packed ternary weights (2 bits each — same v1 encoding)
//   x_i8    : [K] INT8 activations (user pre-quantizes FP16 A with row scale)
//   x_scale : single FP32 activation scale (s.t. real_A = x_i8 * x_scale)
//   scales  : [M] per-row weight scales
//   y       : [M] FP32 output (real value, post-dequant)
//
// No multiplications on hot path — just conditional INT32 add/subtract based
// on ternary bits. Matches the BitNet 1.58 arxiv paper's insight.

#include <hip/hip_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)
#define VALS_PER_U32 16
// INT8 fits 4x more per LDS byte than FP32 — use larger tile
#define LDS_TILE_I8 2048

__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void ternary_gemv_phase5_kernel(
    const uint32_t* __restrict__ packed,   // [M, K/16]
    const int8_t*   __restrict__ x_i8,     // [K]
    float                       x_scale,   // scalar
    const float*    __restrict__ scales,   // [M]
    float*          __restrict__ y,        // [M]
    int M,
    int K)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_id = tid / WARP_SIZE;

    const int packed_k = K / VALS_PER_U32;
    const uint32_t* row_weights = packed + (size_t)row * packed_k;
    const float scale = scales[row];

    __shared__ int8_t x_shared[LDS_TILE_I8];

    int32_t acc = 0;

    for (int k_base = 0; k_base < K; k_base += LDS_TILE_I8) {
        __syncthreads();

        // Cooperative INT8 load: 2048 bytes / 128 threads = 16 bytes each.
        // Vectorize via uint4 (128-bit) loads for coalesced global access.
        using v16 = __attribute__((ext_vector_type(16))) int8_t;
        const int k_start = k_base + tid * 16;
        v16* __restrict__ lds_v = reinterpret_cast<v16*>(x_shared) + tid;
        if (k_start + 16 <= K) {
            const v16* __restrict__ src_v = reinterpret_cast<const v16*>(x_i8 + k_start);
            *lds_v = *src_v;
        } else {
            // Tail: scalar for the last partial vector
            int8_t tmp[16];
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                int k = k_start + b;
                tmp[b] = (k < K) ? x_i8[k] : (int8_t)0;
            }
            #pragma unroll
            for (int b = 0; b < 16; ++b) x_shared[tid * 16 + b] = tmp[b];
        }
        __syncthreads();

        // Each thread processes its slice of this tile's packed uint32s.
        // tile covers LDS_TILE_I8 K values = LDS_TILE_I8 / 16 = 128 uint32s.
        // With 128 threads that's 1 uint32 per thread per tile — perfect 1:1.
        const int tile_u32_start = k_base / VALS_PER_U32;
        const int tile_u32_count = LDS_TILE_I8 / VALS_PER_U32;

        for (int u = tid; u < tile_u32_count; u += BLOCK_SIZE) {
            int global_u32 = tile_u32_start + u;
            if (global_u32 >= packed_k) break;

            const uint32_t word = row_weights[global_u32];
            const int x_offset = u * VALS_PER_U32;

            #pragma unroll
            for (int v = 0; v < VALS_PER_U32; v++) {
                const uint32_t bits = (word >> (v * 2)) & 0x3;
                const int xv = (int)x_shared[x_offset + v];

                // Branchless ternary accumulator.
                //   bits=00 -> +0
                //   bits=01 -> +xv
                //   bits=10 -> -xv
                //   bits=11 -> +0 (safe, shouldn't exist)
                const int sign = (int)(bits == 1) - (int)(bits == 2);
                acc += sign * xv;
            }
        }
    }

    // Wave32 reduction (INT32 add via shfl_xor).
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor(acc, offset);
    }

    // Cross-warp reduction via LDS.
    __shared__ int32_t warp_results[NUM_WARPS];

    if (lane == 0) warp_results[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        acc = (lane < NUM_WARPS) ? warp_results[lane] : 0;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_xor(acc, offset);
        }
        if (lane == 0) {
            // Dequant: real_y = acc * x_scale * weight_row_scale
            y[row] = (float)acc * x_scale * scale;
        }
    }
}

extern "C" {

void ternary_gemv_phase5_launch(
    const void* packed, const void* x_i8, float x_scale,
    const void* scales, void* y,
    int M, int K, void* stream)
{
    dim3 grid(M, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    hipLaunchKernelGGL(ternary_gemv_phase5_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const uint32_t*>(packed),
                       static_cast<const int8_t*>(x_i8),
                       x_scale,
                       static_cast<const float*>(scales),
                       static_cast<float*>(y),
                       M, K);
}

}  // extern "C"
