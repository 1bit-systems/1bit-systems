// Fused Wave32 Ternary GEMV for gfx1151 (RDNA 3.5)
// No dequantization. No floating-point multiply in the inner loop.
// Just add, subtract, or skip based on ternary encoding.
//
// "There is no spoon." — Neo
//
// Encoding: 2 bits per value, 16 values per uint32
//   0b00 = 0 (skip)
//   0b01 = +1 (add)
//   0b10 = -1 (subtract)
//
// Grid:  (M, 1, 1) — one block per output row
// Block: (128, 1, 1) — 4 Wave32 warps
//
// Performance target: 250+ tok/s on BitNet-2B-4T (gfx1151)

#include <hip/hip_runtime.h>

// 128 threads = 4 Wave32 warps — optimal for RDNA 3.5
#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// Each uint32 holds 16 ternary values (2 bits each)
#define VALS_PER_U32 16

// LDS tile for activation vector — 512 floats shared across all warps
#define LDS_TILE 512

__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void ternary_gemv_kernel(
    const uint32_t* __restrict__ packed,  // [M, K/16] packed ternary weights
    const float* __restrict__ x,          // [K] activation vector
    const float* __restrict__ scales,     // [M] per-row scales
    float* __restrict__ y,                // [M] output
    int M,                                // output dimension (rows)
    int K)                                // input dimension (cols)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_id = tid / WARP_SIZE;

    const int packed_k = K / VALS_PER_U32;  // uint32s per row
    const uint32_t* row_weights = packed + (size_t)row * packed_k;
    const float scale = scales[row];

    __shared__ float x_shared[LDS_TILE];

    float acc = 0.0f;

    // Process K in tiles of LDS_TILE
    for (int k_base = 0; k_base < K; k_base += LDS_TILE) {
        // Cooperative load of activation tile into LDS
        // 128 threads load 512 floats = 4 per thread
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < LDS_TILE / BLOCK_SIZE; i++) {
            int k_idx = k_base + tid + i * BLOCK_SIZE;
            int lds_idx = tid + i * BLOCK_SIZE;
            x_shared[lds_idx] = (k_idx < K) ? x[k_idx] : 0.0f;
        }
        __syncthreads();

        // Each thread processes its share of packed uint32s in this tile
        // Tile covers k_base to k_base+LDS_TILE
        // That's LDS_TILE/16 = 32 uint32s to process across 128 threads
        int tile_u32_start = k_base / VALS_PER_U32;
        int tile_u32_count = LDS_TILE / VALS_PER_U32;  // 32

        for (int u = tid; u < tile_u32_count; u += BLOCK_SIZE) {
            int global_u32 = tile_u32_start + u;
            if (global_u32 >= packed_k) break;

            uint32_t word = row_weights[global_u32];

            // Offset into x_shared for this uint32's 16 values
            int x_offset = u * VALS_PER_U32;

            // Unroll 16 ternary values from this uint32
            // Each value is 2 bits: bits [2i+1 : 2i]
            #pragma unroll
            for (int v = 0; v < VALS_PER_U32; v++) {
                uint32_t bits = (word >> (v * 2)) & 0x3;
                float xv = x_shared[x_offset + v];

                // Branchless: multiply by (bits==1) - (bits==2)
                // bits=00(0): 0-0 = 0, skip
                // bits=01(1): 1-0 = +1, add
                // bits=10(2): 0-1 = -1, subtract
                // bits=11(3): shouldn't exist, but 1-1 = 0 (safe)
                float sign = (float)(bits == 1) - (float)(bits == 2);
                acc += sign * xv;
            }
        }
    }

    // Wave32 reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor(acc, offset);
    }

    // Cross-warp reduction via LDS
    __shared__ float warp_results[NUM_WARPS];

    if (lane == 0) {
        warp_results[warp_id] = acc;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        acc = (lane < NUM_WARPS) ? warp_results[lane] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_xor(acc, offset);
        }

        // Lane 0 writes final result
        if (lane == 0) {
            y[row] = acc * scale;
        }
    }
}

// Batched version for prefill — x is [B, K]
__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void ternary_gemv_batched_kernel(
    const uint32_t* __restrict__ packed,  // [M, K/16]
    const float* __restrict__ x,          // [B, K]
    const float* __restrict__ scales,     // [M]
    float* __restrict__ y,                // [B, M]
    int M, int K, int B)
{
    const int row = blockIdx.x;
    const int batch = blockIdx.y;
    if (row >= M || batch >= B) return;

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_id = tid / WARP_SIZE;

    const int packed_k = K / VALS_PER_U32;
    const uint32_t* row_weights = packed + (size_t)row * packed_k;
    const float* x_batch = x + (size_t)batch * K;
    const float scale = scales[row];

    __shared__ float x_shared[LDS_TILE];

    float acc = 0.0f;

    for (int k_base = 0; k_base < K; k_base += LDS_TILE) {
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < LDS_TILE / BLOCK_SIZE; i++) {
            int k_idx = k_base + tid + i * BLOCK_SIZE;
            int lds_idx = tid + i * BLOCK_SIZE;
            x_shared[lds_idx] = (k_idx < K) ? x_batch[k_idx] : 0.0f;
        }
        __syncthreads();

        int tile_u32_start = k_base / VALS_PER_U32;
        int tile_u32_count = LDS_TILE / VALS_PER_U32;

        for (int u = tid; u < tile_u32_count; u += BLOCK_SIZE) {
            int global_u32 = tile_u32_start + u;
            if (global_u32 >= packed_k) break;

            uint32_t word = row_weights[global_u32];
            int x_offset = u * VALS_PER_U32;

            #pragma unroll
            for (int v = 0; v < VALS_PER_U32; v++) {
                uint32_t bits = (word >> (v * 2)) & 0x3;
                float xv = x_shared[x_offset + v];
                float sign = (float)(bits == 1) - (float)(bits == 2);
                acc += sign * xv;
            }
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor(acc, offset);
    }

    __shared__ float warp_results[NUM_WARPS];
    if (lane == 0) {
        warp_results[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        acc = (lane < NUM_WARPS) ? warp_results[lane] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc += __shfl_xor(acc, offset);
        }
        if (lane == 0) {
            y[(size_t)batch * M + row] = acc * scale;
        }
    }
}

// Host-side launch wrapper
extern "C" {

void ternary_gemv(
    const uint32_t* packed, const float* x, const float* scales,
    float* y, int M, int K, hipStream_t stream)
{
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);
    hipLaunchKernelGGL(ternary_gemv_kernel, grid, block, 0, stream,
        packed, x, scales, y, M, K);
}

void ternary_gemv_batched(
    const uint32_t* packed, const float* x, const float* scales,
    float* y, int M, int K, int B, hipStream_t stream)
{
    dim3 grid(M, B);
    dim3 block(BLOCK_SIZE);
    hipLaunchKernelGGL(ternary_gemv_batched_kernel, grid, block, 0, stream,
        packed, x, scales, y, M, K, B);
}

} // extern "C"
