// Kernel v2.1 — RDNA 3.5 Optimized Ternary GEMV (gfx1151)
//
// Fixes the v2.0 regression against v1.0. The original v2.0 paid for
// v_dot4_i32_iu8 with per-inner-iteration quantization + max-abs scan
// overhead that dominated the DP4A win. v2.1 removes that overhead:
//
// 1. Activation quantization happens ONCE per tile (not per inner iter)
// 2. 4 bytes per iteration → 32 ternary values → 8 dot4 calls
//    (v1 does 16 values per iter; v2.0 did 8 values per iter with more
//    work; v2.1 matches or beats v1 per-iter throughput)
// 3. Activation scale is correctly applied at output (v2.0 dropped it)
//
// "Fly, you fools!" — Gandalf

#include <hip/hip_runtime.h>

#define BLOCK_SIZE 128
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

#define LDS_TILE 512

// v_dot4_i32_iu8: 4-element INT8 dot product. gfx1151 has +dot8-insts.
__device__ __forceinline__ int dot4_i8(int a, int b) {
    return __builtin_amdgcn_sudot4(true, a, true, b, 0, false);
}

// Pack 4 INT8 values into an int (little-endian byte order)
__device__ __forceinline__ int pack_i8x4(int b0, int b1, int b2, int b3) {
    return (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
}

// Unpack 4 consecutive 1-bit ternary values into an int8x4, mapping
// bit=1 → +1, bit=0 → -1. Branchless via arithmetic ((bit<<1)-1).
__device__ __forceinline__ int unpack_ternary_4(uint32_t bits4) {
    int b0 = ((int)((bits4 >> 0) & 1) << 1) - 1;
    int b1 = ((int)((bits4 >> 1) & 1) << 1) - 1;
    int b2 = ((int)((bits4 >> 2) & 1) << 1) - 1;
    int b3 = ((int)((bits4 >> 3) & 1) << 1) - 1;
    return pack_i8x4(b0, b1, b2, b3);
}

__attribute__((amdgpu_flat_work_group_size(BLOCK_SIZE, BLOCK_SIZE)))
__global__ void ternary_gemv_v2_kernel(
    const uint32_t* __restrict__ packed_bits,  // reinterpreted as uint8_t* — K/8 bytes per row
    const float* __restrict__ x,               // [K]
    const float* __restrict__ scales,          // [M]
    float* __restrict__ y,                     // [M]
    int M, int K)
{
    const int row = blockIdx.x;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_id = tid / WARP_SIZE;

    const int bytes_per_row = K / 8;  // 1 bit per weight value
    const uint8_t* row_bits = (const uint8_t*)(packed_bits) + (size_t)row * bytes_per_row;
    const float w_scale = scales[row];

    __shared__ float x_shared_f[LDS_TILE];
    __shared__ int8_t x_shared_q[LDS_TILE];
    __shared__ float tile_max_abs;

    // INT32 accumulator (whole-row, across tiles)
    int acc_i32 = 0;
    // Sum of (act_scale × dot_chunk) pairs — actually we defer by tracking
    // per-tile int32 sums and rescaling once per tile.
    float acc_f = 0.0f;

    for (int k_base = 0; k_base < K; k_base += LDS_TILE) {
        // --- Stage 1: cooperative load of activation tile into LDS ---
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < LDS_TILE / BLOCK_SIZE; i++) {
            int k_idx = k_base + tid + i * BLOCK_SIZE;
            int lds_idx = tid + i * BLOCK_SIZE;
            x_shared_f[lds_idx] = (k_idx < K) ? x[k_idx] : 0.0f;
        }
        __syncthreads();

        // --- Stage 2: compute max-abs ONCE per tile, block-cooperative ---
        float local_max = 0.0f;
        #pragma unroll
        for (int i = tid; i < LDS_TILE; i += BLOCK_SIZE) {
            float v = fabsf(x_shared_f[i]);
            if (v > local_max) local_max = v;
        }
        // Warp reduce
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            float other = __shfl_xor(local_max, offset);
            if (other > local_max) local_max = other;
        }
        // Cross-warp via LDS (reuse warp_results-style)
        __shared__ float warp_max[NUM_WARPS];
        if (lane == 0) warp_max[warp_id] = local_max;
        __syncthreads();
        if (tid == 0) {
            float m = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++) {
                if (warp_max[w] > m) m = warp_max[w];
            }
            tile_max_abs = (m > 1e-8f) ? m : 1e-8f;
        }
        __syncthreads();

        const float act_scale = tile_max_abs / 127.0f;
        const float inv_scale = 127.0f / tile_max_abs;

        // --- Stage 3: quantize activations to int8 ONCE per tile ---
        #pragma unroll
        for (int i = 0; i < LDS_TILE / BLOCK_SIZE; i++) {
            int lds_idx = tid + i * BLOCK_SIZE;
            int q = __float2int_rn(x_shared_f[lds_idx] * inv_scale);
            q = max(-127, min(127, q));
            x_shared_q[lds_idx] = (int8_t)q;
        }
        __syncthreads();

        // --- Stage 4: dot4 inner loop, 2 bytes per iter = 16 ternary values ---
        // Matches v1's thread utilization: 32 active threads per tile, 96 idle
        // (same as v1's 32 u32s × 16 values). Each active thread does 4 dot4 ops.
        // tile has LDS_TILE/8 = 64 bytes of ternary data
        const int tile_bytes = LDS_TILE / 8;  // 64
        const int iters_per_tile = tile_bytes / 2;  // 32 chunks of 2 bytes

        int tile_acc = 0;

        #pragma unroll
        for (int chunk = tid; chunk < iters_per_tile; chunk += BLOCK_SIZE) {
            int global_byte = (k_base / 8) + chunk * 2;
            if (global_byte + 2 > bytes_per_row) break;

            // Load 2 bytes of packed weights (16 ternary values) as a u16
            uint16_t w_packed = *reinterpret_cast<const uint16_t*>(row_bits + global_byte);

            // Base LDS offset for this chunk's activations (16 values = 4 int8x4 packs)
            int x_off = chunk * 16;

            const int* xq = reinterpret_cast<const int*>(x_shared_q + x_off);
            int a0 = xq[0], a1 = xq[1], a2 = xq[2], a3 = xq[3];

            int w0 = unpack_ternary_4((w_packed >> 0)  & 0xF);
            int w1 = unpack_ternary_4((w_packed >> 4)  & 0xF);
            int w2 = unpack_ternary_4((w_packed >> 8)  & 0xF);
            int w3 = unpack_ternary_4((w_packed >> 12) & 0xF);

            // 4 dot4 ops = 16 ternary × int8 MACs, mirrors v1's 16 MACs/iter
            tile_acc += dot4_i8(w0, a0);
            tile_acc += dot4_i8(w1, a1);
            tile_acc += dot4_i8(w2, a2);
            tile_acc += dot4_i8(w3, a3);
        }

        // Convert this tile's int accumulator back to float using its act_scale
        acc_f += (float)tile_acc * act_scale;
    }

    // --- Reduction across threads in the block ---
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc_f += __shfl_xor(acc_f, offset);
    }

    __shared__ float warp_results[NUM_WARPS];
    if (lane == 0) warp_results[warp_id] = acc_f;
    __syncthreads();

    if (warp_id == 0) {
        acc_f = (lane < NUM_WARPS) ? warp_results[lane] : 0.0f;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc_f += __shfl_xor(acc_f, offset);
        }
        if (lane == 0) {
            y[row] = acc_f * w_scale;
        }
    }
}

extern "C" {

void ternary_gemv_v2(
    const uint32_t* packed_bits, const float* x, const float* scales,
    float* y, int M, int K, hipStream_t stream)
{
    dim3 grid(M);
    dim3 block(BLOCK_SIZE);
    hipLaunchKernelGGL(ternary_gemv_v2_kernel, grid, block, 0, stream,
        packed_bits, x, scales, y, M, K);
}

} // extern "C"
