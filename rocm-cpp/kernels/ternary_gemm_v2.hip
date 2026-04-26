// Kernel v2.2 GEMM — Ternary prefill, 4 waves per block, 64x64 output tile
//
// v2.1 had the WMMA op correct but 1 wave per 16x16 tile — launch overhead
// dominated. v2.2 bumps to 4 waves per block, each wave producing a 16x64
// output stripe, for a 64x64 output tile per block.
//
//   Grid:  (M/64, B/64)
//   Block: (128, 1, 1)  — 4 Wave32 warps
//
// Each K-tile iteration:
//   1. Cooperative LDS load of A_slab[64 x 16] (weights, int8) and
//      B_slab[16 x 64] (activations, int8) — 256 int8 per thread.
//   2. Wave w (w = 0..3) loads its row stripe A_slab[w*16 : (w+1)*16, :].
//   3. Wave w does 4 WMMA mma_syncs, one per column stripe of B_slab.
//   4. Float float_acc[4][num_elements] folds in the per-K-tile activation
//      scale so the INT32 accumulator never overflows across large K.
//
// Single block-wide activation scale per K tile (not per WMMA fragment) so
// we pay the max-abs scan once per 16-K slice instead of 4 times.
//
// "One ring to find them." — Gandalf

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

#define WARP_SIZE 32
#define NUM_WARPS 4
#define BLOCK_SIZE (WARP_SIZE * NUM_WARPS)  // 128

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define TILE_M 64   // rows per block
#define TILE_N 64   // cols per block
#define TILES_PER_WAVE_N (TILE_N / WMMA_N)  // 4

__device__ __forceinline__ int quant_i8(float v, float inv_scale) {
    int q = __float2int_rn(v * inv_scale);
    return max(-127, min(127, q));
}

__global__ void ternary_gemm_v2_kernel(
    const uint8_t* __restrict__ packed_bits,  // [M, K/8]
    const float*   __restrict__ x,            // [B, K]
    const float*   __restrict__ scales,       // [M]
    float*         __restrict__ y,            // [B, M]
    int M, int K, int B)
{
    const int tile_m = blockIdx.x * TILE_M;   // first row this block owns
    const int tile_b = blockIdx.y * TILE_N;   // first batch this block owns

    const int tid = threadIdx.x;              // 0..127
    const int lane = tid & (WARP_SIZE - 1);   // 0..31
    const int warp = tid / WARP_SIZE;         // 0..3

    // Per-wave row stripe of the output tile.
    const int wave_row = tile_m + warp * WMMA_M;

    const int bytes_per_row = K / 8;

    // v2.4: compute-path LDS and writeback LDS share the same physical
    // memory through a union — their lifetimes don't overlap. This caps
    // per-block LDS at max(compute, writeback) = 16 KB for c_lds, which
    // on gfx1151's 64 KB per-CU budget restores 4-block-per-CU concurrency.
    struct ComputeLDS {
        int8_t a_lds[TILE_M * WMMA_K];    // 1 KB
        int8_t b_lds[WMMA_K * TILE_N];    // 1 KB
        float  x_stage[WMMA_K * TILE_N];  // 4 KB
        float  act_scale_sh;
        float  wave_max[NUM_WARPS];
    };
    __shared__ union {
        ComputeLDS c;
        float      c_lds[TILE_M * TILE_N];  // 16 KB, writeback only
    } lds;

    int8_t* const a_lds = lds.c.a_lds;
    int8_t* const b_lds = lds.c.b_lds;
    float*  const x_stage = lds.c.x_stage;

    // Float accumulators per wave, one per col stripe (4 stripes × 8 cells/lane = 32 floats/lane).
    float float_acc[TILES_PER_WAVE_N][8] = {{0}};

    // WMMA fragments (allocated once, reused per K tile).
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, col_major> b_frag[TILES_PER_WAVE_N];
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag[TILES_PER_WAVE_N];

    for (int k_tile = 0; k_tile < K; k_tile += WMMA_K) {
        // --- Stage 1: cooperative load of A slab ---
        // A_lds[r, k] = weight at (tile_m + r, k_tile + k). 64*16 = 1024 cells,
        // 128 threads each load 8 cells.
        #pragma unroll
        for (int i = 0; i < TILE_M * WMMA_K / BLOCK_SIZE; i++) {
            int idx = tid + i * BLOCK_SIZE;
            int r = idx / WMMA_K;
            int k = idx % WMMA_K;
            int row = tile_m + r;
            int k_global = k_tile + k;
            int val = 0;
            if (row < M && k_global < K) {
                int bit = (packed_bits[(size_t)row * bytes_per_row + k_global / 8]
                           >> (k_global % 8)) & 1;
                val = bit ? 1 : -1;
            }
            a_lds[r * WMMA_K + k] = (int8_t)val;
        }

        // --- Stage 2a: load activations into LDS float stage (single GMEM read) ---
        // x_stage[k + c*WMMA_K] = x[tile_b + c, k_tile + k]  (col-major layout)
        // 1024 cells, 128 threads × 8 each.
        #pragma unroll
        for (int i = 0; i < WMMA_K * TILE_N / BLOCK_SIZE; i++) {
            int idx = tid + i * BLOCK_SIZE;
            int k = idx % WMMA_K;
            int c = idx / WMMA_K;
            int k_global = k_tile + k;
            int batch = tile_b + c;
            float v = 0.0f;
            if (k_global < K && batch < B) {
                v = x[(size_t)batch * K + k_global];
            }
            x_stage[k + c * WMMA_K] = v;
        }
        __syncthreads();

        // --- Stage 2b: block-cooperative max-abs from LDS ---
        float local_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < WMMA_K * TILE_N / BLOCK_SIZE; i++) {
            int idx = tid + i * BLOCK_SIZE;
            float v = fabsf(x_stage[idx]);
            if (v > local_max) local_max = v;
        }
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            float other = __shfl_xor(local_max, off);
            if (other > local_max) local_max = other;
        }
        if (lane == 0) lds.c.wave_max[warp] = local_max;
        __syncthreads();
        if (tid == 0) {
            float m = 0.0f;
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; w++) {
                if (lds.c.wave_max[w] > m) m = lds.c.wave_max[w];
            }
            lds.c.act_scale_sh = (m > 1e-8f) ? (m / 127.0f) : 1.0f;
        }
        __syncthreads();
        float act_scale = lds.c.act_scale_sh;
        float inv_scale = 1.0f / act_scale;

        // --- Stage 3: quantize from LDS float stage to b_lds (col-major int8) ---
        #pragma unroll
        for (int i = 0; i < WMMA_K * TILE_N / BLOCK_SIZE; i++) {
            int idx = tid + i * BLOCK_SIZE;
            int q = quant_i8(x_stage[idx], inv_scale);
            b_lds[idx] = (int8_t)q;
        }
        __syncthreads();

        // --- Stage 4: WMMA compute ---
        // Wave `warp` owns rows [warp*16 : warp*16 + 16] of A.
        load_matrix_sync(a_frag, a_lds + warp * WMMA_M * WMMA_K, WMMA_K);

        #pragma unroll
        for (int c = 0; c < TILES_PER_WAVE_N; c++) {
            load_matrix_sync(b_frag[c], b_lds + c * WMMA_N * WMMA_K, WMMA_K);
            fill_fragment(c_frag[c], 0);
            mma_sync(c_frag[c], a_frag, b_frag[c], c_frag[c]);
            // Fold into float_acc
            #pragma unroll
            for (int i = 0; i < c_frag[c].num_elements; i++) {
                float_acc[c][i] += (float)c_frag[c].x[i] * act_scale;
            }
        }

        __syncthreads();
    }

    // --- Writeback ---
    // Reuses the LDS union as a 64x64 float tile; compute-side fields are
    // dead at this point so the aliasing is safe.
    float* c_lds = lds.c_lds;
    __syncthreads();

    #pragma unroll
    for (int c = 0; c < TILES_PER_WAVE_N; c++) {
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_store;
        #pragma unroll
        for (int i = 0; i < c_store.num_elements; i++) {
            int32_t bits;
            float f = float_acc[c][i];
            __builtin_memcpy(&bits, &f, sizeof(float));
            c_store.x[i] = bits;
        }
        float* lds_ptr = c_lds + (warp * WMMA_M) * TILE_N + c * WMMA_N;
        store_matrix_sync(reinterpret_cast<int32_t*>(lds_ptr), c_store, TILE_N, mem_row_major);
    }
    __syncthreads();

    // Cooperatively write c_lds to Y with per-row scale.
    #pragma unroll
    for (int i = 0; i < TILE_M * TILE_N / BLOCK_SIZE; i++) {
        int idx = tid + i * BLOCK_SIZE;
        int r = idx / TILE_N;
        int c = idx % TILE_N;
        int row = tile_m + r;
        int batch = tile_b + c;
        if (row < M && batch < B) {
            y[(size_t)batch * M + row] = c_lds[r * TILE_N + c] * scales[row];
        }
    }
}

extern "C" {

void ternary_gemm_v2(
    const uint8_t* packed_bits, const float* x, const float* scales,
    float* y, int M, int K, int B, hipStream_t stream)
{
    dim3 grid((M + TILE_M - 1) / TILE_M, (B + TILE_N - 1) / TILE_N);
    dim3 block(BLOCK_SIZE);
    hipLaunchKernelGGL(ternary_gemm_v2_kernel, grid, block, 0, stream,
        packed_bits, x, scales, y, M, K, B);
}

} // extern "C"
