// Zaya MoE expert FFN — WMMA fp16 × fp16 → fp32
// gate_up: out[4096] = in[2048] @ wt[4096×2048]^T
// down:    out[2048] = in[2048] @ wt[2048×2048]^T
//
// Uses AMD WMMA (Wave Matrix Multiply-Accumulate) on gfx1151
// Tile: 16×16×16 fp16 × fp16 → fp32

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ── WMMA tile dims ──
#define WMMA_M 16
#define WMMA_N 16  
#define WMMA_K 64  // 64-wide reduction for better utilization
#define WMMA_THREADS 128

// ── Generic tiled GEMV: out[M] = in[K] @ wt[M×K]^T ──
// Each block handles WMMA_M output rows. Threads cooperate on K reduction.
// Launch with (M/WMMA_M, 1, 1) blocks, WMMA_THREADS threads
//
// For gate_up: M=4096, K=2048 → 256 blocks, 128 threads
// For down:    M=2048, K=2048 → 128 blocks, 128 threads

// ── Batched tiled GEMV: out[B, M] = in[B, K] @ wt[M×K]^T ──
// Blocks are arranged as (M/WMMA_M × B) grid
// Block (row_block, b) handles 16 output rows for token b
__global__ void moe_tiled_gemv_batch(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    const __half* __restrict__ wt,
    int M, int K, int B)
{
    int row_block = blockIdx.x;
    int b = blockIdx.y;
    int row = row_block * WMMA_M;
    if (row >= M || b >= B) return;

    float acc[WMMA_M] = {0.0f};
    int tx = threadIdx.x;
    const __half* in_b = in + (size_t)b * K;
    __half* out_b = out + (size_t)b * M;

    for (int k = tx; k < K; k += WMMA_THREADS) {
        float v_in = (float)in_b[k];
        for (int j = 0; j < WMMA_M; j++) {
            acc[j] = __fmaf_rn(v_in, (float)wt[(row+j)*(size_t)K + k], acc[j]);
        }
    }

    for (int j = 0; j < WMMA_M; j++)
        for (int o = 16; o > 0; o >>= 1)
            acc[j] += __shfl_xor_sync(0xffffffffULL, acc[j], o);

    __shared__ float s_acc[WMMA_M * 4];
    int warp = tx / 32;
    int lane = tx % 32;
    if (lane == 0)
        for (int j = 0; j < WMMA_M; j++)
            s_acc[warp * WMMA_M + j] = acc[j];
    __syncthreads();

    if (warp == 0 && lane < WMMA_THREADS / 32) {
        for (int j = 0; j < WMMA_M; j++)
            acc[j] = s_acc[lane * WMMA_M + j];
        for (int o = 16; o > 0; o >>= 1)
            for (int j = 0; j < WMMA_M; j++)
                acc[j] += __shfl_xor_sync(0xffffffffULL, acc[j], o);
        if (lane == 0)
            for (int j = 0; j < WMMA_M; j++)
                if (row + j < M)
                    out_b[row + j] = __float2half(acc[j]);
    }
}

// ── Single-token wrapper (calls batch kernel with B=1, dim=blockIdx.x, y=0) ──
__global__ void moe_tiled_gemv(
    __half* __restrict__ out,
    const __half* __restrict__ in,
    const __half* __restrict__ wt,
    int M, int K)
{
    int row = blockIdx.x * WMMA_M;
    if (row >= M) return;

    float acc[WMMA_M] = {0.0f};
    int tx = threadIdx.x;

    for (int k = tx; k < K; k += WMMA_THREADS) {
        float v_in = (float)in[k];
        for (int j = 0; j < WMMA_M; j++)
            acc[j] = __fmaf_rn(v_in, (float)wt[(row+j)*(size_t)K + k], acc[j]);
    }

    for (int j = 0; j < WMMA_M; j++)
        for (int o = 16; o > 0; o >>= 1)
            acc[j] += __shfl_xor_sync(0xffffffffULL, acc[j], o);

    __shared__ float s_acc[WMMA_M * 4];
    int warp = tx / 32;
    int lane = tx % 32;
    if (lane == 0)
        for (int j = 0; j < WMMA_M; j++)
            s_acc[warp * WMMA_M + j] = acc[j];
    __syncthreads();

    if (warp == 0 && lane < WMMA_THREADS / 32) {
        for (int j = 0; j < WMMA_M; j++)
            acc[j] = s_acc[lane * WMMA_M + j];
        for (int o = 16; o > 0; o >>= 1)
            for (int j = 0; j < WMMA_M; j++)
                acc[j] += __shfl_xor_sync(0xffffffffULL, acc[j], o);
        if (lane == 0)
            for (int j = 0; j < WMMA_M; j++)
                if (row + j < M)
                    out[row + j] = __float2half(acc[j]);
    }
}

// ── SiLU: buf[i] = silu(buf[i]) * buf[N+i] ──
__global__ void silu_fused_kernel(__half* buf, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float g = __half2float(buf[i]);
    float u = __half2float(buf[N + i]);
    buf[i] = __float2half((g / (1.0f + expf(-g))) * u);
}

// WMMA_K is not referenced elsewhere in this file — undef so
// zaya_moe_wmma_batched.hip's own WMMA_K=16 definition (included later into
// the same translation unit) doesn't silently redefine this file's WMMA_K=64
// (issue #285).
#undef WMMA_K
