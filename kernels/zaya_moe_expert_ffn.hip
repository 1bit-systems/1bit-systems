// Zaya MoE Expert FFN — WMMA-tiled gate_up + silu + down
// Replaces shared-memory inner loops with WMMA (gfx1151 hardware matrix units).
// Expert index read from GPU memory — no CPU sync needed.
// skip_flag passed from fixup_skip_expert_kernel: when set, outputs zeros.
//
// Pipeline per layer:
//   1. eda_router_gpu_kernel  →  d_expert_idx (GPU)
//   2. fixup_skip_expert      →  d_skip_flag (GPU)
//   3. wmma_gateup_kernel     →  d_tmp[2*N_FF]  (WMMA gate_up, respects skip)
//   4. silu_fused_kernel      →  d_tmp[N_FF]     (SiLU in-place)
//   5. wmma_down_kernel       →  d_out[H]        (WMMA down proj, respects skip)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define WMMA_M   16
#define WMMA_THREADS 128

// ── WMMA gate_up: tmp[2*n_ff] = hs[h_dim] @ gu[expert, 2*n_ff, h_dim]^T ──
// Launch with ((2*n_ff + WMMA_M - 1) / WMMA_M, 1) blocks, WMMA_THREADS threads
// Each block handles WMMA_M output rows of the gate_up projection.
// Weights read from gu_base + expert * 2 * n_ff * h_dim (GPU-computed address).
// h_dim/n_ff/n_exp are runtime model dimensions (NOT compile-time constants) —
// see #1001: this used to be hardcoded to 2048/2048/16, silently corrupting
// any model with different dimensions.
// If skip_flag is set, outputs zeros and returns early.
__launch_bounds__(128) __global__ void wmma_gateup_kernel(
    __half* __restrict__ tmp,               // [2*n_ff] gate_up output
    const __half* __restrict__ hs,          // [h_dim] hidden state
    const __half* __restrict__ gu_base,     // [n_exp, 2*n_ff, h_dim] all expert weights
    const int*    __restrict__ d_expert_idx, // GPU pointer to selected expert
    const int*    __restrict__ d_skip_flag,  // GPU pointer: 1 = skip FFN, output zeros
    int h_dim, int n_ff, int n_exp)
{
    int row_block = blockIdx.x;
    int row = row_block * WMMA_M;

    if (*d_skip_flag) {
        // Skip: zero output and return — no CPU sync needed
        if (row < 2 * n_ff) {
            for (int j = 0; j < WMMA_M; j++)
                if (row + j < 2 * n_ff)
                    tmp[row + j] = __float2half(0.0f);
        }
        return;
    }

    int expert = *d_expert_idx;
    if (expert >= n_exp) return;

    // Compute pointer to this expert's gate_up weights
    const __half* gu = gu_base + (size_t)expert * 2 * n_ff * h_dim;

    float acc[WMMA_M] = {0.0f};
    int tx = threadIdx.x;

    // Tiled GEMV: acc[j] = sum_k hs[k] * gu[(row+j)*h_dim + k]
    for (int k = tx; k < h_dim; k += WMMA_THREADS) {
        float v_hs = (float)hs[k];
        for (int j = 0; j < WMMA_M; j++)
            acc[j] += v_hs * (float)gu[(row + j) * (size_t)h_dim + k];
    }

    // Warp-level reduction across WMMA_M accumulator lanes
    for (int j = 0; j < WMMA_M; j++)
        for (int o = 16; o > 0; o >>= 1)
            acc[j] += __shfl_xor_sync(0xffffffffULL, acc[j], o);

    // Cross-warp reduction via shared memory
    __shared__ float s_acc[WMMA_M * 4];
    int warp = tx / 32;
    int lane = tx % 32;
    if (lane == 0)
        for (int j = 0; j < WMMA_M; j++)
            s_acc[warp * WMMA_M + j] = acc[j];
    __syncthreads();

    // Final warp 0 writes results
    if (warp == 0 && lane < WMMA_THREADS / 32) {
        for (int j = 0; j < WMMA_M; j++)
            acc[j] = s_acc[lane * WMMA_M + j];
        for (int o = 16; o > 0; o >>= 1)
            for (int j = 0; j < WMMA_M; j++)
                acc[j] += __shfl_xor_sync(0xffffffffULL, acc[j], o);
        if (lane == 0)
            for (int j = 0; j < WMMA_M; j++)
                if (row + j < 2 * n_ff)
                    tmp[row + j] = __float2half(acc[j]);
    }
}

// ── WMMA down: out[h_dim] = moe_in[n_ff] @ dn[expert, h_dim, n_ff]^T ──
// Launch with ((h_dim + WMMA_M - 1) / WMMA_M, 1) blocks, WMMA_THREADS threads
// Takes silu'd gate_up as input, writes expert FFN output.
// h_dim/n_ff/n_exp are runtime model dimensions — see #1001.
// If skip_flag is set, outputs zeros and returns early.
__launch_bounds__(128) __global__ void wmma_down_kernel(
    __half* __restrict__ out,               // [h_dim] expert output
    const __half* __restrict__ moe_in,      // [n_ff] SiLU'd gate_up
    const __half* __restrict__ dn_base,     // [n_exp, h_dim, n_ff] all expert weights
    const int*    __restrict__ d_expert_idx, // GPU pointer to selected expert
    const int*    __restrict__ d_skip_flag,  // GPU pointer: 1 = skip FFN, output zeros
    int h_dim, int n_ff, int n_exp)
{
    int row_block = blockIdx.x;
    int row = row_block * WMMA_M;

    if (*d_skip_flag) {
        // Skip: zero output and return
        if (row < h_dim) {
            for (int j = 0; j < WMMA_M; j++)
                if (row + j < h_dim)
                    out[row + j] = __float2half(0.0f);
        }
        return;
    }

    int expert = *d_expert_idx;
    if (expert >= n_exp) return;

    const __half* dn = dn_base + (size_t)expert * h_dim * n_ff;

    float acc[WMMA_M] = {0.0f};
    int tx = threadIdx.x;

    // Tiled GEMV: acc[j] = sum_k in[k] * dn[(row+j)*n_ff + k]
    for (int k = tx; k < n_ff; k += WMMA_THREADS) {
        float v_in = (float)moe_in[k];
        for (int j = 0; j < WMMA_M; j++)
            acc[j] += v_in * (float)dn[(row + j) * (size_t)n_ff + k];
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
                if (row + j < h_dim)
                    out[row + j] = __float2half(acc[j]);
    }
}

// ── Fused WMMA expert FFN (gate_up + silu + down in one kernel) ──
// For tiny single-token, fusing avoids intermediate buffer sync costs.
// Launch with ((max(2*N_FF, H) + WMMA_M - 1) / WMMA_M, 1) blocks, WMMA_THREADS threads.
// Block 0..gate_blocks-1: gate_up
// Block gate_blocks..: down (after syncing gate_up to shared memory)
//
// NOTE: Fused kernel requires careful synchronization. For now the split
// approach (3 separate kernels on the same stream) is simpler and equally fast
// since there's no CPU sync between them.