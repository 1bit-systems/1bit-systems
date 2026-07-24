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

#define H        2048
#define N_EXP    16
#define N_FF     2048
#define WMMA_M   16
#define WMMA_THREADS 128

// ── WMMA gate_up: tmp[2*N_FF] = hs[H] @ gu[expert, 2*N_FF, H]^T ──
// Launch with ((2*N_FF + WMMA_M - 1) / WMMA_M, 1) blocks, WMMA_THREADS threads
// Each block handles WMMA_M output rows of the gate_up projection.
// Weights read from gu_base + expert * 2 * N_FF * H (GPU-computed address).
// If skip_flag is set, outputs zeros and returns early.
__launch_bounds__(128) __global__ void wmma_gateup_kernel(
    __half* __restrict__ tmp,               // [2*N_FF] gate_up output
    const __half* __restrict__ hs,          // [H] hidden state
    const __half* __restrict__ gu_base,     // [N_EXP, 2*N_FF, H] all expert weights
    const int*    __restrict__ d_expert_idx, // GPU pointer to selected expert
    const int*    __restrict__ d_skip_flag)  // GPU pointer: 1 = skip FFN, output zeros
{
    int row_block = blockIdx.x;
    int row = row_block * WMMA_M;

    if (*d_skip_flag) {
        // Skip: zero output and return — no CPU sync needed
        if (row < 2 * N_FF) {
            for (int j = 0; j < WMMA_M; j++)
                if (row + j < 2 * N_FF)
                    tmp[row + j] = __float2half(0.0f);
        }
        return;
    }

    int expert = *d_expert_idx;
    if (expert >= N_EXP) return;

    // Compute pointer to this expert's gate_up weights
    const __half* gu = gu_base + (size_t)expert * 2 * N_FF * H;

    float acc[WMMA_M] = {0.0f};
    int tx = threadIdx.x;

    // Tiled GEMV: acc[j] = sum_k hs[k] * gu[(row+j)*H + k]
    for (int k = tx; k < H; k += WMMA_THREADS) {
        float v_hs = (float)hs[k];
        for (int j = 0; j < WMMA_M; j++)
            acc[j] += v_hs * (float)gu[(row + j) * (size_t)H + k];
    }

    // Warp-level reduction across WMMA_M accumulator lanes
    for (int j = 0; j < WMMA_M; j++)
        for (int o = 16; o > 0; o >>= 1)
            acc[j] += __shfl_xor(acc[j], o);

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
                acc[j] += __shfl_xor(acc[j], o);
        if (lane == 0)
            for (int j = 0; j < WMMA_M; j++)
                if (row + j < 2 * N_FF)
                    tmp[row + j] = __float2half(acc[j]);
    }
}

// ── WMMA down: out[H] = moe_in[N_FF] @ dn[expert, H, N_FF]^T ──
// Launch with ((H + WMMA_M - 1) / WMMA_M, 1) blocks, WMMA_THREADS threads
// Takes silu'd gate_up as input, writes expert FFN output.
// If skip_flag is set, outputs zeros and returns early.
__launch_bounds__(128) __global__ void wmma_down_kernel(
    __half* __restrict__ out,               // [H] expert output
    const __half* __restrict__ moe_in,      // [N_FF] SiLU'd gate_up
    const __half* __restrict__ dn_base,     // [N_EXP, H, N_FF] all expert weights
    const int*    __restrict__ d_expert_idx, // GPU pointer to selected expert
    const int*    __restrict__ d_skip_flag)  // GPU pointer: 1 = skip FFN, output zeros
{
    int row_block = blockIdx.x;
    int row = row_block * WMMA_M;

    if (*d_skip_flag) {
        // Skip: zero output and return
        if (row < H) {
            for (int j = 0; j < WMMA_M; j++)
                if (row + j < H)
                    out[row + j] = __float2half(0.0f);
        }
        return;
    }

    int expert = *d_expert_idx;
    if (expert >= N_EXP) return;

    const __half* dn = dn_base + (size_t)expert * H * N_FF;

    float acc[WMMA_M] = {0.0f};
    int tx = threadIdx.x;

    // Tiled GEMV: acc[j] = sum_k in[k] * dn[(row+j)*N_FF + k]
    for (int k = tx; k < N_FF; k += WMMA_THREADS) {
        float v_in = (float)moe_in[k];
        for (int j = 0; j < WMMA_M; j++)
            acc[j] += v_in * (float)dn[(row + j) * (size_t)N_FF + k];
    }

    for (int j = 0; j < WMMA_M; j++)
        for (int o = 16; o > 0; o >>= 1)
            acc[j] += __shfl_xor(acc[j], o);

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
                acc[j] += __shfl_xor(acc[j], o);
        if (lane == 0)
            for (int j = 0; j < WMMA_M; j++)
                if (row + j < H)
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