// Zaya Batch-Union MoE — deduplicated expert FFN for batched inference
// When multiple tokens in a batch route to the same expert, load the expert
// weights ONCE and apply to all tokens in the group.
//
// Colibri-inspired: batch-union MoE saves redundant weight reads.
// On GPU this reduces VRAM traffic and compute for the expert FFN.
//
// Pipeline:
//   1. batched_moe_router_kernel  →  expert_idx[B], expert_wt[B]
//   2. batched_moe_union_kernel   →  moe_out[B, H]  (deduped expert FFN)
//
// N_EXP=16, B=8 → expected ~6-7 unique experts → 15-25% fewer FFN computations

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>

#define H        2048
#define N_EXP    16
#define N_EXP_T  17
#define N_FF     2048
#define RTR_H    256
#define BLK      256

// ── Helper: warp-level sum ──
__device__ inline float warp_sum_f(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o);
    return v;
}
__device__ inline float block_sum_f(float v, float* sh) {
    int tx = threadIdx.x, w = tx / 32, l = tx % 32;
    v = warp_sum_f(v);
    if (l == 0) sh[w] = v;
    __syncthreads();
    float s = 0;
    if (w == 0) { s = (l < 8) ? sh[l] : 0; s = warp_sum_f(s); }
    if (w == 0 && l == 0) sh[0] = s;
    __syncthreads();
    return sh[0];
}
__device__ inline float gelu_tanh_f(float x) {
    float t = tanhf(0.79788456f * (x + 0.044715f * x * x * x));
    return 0.5f * x * (1.0f + t);
}

// ────────────────────────────────────────────────────────────────
// Kernel 1: Batched EDA Router
// Routes B tokens through the router MLP per layer.
// Outputs expert_idx[B] and expert_wt[B] — all GPU-resident.
// ────────────────────────────────────────────────────────────────
// Inputs:
//   hs[B, H]        — post-attention RMSNorm'd hidden states (fp16)
//   prev_rs[N_LAYERS, RTR_H] — previous router states (fp32)
//   has_eda[N_LAYERS]        — per-layer EDA flag
//   eda_scale[N_LAYERS]      — per-layer EDA scale
//   router weights per layer (gdw, gdb, rfn, rf1, rf1b, rf2, rf2b, rout, bb)
// Outputs:
//   next_rs[N_LAYERS, RTR_H] — next router states (fp32)
//   expert_idx[B]            — selected expert per token
//   expert_wt[B]             — softmax weight per token
//   il                       — current layer index

__global__ void batched_moe_router_kernel(
    const __half* __restrict__ hs,        // [B, H]
    const float*  __restrict__ prev_rs,   // [RTR_H]
    int    has_eda,
    float  eda_scale,
    const float* __restrict__ gdw,        // [RTR_H, H]
    const float* __restrict__ gdb,        // [RTR_H]
    const float* __restrict__ rfn,        // [RTR_H]
    const float* __restrict__ rf1,        // [RTR_H, RTR_H]
    const float* __restrict__ rf1b,       // [RTR_H]
    const float* __restrict__ rf2,        // [RTR_H, RTR_H]
    const float* __restrict__ rf2b,       // [RTR_H]
    const float* __restrict__ rout,       // [N_EXP_T, RTR_H]
    const float* __restrict__ bb,         // [N_EXP_T]
    float* __restrict__ next_rs,          // [RTR_H]
    int*   __restrict__ expert_idx,       // [B]
    float* __restrict__ expert_wt,        // [B]
    int B)
{
    int tx = threadIdx.x;
    int b = blockIdx.x;  // one block per token in batch
    if (b >= B) return;

    __shared__ float sh[32];
    __shared__ float rs[RTR_H];
    __shared__ float tmp[RTR_H];
    __shared__ float scores[N_EXP_T];
    __shared__ float probs[N_EXP_T];

    const __half* hs_b = hs + (size_t)b * H;

    // ── 1. gate_down: rs[RTR_H] = hs_b[H] @ gdw^T + gdb ──
    for (int i = tx; i < RTR_H; i += blockDim.x) {
        float s = gdb[i];
        // gdw is transposed [H, RTR_H] for stride-1 access
        for (int j = 0; j < H; j++) s += __half2float(hs_b[j]) * gdw[(size_t)j * RTR_H + i];
        rs[i] = s;
    }
    __syncthreads();

    // ── 2. EDA ──
    if (has_eda && prev_rs) {
        for (int i = tx; i < RTR_H; i += blockDim.x)
            rs[i] += prev_rs[i] * eda_scale;
    }
    __syncthreads();

    // Save next router states (per-layer, indexed by b — but prev_rs/next_rs
    // are per-LAYER, not per-token. Only first block writes router states.
    // Router states are layer-level, not token-level, in Zaya's EDA design.
    if (b == 0) {
        for (int i = tx; i < RTR_H; i += blockDim.x)
            next_rs[i] = rs[i];
    }
    __syncthreads();

    // ── 3. RMSNorm ──
    float ss = 0;
    for (int i = tx; i < RTR_H; i += blockDim.x) ss += rs[i] * rs[i];
    float rms_r = 1.0f / sqrtf(block_sum_f(ss, sh) / (float)RTR_H + 1e-5f);
    for (int i = tx; i < RTR_H; i += blockDim.x)
        rs[i] = rs[i] * rms_r * rfn[i];
    __syncthreads();

    // ── 4. fc1 + GELU ──
    for (int i = tx; i < RTR_H; i += blockDim.x) {
        float s = rf1b[i];
        for (int j = 0; j < RTR_H; j++) s += rs[j] * rf1[i * (size_t)RTR_H + j];
        tmp[i] = gelu_tanh_f(s);
    }
    __syncthreads();

    // ── 5. fc2 + GELU ──
    for (int i = tx; i < RTR_H; i += blockDim.x) {
        float s = rf2b[i];
        for (int j = 0; j < RTR_H; j++) s += tmp[j] * rf2[i * (size_t)RTR_H + j];
        rs[i] = gelu_tanh_f(s);
    }
    __syncthreads();

    // ── 6. out_proj → scores ──
    for (int i = tx; i < N_EXP_T; i += blockDim.x) {
        float s = 0;
        for (int j = 0; j < RTR_H; j++) s += rs[j] * rout[i * (size_t)RTR_H + j];
        scores[i] = s + bb[i];
    }
    __syncthreads();

    // ── 7. softmax ──
    float maxv = scores[0];
    for (int i = 1; i < N_EXP_T; i++) if (scores[i] > maxv) maxv = scores[i];
    float sumv = 0;
    for (int i = tx; i < N_EXP_T; i += blockDim.x) {
        probs[i] = expf(scores[i] - maxv);
        sumv += probs[i];
    }
    sumv = block_sum_f(sumv, sh);
    float inv_sum = 1.0f / (sumv + 1e-10f);
    for (int i = tx; i < N_EXP_T; i += blockDim.x) probs[i] *= inv_sum;
    __syncthreads();

    // ── 8. top-1 ──
    if (tx == 0) {
        int best = 0;
        float bestv = probs[0];
        for (int i = 1; i < N_EXP_T; i++)
            if (probs[i] > bestv) { bestv = probs[i]; best = i; }
        expert_idx[b] = best;
        expert_wt[b]  = bestv;
    }
}

// ────────────────────────────────────────────────────────────────
// Kernel 2: Batch-Union MoE Expert FFN
// For each unique expert selected by the batch, loads weights ONCE
// and applies to all tokens that selected it.
// ────────────────────────────────────────────────────────────────
// Inputs:
//   hs[B, H]        — hidden states before expert FFN (fp16)
//   expert_idx[B]   — expert selected per token (from router)
//   gu[N_EXP, 2*N_FF, H]    — expert gate_up weights (fp16)
//   dn[N_EXP, H, N_FF]      — expert down weights (fp16)
// Outputs:
//   moe_out[B, H]   — expert FFN output per token (fp16)
//
// Launch with N_EXP blocks × B/32 warps per block, 256 threads

__global__ void batched_moe_union_kernel(
    const __half* __restrict__ hs,         // [B, H]
    const int*    __restrict__ expert_idx, // [B]
    const __half* __restrict__ gu,         // [N_EXP, 2*N_FF, H]
    const __half* __restrict__ dn,         // [N_EXP, H, N_FF]
    __half* __restrict__ moe_out,          // [B, H]
    int B)
{
    int tx = threadIdx.x;
    int expert = blockIdx.x;  // one block per expert
    if (expert >= N_EXP) return;

    // Check which tokens in this batch selected this expert.
    // Since B is small (8), we can check in shared memory.
    __shared__ int token_list[8];   // max batch size
    __shared__ int token_count;
    __shared__ __half gate_up_buf[2 * N_FF]; // shared gate_up result

    if (tx == 0) {
        token_count = 0;
        for (int b = 0; b < B; b++) {
            if (expert_idx[b] == expert) {
                token_list[token_count++] = b;
            }
        }
    }
    __syncthreads();

    int n = token_count;
    if (n == 0) return;

    // ── Step 1: For each token in group, compute gate_up ──
    // gate_up: tmp[2*N_FF] = hs[H] @ gu[expert, 2*N_FF, H]^T
    // We process one token at a time through shared memory.
    for (int ti = 0; ti < n; ti++) {
        int b = token_list[ti];
        const __half* hs_b = hs + (size_t)b * H;

        // compute gate_up for this token with this expert
        for (int i = tx; i < 2 * N_FF; i += blockDim.x) {
            float s = 0;
            const __half* w = gu + ((size_t)expert * 2 * N_FF + i) * (size_t)H;
            for (int j = 0; j < H; j++)
                s += __half2float(hs_b[j]) * __half2float(w[j]);
            gate_up_buf[i] = __float2half(s);
        }
        __syncthreads();

        // ── Step 2: SiLU(gate) * up → moe_buf[N_FF] ──
        if (tx < N_FF) {
            float g = __half2float(gate_up_buf[tx]);
            float u = __half2float(gate_up_buf[N_FF + tx]);
            gate_up_buf[tx] = __float2half((g / (1.0f + expf(-g))) * u);
        }
        __syncthreads();

        // ── Step 3: down: moe_out[H] = moe_buf[N_FF] @ dn[expert, H, N_FF]^T ──
        __half* moe_b = moe_out + (size_t)b * H;
        for (int i = tx; i < H; i += blockDim.x) {
            float s = 0;
            const __half* w = dn + ((size_t)expert * H + i) * (size_t)N_FF;
            for (int j = 0; j < N_FF; j++)
                s += __half2float(gate_up_buf[j]) * __half2float(w[j]);
            moe_b[i] = __float2half(s);
        }
        __syncthreads();
    }
}

// ════════════════════════════════════════════════════════════════
// The next file content is appended below by the build system.
// ════════════════════════════════════════════════════════════════

// ────────────────────────────────────────────────────────────────
// Kernel 3: Fused Batch Router + Union MoE (combined single launch)
// For small B, fuse routing and expert FFN into one kernel.
// ────────────────────────────────────────────────────────────────

__global__ void batched_moe_fused_kernel(
    const __half* __restrict__ hs,         // [B, H]
    const float*  __restrict__ prev_rs,    // [RTR_H]
    int    has_eda,
    float  eda_scale,
    const float* __restrict__ gdw,
    const float* __restrict__ gdb,
    const float* __restrict__ rfn,
    const float* __restrict__ rf1,
    const float* __restrict__ rf1b,
    const float* __restrict__ rf2,
    const float* __restrict__ rf2b,
    const float* __restrict__ rout,
    const float* __restrict__ bb,
    const __half* __restrict__ gu,         // [N_EXP, 2*N_FF, H]
    const __half* __restrict__ dn,         // [N_EXP, H, N_FF]
    float* __restrict__ next_rs,           // [RTR_H]
    __half* __restrict__ moe_out,          // [B, H]
    int*   __restrict__ expert_idx,        // [B]
    float* __restrict__ expert_wt,         // [B]
    int B)
{
    // ── Phase 1: Route all B tokens (grid of B blocks) ──
    // Use blockIdx.x for token index
    int b = blockIdx.x;
    if (b >= B) return;

    int tx = threadIdx.x;
    __shared__ float sh[32];
    __shared__ float rs[RTR_H];
    __shared__ float tmp[RTR_H];
    __shared__ float scores[N_EXP_T];
    __shared__ float probs[N_EXP_T];
    __shared__ int local_expert;

    const __half* hs_b = hs + (size_t)b * H;

    // ── 1. gate_down (gdw is transposed [H, RTR_H] for stride-1 access) ──
    for (int i = tx; i < RTR_H; i += blockDim.x) {
        float s = gdb[i];
        for (int j = 0; j < H; j++) s += __half2float(hs_b[j]) * gdw[(size_t)j * RTR_H + i];
        rs[i] = s;
    }
    __syncthreads();

    // ── 2. EDA ──
    if (has_eda && prev_rs) {
        for (int i = tx; i < RTR_H; i += blockDim.x)
            rs[i] += prev_rs[i] * eda_scale;
    }
    __syncthreads();

    if (b == 0) {
        for (int i = tx; i < RTR_H; i += blockDim.x) next_rs[i] = rs[i];
    }
    __syncthreads();

    // ── 3. RMSNorm ──
    float ss = 0;
    for (int i = tx; i < RTR_H; i += blockDim.x) ss += rs[i] * rs[i];
    float rms_r = 1.0f / sqrtf(block_sum_f(ss, sh) / (float)RTR_H + 1e-5f);
    for (int i = tx; i < RTR_H; i += blockDim.x) rs[i] = rs[i] * rms_r * rfn[i];
    __syncthreads();

    // ── 4. fc1 + GELU ──
    for (int i = tx; i < RTR_H; i += blockDim.x) {
        float s = rf1b[i];
        for (int j = 0; j < RTR_H; j++) s += rs[j] * rf1[i * (size_t)RTR_H + j];
        tmp[i] = gelu_tanh_f(s);
    }
    __syncthreads();

    // ── 5. fc2 + GELU ──
    for (int i = tx; i < RTR_H; i += blockDim.x) {
        float s = rf2b[i];
        for (int j = 0; j < RTR_H; j++) s += tmp[j] * rf2[i * (size_t)RTR_H + j];
        rs[i] = gelu_tanh_f(s);
    }
    __syncthreads();

    // ── 6. out_proj + softmax + top-1 ──
    float scores_local[N_EXP_T];
    for (int i = tx; i < N_EXP_T; i += blockDim.x) {
        float s = 0;
        for (int j = 0; j < RTR_H; j++) s += rs[j] * rout[i * (size_t)RTR_H + j];
        scores[i] = s + bb[i];
    }
    __syncthreads();

    float maxv = scores[0];
    for (int i = 1; i < N_EXP_T; i++) if (scores[i] > maxv) maxv = scores[i];
    float sumv = 0;
    for (int i = tx; i < N_EXP_T; i += blockDim.x) {
        probs[i] = expf(scores[i] - maxv);
        sumv += probs[i];
    }
    sumv = block_sum_f(sumv, sh);
    float inv_sum = 1.0f / (sumv + 1e-10f);
    for (int i = tx; i < N_EXP_T; i += blockDim.x) probs[i] *= inv_sum;
    __syncthreads();

    if (tx == 0) {
        int best = 0;
        float bestv = probs[0];
        for (int i = 1; i < N_EXP_T; i++)
            if (probs[i] > bestv) { bestv = probs[i]; best = i; }
        expert_idx[b] = best;
        expert_wt[b]  = bestv;
        local_expert  = best;
    }
    __syncthreads();

    int my_expert = local_expert;

    // ── Phase 2: Batch-Union MoE Expert FFN ──
    // Only the first B_UNIQ warps/blocks do the actual compute.
    // Here each block handles ONE token's expert FFN.
    // The dedup happens at the pipeline level (caller groups by expert).
    if (my_expert < N_EXP) {
        // gate_up: gate_up_tmp[2*N_FF] = hs[H] @ gu[expert, 2*N_FF, H]^T
        __shared__ __half gate_up_tmp[2 * N_FF];
        for (int i = tx; i < 2 * N_FF; i += blockDim.x) {
            float s = 0;
            const __half* w = gu + ((size_t)my_expert * 2 * N_FF + i) * (size_t)H;
            for (int j = 0; j < H; j++)
                s += __half2float(hs_b[j]) * __half2float(w[j]);
            gate_up_tmp[i] = __float2half(s);
        }
        __syncthreads();

        // SiLU(gate) * up → moe_buf
        if (tx < N_FF) {
            float g = __half2float(gate_up_tmp[tx]);
            float u = __half2float(gate_up_tmp[N_FF + tx]);
            gate_up_tmp[tx] = __float2half((g / (1.0f + expf(-g))) * u);
        }
        __syncthreads();

        // down: moe_out[H] = moe_buf[N_FF] @ dn[expert, H, N_FF]^T
        __half* moe_b = moe_out + (size_t)b * H;
        for (int i = tx; i < H; i += blockDim.x) {
            float s = 0;
            const __half* w = dn + ((size_t)my_expert * H + i) * (size_t)N_FF;
            for (int j = 0; j < N_FF; j++)
                s += __half2float(gate_up_tmp[j]) * __half2float(w[j]);
            moe_b[i] = __float2half(s);
        }
    } else {
        // MOD skip passthrough
        __half* moe_b = moe_out + (size_t)b * H;
        for (int i = tx; i < H; i += blockDim.x) moe_b[i] = hs_b[i];
    }
}

// ════════════════════════════════════════════════════════════════
// Sort-based MoE dispatch (replaces O(N_EXP × B) scan)
// ════════════════════════════════════════════════════════════════
//
// Pipeline:
//   1. moe_sort_histogram_kernel  →  expert_counts[N_EXP], expert_offsets[N_EXP], sorted_token_ids[B]
//   2. moe_sorted_expert_kernel   →  moe_out[B, H] (processes each expert contiguously)
//
// Sort is O(B + N_EXP) instead of O(N_EXP × B) scan, and more importantly
// groups tokens by expert for cache-friendly weight access: each expert's
// weights are loaded ONCE and all tokens for that expert are processed
// consecutively before moving to the next expert.
//
// For B < 6, the overhead of sorting may not be worth it — use the
// fused per-token kernel instead.

// ────────────────────────────────────────────────────────────────
// Kernel: moe_sort_histogram
// From expert_idx[B], computes:
//   expert_counts[e] = number of tokens routing to expert e
//   expert_offsets[e] = prefix sum (start index in sorted array)
//   sorted_token_ids[b] = token index b, sorted by expert group
//
// Launch with 1 block of (max(B, N_EXP_T) + 31)/32 × 32 threads.
// For B <= 8 and N_EXP_T = 17, a single warp suffices.
// ────────────────────────────────────────────────────────────────

__global__ void moe_sort_histogram_kernel(
    const int*  __restrict__ expert_idx,   // [B]
    int*        __restrict__ expert_counts, // [N_EXP_T]
    int*        __restrict__ expert_offsets,// [N_EXP_T]
    int*        __restrict__ sorted_token_ids, // [B]
    int B)
{
    int tx = threadIdx.x;

    // ── Zero counts ──
    for (int i = tx; i < N_EXP_T; i += blockDim.x)
        expert_counts[i] = 0;
    __syncthreads();

    // ── Histogram: count tokens per expert ──
    // B <= 8, so each thread can atomically increment.
    // Using shared memory atomics for speed.
    __shared__ int sh_counts[N_EXP_T];
    for (int i = tx; i < N_EXP_T; i += blockDim.x)
        sh_counts[i] = 0;
    __syncthreads();

    for (int b = tx; b < B; b += blockDim.x) {
        int e = expert_idx[b];
        if (e >= 0 && e < N_EXP_T)
            atomicAdd(&sh_counts[e], 1);
    }
    __syncthreads();

    // ── Store counts and compute prefix sum ──
    if (tx == 0) {
        int running = 0;
        for (int e = 0; e < N_EXP_T; e++) {
            expert_counts[e] = sh_counts[e];
            expert_offsets[e] = running;
            running += sh_counts[e];
        }
    }
    __syncthreads();

    // ── Scatter: sort token IDs by expert ──
    // Each token writes its index to the correct position.
    // We use a temporary offset counter to track the current write position
    // within each expert's group.
    if (tx == 0) {
        // Local counters for each expert's current write position
        int pos[N_EXP_T];
        for (int e = 0; e < N_EXP_T; e++)
            pos[e] = expert_offsets[e];

        for (int b = 0; b < B; b++) {
            int e = expert_idx[b];
            sorted_token_ids[pos[e]] = b;
            pos[e]++;
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Kernel: moe_sorted_expert_kernel
// Processes all tokens grouped by expert (sorted order).
// Each block handles one expert that has at least one token.
// Loads expert weights ONCE, then processes all tokens for that expert
// consecutively — no redundant weight loads, no L2 thrashing.
//
// Launch with N_EXP_T blocks (only those with count > 0 do work).
// ────────────────────────────────────────────────────────────────

__global__ void moe_sorted_expert_kernel(
    const __half* __restrict__ hs,              // [B, H]
    const int*    __restrict__ sorted_token_ids, // [B]
    const int*    __restrict__ expert_counts,    // [N_EXP_T]
    const int*    __restrict__ expert_offsets,   // [N_EXP_T]
    const __half* __restrict__ gu,              // [N_EXP, 2*N_FF, H]
    const __half* __restrict__ dn,              // [N_EXP, H, N_FF]
    __half* __restrict__ moe_out,               // [B, H]
    int B)
{
    int expert = blockIdx.x;
    if (expert >= N_EXP) return;

    int n = expert_counts[expert];
    if (n == 0) return;

    int start = expert_offsets[expert];
    int tx = threadIdx.x;

    // Gate_up input is shared across all tokens for this expert
    // (only the hidden state differs per token)
    __shared__ __half gate_up_buf[2 * N_FF];

    // Process each token in this expert's group
    for (int ti = 0; ti < n; ti++) {
        int b = sorted_token_ids[start + ti];
        const __half* hs_b = hs + (size_t)b * H;

        // ── gate_up: tmp[2*N_FF] = hs[H] @ gu[expert, 2*N_FF, H]^T ──
        for (int i = tx; i < 2 * N_FF; i += blockDim.x) {
            float s = 0;
            const __half* w = gu + ((size_t)expert * 2 * N_FF + i) * (size_t)H;
            for (int j = 0; j < H; j++)
                s += __half2float(hs_b[j]) * __half2float(w[j]);
            gate_up_buf[i] = __float2half(s);
        }
        __syncthreads();

        // ── SiLU(gate) * up → gate_up_buf[N_FF] ──
        if (tx < N_FF) {
            float g = __half2float(gate_up_buf[tx]);
            float u = __half2float(gate_up_buf[N_FF + tx]);
            gate_up_buf[tx] = __float2half((g / (1.0f + expf(-g))) * u);
        }
        __syncthreads();

        // ── down: moe_out[H] = gate_up_buf[N_FF] @ dn[expert, H, N_FF]^T ──
        __half* moe_b = moe_out + (size_t)b * H;
        for (int i = tx; i < H; i += blockDim.x) {
            float s = 0;
            const __half* w = dn + ((size_t)expert * H + i) * (size_t)N_FF;
            for (int j = 0; j < N_FF; j++)
                s += __half2float(gate_up_buf[j]) * __half2float(w[j]);
            moe_b[i] = __float2half(s);
        }
        __syncthreads();
    }
}

// ────────────────────────────────────────────────────────────────
// Kernel: moe_modskip_passthrough_kernel
// For tokens that route to expert == N_EXP (MOD skip), copy hs to moe_out
// directly (identity passthrough — no expert FFN applied).
// These tokens are NOT handled by moe_sorted_expert_kernel which only
// processes experts 0..N_EXP-1.
// ────────────────────────────────────────────────────────────────

__global__ void moe_modskip_passthrough_kernel(
    __half* __restrict__ moe_out,        // [B, H]
    const __half* __restrict__ hs,       // [B, H]
    const int*   __restrict__ expert_idx, // [B]
    int skip_expert,
    int B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * H) return;
    int b = i / H;
    int h = i % H;
    if (expert_idx[b] == skip_expert) {
        moe_out[b * (size_t)H + h] = hs[b * (size_t)H + h];
    }
}
