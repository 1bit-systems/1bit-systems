// Standalone CCA attention kernel — Zaya's Cross-Computer Attention
//
// Computes: Q, K proj → conv_qk (depthwise+grouped) → qk_means → L2 norm → RoPE → GQA
//
// Build:
//   cd /home/bcloud/1bit/build && cmake --build . --target test_cca_attn -j8
//
// Reference data from PyTorch: /tmp/zaya_ref.npz

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cassert>
#include <random>
#include <cstring>

// ── CCA attention kernel ─────────────────────────────────────────────────────
// Computes the full CCA attention forward pass for one token:
//   1. RMSNorm
//   2. Q, K projections
//   3. val_proj1(current_hs), val_proj2(delayed_hs)
//   4. conv_qk depthwise + grouped
//   5. qk_means + add
//   6. L2 normalize Q, K (with per-head temperature for K)
//   7. RoPE (partial: first n_rot dims)
//   8. GQA attention Q @ K^T → softmax → @ V → output proj
//
// Grid: 1 block per token
// Block: 128 threads, 4 waves

#define BLOCK_SIZE 128
#define WARP_SIZE 32

// Config (matches ZAYA1-8B)
#define N_HEADS 8
#define N_KV_HEADS 2
#define HEAD_DIM 128
#define HIDDEN 2048
#define N_ROT 64       // partial_rotary_factor * head_dim = 0.5 * 128
#define D_CONV 2       // cca_time0 = cca_time1 = 2
#define N_GROUPS (N_HEADS + N_KV_HEADS)  // 10
#define GQA (N_HEADS / N_KV_HEADS)       // 4
#define FP16_MAX 65504.0f

// Q, K, V dimensions
#define Q_DIM (N_HEADS * HEAD_DIM)     // 1024
#define K_DIM (N_KV_HEADS * HEAD_DIM)  // 256
#define QKV_DIM (Q_DIM + K_DIM)        // 1280

// ── RoPE (partial: applies to first N_ROT dims) ─────────────────────────────
__device__ inline void apply_rope_half(__half* q, __half* k, int pos, int stride_q, int stride_k) {
    for (int h = 0; h < N_HEADS; h++) {
        for (int d = 0; d < N_ROT; d += 2) {
            float theta = powf(500000.0f, -2.0f * d / HEAD_DIM);
            float cos_val = cosf(pos * theta);
            float sin_val = sinf(pos * theta);
            
            int idx0 = h * stride_q + d;
            int idx1 = h * stride_q + d + 1;
            float q0 = __half2float(q[idx0]);
            float q1 = __half2float(q[idx1]);
            q[idx0] = __float2half(q0 * cos_val - q1 * sin_val);
            q[idx1] = __float2half(q0 * sin_val + q1 * cos_val);
        }
    }
    for (int h = 0; h < N_KV_HEADS; h++) {
        for (int d = 0; d < N_ROT; d += 2) {
            float theta = powf(500000.0f, -2.0f * d / HEAD_DIM);
            float cos_val = cosf(pos * theta);
            float sin_val = sinf(pos * theta);
            
            int idx0 = h * stride_k + d;
            int idx1 = h * stride_k + d + 1;
            float k0 = __half2float(k[idx0]);
            float k1 = __half2float(k[idx1]);
            k[idx0] = __float2half(k0 * cos_val - k1 * sin_val);
            k[idx1] = __float2half(k0 * sin_val + k1 * cos_val);
        }
    }
}

// ── CCA attention kernel ────────────────────────────────────────────────────
// Inputs:
//   hidden_states [HIDDEN] — RMSNormed input to CCA
//   prev_hs [HIDDEN] — delayed hidden state from previous decode step (0 for first)
//   conv_state [2 * QKV_DIM] — conv state from previous step (0 for first)
//   position — RoPE position (0 for first token)
//
// Weights (all pre-loaded):
//   wq [Q_DIM * HIDDEN], wk [K_DIM * HIDDEN]
//   val_proj1 [(K_DIM/2) * HIDDEN], val_proj2 [(K_DIM/2) * HIDDEN]
//   o_proj [HIDDEN * Q_DIM]
//   conv_dw [QKV_DIM * D_CONV], conv_dw_b [QKV_DIM]
//   conv_grp [QKV_DIM * (QKV_DIM/N_GROUPS) * D_CONV], conv_grp_b [QKV_DIM]
//   k_scale [N_KV_HEADS]
//   norm_weight [HIDDEN] — RMSNorm weight
//
// Outputs:
//   attn_output [HIDDEN] — projected attention output
//   next_conv_state [2 * QKV_DIM] — conv state for next decode
//   next_prev_hs [HIDDEN] — hidden state for next decode's delayed stream

__launch_bounds__(128) __global__ void cca_attn_kernel(
    // Inputs
    const __half* __restrict__ hidden_states,  // [HIDDEN]
    const __half* __restrict__ prev_hs,         // [HIDDEN]
    const __half* __restrict__ conv_state_in,   // [2 * QKV_DIM]
    int position,
    // Weights
    const __half* __restrict__ wq,              // [Q_DIM * HIDDEN]
    const __half* __restrict__ wk,              // [K_DIM * HIDDEN]
    const __half* __restrict__ wv1,             // [(K_DIM/2) * HIDDEN]
    const __half* __restrict__ wv2,             // [(K_DIM/2) * HIDDEN]
    const __half* __restrict__ wo,              // [HIDDEN * Q_DIM]
    const float*  __restrict__ conv_dw,         // [QKV_DIM * D_CONV]
    const float*  __restrict__ conv_dw_b,       // [QKV_DIM]
    const float*  __restrict__ conv_grp,        // [QKV_DIM * (QKV_DIM/N_GROUPS) * D_CONV]
    const float*  __restrict__ conv_grp_b,      // [QKV_DIM]
    const float*  __restrict__ k_scale,         // [N_KV_HEADS]
    const __half* __restrict__ norm_weight,     // [HIDDEN]
    // Outputs
    __half* __restrict__ attn_output,           // [HIDDEN]
    __half* __restrict__ next_conv_state,       // [2 * QKV_DIM]
    __half* __restrict__ next_prev_hs)          // [HIDDEN]
{
    // ── 1. RMSNorm ──
    // First, compute the norm in shared memory
    __shared__ float s_norm_sum;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x) {
        float v = __half2float(hidden_states[i]);
        sum_sq += v * v;
    }
    
    // Reduction to get sum of squares
    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor(sum_sq, offset);
    
    // Block-level reduction via shared memory
    if (threadIdx.x % WARP_SIZE == 0) {
        // Each warp's first thread writes to shared memory
        // Not needed for now - let's use a simpler approach
    }
    
    // WARNING: This kernel is an incomplete stub (#403). It does NOT produce
    // valid output. The server uses cca_custom_kernel from zaya_cca_custom.hip
    // instead. Trap if accidentally launched to avoid silent garbage output.
    __builtin_trap();
}

// GQA is not referenced elsewhere in this file — undef so later .hip files
// #include'd into the same translation unit (zaya_engine.cpp) that also
// define GQA don't trigger a silent macro-redefinition (issue #285).
#undef GQA
