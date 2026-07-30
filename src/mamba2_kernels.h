// mamba2_kernels.h — Mamba2 selective scan + conv1d kernels
// Ported from state-spaces/mamba (BSD-3-Clause) and llama.cpp (MIT).
// Implements the Mamba2 structured state space dual (SSD) layer.
//
// Mamba2 architecture:
//   y = selective_scan(conv1d(silu(in_proj(x))), A, B, C, dt)
//
// Key difference from Mamba1: grouped heads (n_group), head_dim, dt_rank = n_head
#pragma once

#include <cstdint>
#include <vector>
#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>

// ── Mamba2 configuration (matches GGUF KV metadata) ──
struct Mamba2Config {
    int64_t d_model;      // hidden size (n_embd)
    int64_t d_state;      // SSM state size (typically 64 for Zamba2)
    int64_t d_conv;       // conv1d kernel size (typically 4)
    int64_t d_inner;      // inner dimension (expansion_factor * d_model)
    int64_t n_head;       // number of heads (= dt_rank in Mamba2)
    int64_t n_group;      // number of groups for grouped heads
    int64_t head_dim;     // head dimension = d_inner / n_head
    float rms_norm_eps = 1e-5f;
};

// ── CPU reference implementation of Mamba2 layer forward pass ──
// All dimensions:
//   x:    [d_model]             — input token embedding
//   ssm:  [d_state, d_inner]    — SSM hidden state (flattened)
//   conv: [d_conv-1, d_inner + 2*n_group*d_state] — conv state (flattened)
//
// Weights (per layer):
//   in_proj:   [d_in_proj, d_model]    where d_in_proj = d_inner + (d_inner + 2*n_group*d_state) + n_head
//   conv1d:    [d_conv, d_inner + 2*n_group*d_state]
//   conv1d_b:  [d_inner + 2*n_group*d_state]
//   dt_bias:   [n_head]
//   A_log:     [n_head]    (stored as log for stability)
//   D:         [n_head]    (skip connection)
//   norm_w:    [d_inner / n_group, n_group]  (group norm weight)
//   out_proj:  [d_model, d_inner]
//
// Output:
//   y:     [d_model]
//   ssm:   [d_state, d_inner]  (updated)
//   conv:  [d_conv-1, d_inner + 2*n_group*d_state]  (updated)

void mamba2_cpu_forward(
    // Inputs
    const float* x,                // [d_model]
    const float* in_proj_w,        // [d_in_proj, d_model]
    const float* conv1d_w,         // [d_conv, d_conv_dim]
    const float* conv1d_b,         // [d_conv_dim]
    const float* dt_bias,          // [n_head]
    const float* A_log,            // [n_head]
    const float* D,                // [n_head]
    const float* norm_w,           // [d_inner / n_group, n_group]
    const float* out_proj_w,       // [d_model, d_inner]
    // State (in-place updated)
    float* conv_state,             // [d_conv-1, d_conv_dim]
    float* ssm_state,              // [d_state, d_inner]
    // Output
    float* y,                      // [d_model]
    // Config
    const Mamba2Config& cfg
);

// ── HIP kernel declarations (implemented in mamba2_kernels.hip) ──
#ifdef __HIPCC__
#include <hip/hip_runtime.h>

// ════════════════════════════════════════════════════════════════════════
// TUNABLE KERNELS (recommended for all new code)
// ════════════════════════════════════════════════════════════════════════

// Tiled FP32 GEMV with LDS-cached activations. 4 rows per block, 256 threads.
// 2-5× faster than the naive gemv_kernel on Zamba2-2.7B shapes.
__global__ void mamba2_tiled_gemv_kernel(
    const float* __restrict__ W,    // [M, K] row-major
    const float* __restrict__ x,    // [K]
    float* __restrict__ y,          // [M]
    int M, int K);

// Tuned Mamba2 selective scan. Process ALL head_dim elements per head
// with d_state parallelism. Fixes the old kernel bug that only read
// the 1st x element. Uses __shfl and LDS for reduction.
__global__ void mamba2_selective_scan_tuned_kernel(
    const float* __restrict__ x,     // [B, L, d_inner]
    const float* __restrict__ dt,    // [B, L, n_head]
    const float* __restrict__ A_log, // [n_head]
    const float* __restrict__ B,     // [B, L, n_group, d_state]
    const float* __restrict__ C,     // [B, L, n_group, d_state]
    const float* __restrict__ D,     // [n_head]
    float* __restrict__ y,           // [B, L, d_inner]
    float* __restrict__ final_state, // [B, d_state, d_inner]
    int B_dim, int L, int d_inner, int d_state,
    int n_head, int n_group, int head_dim);

// Fused scan: combines dt_bias add + softplus + selective scan in one kernel.
// dt_bias may be nullptr (assumes dt already has softplus applied).
__global__ void mamba2_scan_fused_kernel(
    const float* __restrict__ x,     // [B, L, d_inner]
    const float* __restrict__ dt,    // [B, L, n_head]
    const float* __restrict__ dt_bias, // [n_head] or nullptr
    const float* __restrict__ A_log, // [n_head]
    const float* __restrict__ B,     // [B, L, n_group, d_state]
    const float* __restrict__ C,     // [B, L, n_group, d_state]
    const float* __restrict__ D,     // [n_head]
    float* __restrict__ y,           // [B, L, d_inner]
    float* __restrict__ final_state, // [B, d_state, d_inner]
    int B_dim, int L, int d_inner, int d_state,
    int n_head, int n_group, int head_dim);

// Tuned conv1d with LDS-cached weights. Tiled across conv_dim.
__global__ void mamba2_conv1d_tuned_kernel(
    const float* __restrict__ x,     // [B, L, conv_dim]
    const float* __restrict__ w,     // [d_conv, conv_dim]
    const float* __restrict__ b,     // [conv_dim]
    float* __restrict__ y,           // [B, L, conv_dim]
    float* __restrict__ state,       // [B, d_conv-1, conv_dim]
    int B_dim, int L, int d_conv, int conv_dim);

// Group RMS norm for Mamba2 group normalization.
__global__ void mamba2_group_norm_kernel(
    float* __restrict__ y,           // [B, d_inner] in-place
    const float* __restrict__ norm_w, // [d_inner/n_group, n_group]
    int d_inner, int n_group, int batch, float eps);

// Fused: y[i] *= silu(z[i]) — combines gate apply in one kernel.
__global__ void silu_mul_kernel(float* y, float* z, int n);

// ════════════════════════════════════════════════════════════════════════
// LEGACY KERNELS (compatibility stubs — delegate to tuned variants)
// ════════════════════════════════════════════════════════════════════════

// Legacy selective scan (kept for API compat with zamba2_engine_hip.hip).
// Now has the CORRECT per-head_dim-element scan loop.
__global__ void mamba2_selective_scan_kernel(
    const float* __restrict__ x,     // [B, L, d_inner]
    const float* __restrict__ dt,    // [B, L, n_head]
    const float* __restrict__ A,     // [n_head]
    const float* __restrict__ B,     // [B, L, n_group, d_state]
    const float* __restrict__ C,     // [B, L, n_group, d_state]
    const float* __restrict__ D,     // [n_head]
    float* __restrict__ y,           // [B, L, d_inner]
    float* __restrict__ final_state, // [B, d_state, d_inner]
    int B_dim, int L, int d_inner, int d_state,
    int n_head, int n_group, int head_dim);

// Legacy conv1d (kept for compat).
__global__ void mamba2_conv1d_kernel(
    const float* __restrict__ x,     // [B, L, conv_dim]
    const float* __restrict__ w,     // [d_conv, conv_dim]
    const float* __restrict__ b,     // [conv_dim]
    float* __restrict__ y,           // [B, L, conv_dim]
    float* __restrict__ state,       // [B, d_conv-1, conv_dim]
    int B_dim, int L, int d_conv, int conv_dim);

// Legacy simple GEMV (1 thread per row, no LDS).
__global__ void gemv_kernel(const float*, const float*, float*, int, int);

// Legacy element-wise kernels.
__global__ void silu_kernel(float*, int);
__global__ void add_dt_bias_kernel(float*, const float*, int);
__global__ void mul_kernel(float*, const float*, int);

// ════════════════════════════════════════════════════════════════════════
// HOST-SIDE LAUNCHERS
// ════════════════════════════════════════════════════════════════════════

// Tuned decode block: in_proj → silu(conv1d) → selective_scan → silu(z)*y → out_proj
// Uses tiled GEMV, LDS-cached conv1d, fused scan + dt_bias.
// All pointers must be device pointers.
void mamba2_gpu_decode_block_tuned(
    const float* x_in,           // [d_model]
    const float* in_proj_w,      // [d_in_proj, d_model]
    const float* conv1d_w,       // [d_conv, conv_dim]
    const float* conv1d_b,       // [conv_dim]
    const float* dt_bias,        // [n_head]
    const float* A_log,          // [n_head]
    const float* D,              // [n_head]
    const float* norm_w,         // [d_inner/n_group, n_group] or nullptr
    const float* out_proj_w,     // [d_model, d_inner]
    float* conv_state,           // [d_conv-1, conv_dim] updated
    float* ssm_state,            // [d_state, d_inner] updated
    float* y_out,                // [d_model]
    float* tmp,                  // [max(d_in_proj, conv_dim, d_inner)] workspace
    int d_model, int d_inner, int d_state, int d_conv,
    int n_head, int n_group, int head_dim, int conv_dim,
    hipStream_t stream);

// Legacy decode block (now forwards to mamba2_gpu_decode_block_tuned).
void mamba2_gpu_decode_block(
    const float* x_in,
    const float* in_proj_w,
    const float* conv1d_w, const float* conv1d_b,
    const float* dt_bias, const float* A_log, const float* D,
    const float* norm_w,
    const float* out_proj_w,
    float* conv_state, float* ssm_state, float* y_out,
    float* tmp,
    int d_model, int d_inner, int d_state, int d_conv,
    int n_head, int n_group, int head_dim, int conv_dim,
    hipStream_t stream);

#endif // __HIPCC__
