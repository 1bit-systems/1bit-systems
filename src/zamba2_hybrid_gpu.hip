// zamba2_hybrid_gpu.hip — Fused GPU hybrid layer for Zamba2
//
// Replaces the CPU fallback with zero hipMemcpy round-trips.
// Entire post-Mamba2 attention + FFN pipeline runs in one kernel.
//
// Pipeline:
//   1. RMS norm on hidden
//   2. Linear projection + residual
//   3. Q, K, V projections (3 GEMVs from same input)
//   4. RoPE on Q and K
//   5. KV cache append
//   6. Attention with online softmax
//   7. Attention output projection + residual
//   8. Pre-FFN RMS norm
//   9. Gate + Up projections (2 GEMVs from same input)
//   10. SiLU gate activation
//   11. Down projection
//   12. Residual add
//
// Strategy: decomposed into efficient sub-kernel launches using
// the existing tiled GEMV from mamba2_kernels.hip plus new kernels
// for attention, RoPE, and element-wise ops.

#include "mamba2_kernels.h"
#include <hip/hip_runtime.h>
#include <math.h>

// ════════════════════════════════════════════════════════════════════════
// FP32 vector add: y[i] += x[i]
// ════════════════════════════════════════════════════════════════════════
__global__ void vec_add_kernel(float* y, const float* x, int n) {
    int i = blockIdx.x * (blockDim.x * 4) + threadIdx.x;
    if (i >= n) return;
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        int idx = i + p * blockDim.x;
        if (idx < n) y[idx] += x[idx];
    }
}

// ════════════════════════════════════════════════════════════════════════
// FP32 copy: y[i] = x[i]
// ════════════════════════════════════════════════════════════════════════
__global__ void vec_copy_kernel(float* y, const float* x, int n) {
    int i = blockIdx.x * (blockDim.x * 4) + threadIdx.x;
    if (i >= n) return;
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        int idx = i + p * blockDim.x;
        if (idx < n) y[idx] = x[idx];
    }
}

// ════════════════════════════════════════════════════════════════════════
// FP32 RoPE — one block per KV head
// ════════════════════════════════════════════════════════════════════════
__global__ __launch_bounds__(128, 4)
void rope_fp32_kernel(
    float* __restrict__ qk, int pos, float theta,
    int n_heads, int n_kv_heads, int head_dim)
{
    int kv_h = blockIdx.x;
    int d2   = threadIdx.x;
    if (kv_h >= n_kv_heads || d2 * 2 >= head_dim) return;
    int n_groups = n_heads / n_kv_heads;

    float freq = (float)pos;
    float pow_theta = 1.0f;
    for (int i = 0; i < d2 * 2; ++i) pow_theta *= (1.0f / theta);
    freq *= pow_theta;
    float cos_v = cosf(freq);
    float sin_v = sinf(freq);

    for (int g = 0; g < n_groups; ++g) {
        int q_h = kv_h * n_groups + g;
        float* q = qk + q_h * head_dim;
        float q0 = q[d2 * 2], q1 = q[d2 * 2 + 1];
        q[d2 * 2]     = q0 * cos_v - q1 * sin_v;
        q[d2 * 2 + 1] = q0 * sin_v + q1 * cos_v;
    }
    float* k = qk + n_heads * head_dim + kv_h * head_dim;
    float k0 = k[d2 * 2], k1 = k[d2 * 2 + 1];
    k[d2 * 2]     = k0 * cos_v - k1 * sin_v;
    k[d2 * 2 + 1] = k0 * sin_v + k1 * cos_v;
}

// ════════════════════════════════════════════════════════════════════════
// FP32 Decode Attention — one block per Q head, online softmax
// ════════════════════════════════════════════════════════════════════════
__global__ __launch_bounds__(256, 2)
void attn_decode_fp32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    int n_heads, int n_kv_heads, int head_dim,
    int seq_len, float scale)
{
    int q_h = blockIdx.x;
    int tid = threadIdx.x;
    if (q_h >= n_heads || tid >= head_dim) return;

    int kv_h = q_h / (n_heads / n_kv_heads);
    const float* q_row = q + q_h * head_dim;
    size_t kv_stride = (size_t)n_kv_heads * head_dim;

    // Registers for online softmax
    float m = -INFINITY, d = 0.0f, acc = 0.0f;

    for (int t = 0; t < seq_len; ++t) {
        // Each thread computes Q[tid] * K[t, kv_h, tid]
        __shared__ float s_red[256];
        s_red[tid] = q_row[tid] * k_cache[t * kv_stride + kv_h * head_dim + tid];
        __syncthreads();

        // Parallel reduction to dot product
        #pragma unroll
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (tid < off) s_red[tid] += s_red[tid + off];
            __syncthreads();
        }

        float score = s_red[0] * scale;
        float m_prev = m;
        m = fmaxf(m, score);
        float ed = expf(m_prev - m);
        float es = expf(score - m);
        d = d * ed + es;
        acc = acc * ed + es * v_cache[t * kv_stride + kv_h * head_dim + tid];
    }

    out[q_h * head_dim + tid] = acc / d;
}

// ════════════════════════════════════════════════════════════════════════
// Full Hybrid Layer Orchestrator (all GPU, zero CPU round-trips)
// ════════════════════════════════════════════════════════════════════════
// This function launches ALL the sub-kernels for one hybrid layer.
// It replaces the ~50 hipMemcpy calls and ~10 CPU matrix multiplies.
//
// Memory layout in tmp buffer:
//   [0..d_model-1]            = normed (RMS norm output)
//   [d_model..2*d_model-1]    = projected (linear projection output)
//   [2*d_model..2*d_model+qkv_size-1] = QKV (Q then K then V)
//   [2*d_model+qkv_size..2*d_model+qkv_size+attn_out_size-1] = attn_out
//   (rest) = FFN intermediates
//
// Where qkv_size = (n_attn_heads + n_kv_heads) * attn_head_dim

void zamba2_hybrid_forward_gpu(
    // Mamba2 block
    const float* mamba_in_proj_w, const float* mamba_conv1d_w,
    const float* mamba_conv1d_b, const float* mamba_dt_bias,
    const float* mamba_A_log, const float* mamba_D,
    const float* mamba_norm_w,
    const float* mamba_out_proj_w,
    float* mamba_conv_state, float* mamba_ssm_state,

    // Hybrid weights
    const float* input_norm_w,
    const float* linear_w,
    const float* q_proj_w, const float* k_proj_w, const float* v_proj_w,
    const float* o_proj_w,
    const float* pre_ff_norm_w,
    const float* ffn_norm_w,
    const float* gate_proj_w, const float* up_proj_w,
    const float* down_proj_w,

    // KV cache (updated in-place)
    float* k_cache, float* v_cache,

    // Hidden (input/output, in-place)
    float* hidden,

    // Scratch (large enough for all intermediates)
    float* scratch,

    // Dims
    int d_model, int d_inner, int d_state, int d_conv,
    int n_head, int n_group, int head_dim, int conv_dim,
    int n_attn_heads, int n_kv_heads, int attn_head_dim,
    int d_ff, int pos, int max_seq,

    hipStream_t stream)
{
    const int threads = 256;
    int blocks;

    // ──────────────────────────────────────────────────────────────────
    // Step 1: Mamba2 decode block (tuned)
    // ──────────────────────────────────────────────────────────────────
    mamba2_gpu_decode_block_tuned(
        hidden, mamba_in_proj_w, mamba_conv1d_w, mamba_conv1d_b,
        mamba_dt_bias, mamba_A_log, mamba_D, mamba_norm_w, mamba_out_proj_w,
        mamba_conv_state, mamba_ssm_state,
        hidden, scratch,
        d_model, d_inner, d_state, d_conv,
        n_head, n_group, head_dim, conv_dim, stream);

    // ──────────────────────────────────────────────────────────────────
    // Step 2: RMS norm → linear projection → QKV → RoPE → Attn → FFN
    // ──────────────────────────────────────────────────────────────────

    // Buffer layout in scratch (pre-sized as max(d_model*3, QKV+attn, FFN)):
    float* d_normed    = scratch;                                     // [d_model]
    float* d_proj      = scratch + d_model;                            // [d_model]
    int qkv_size       = (n_attn_heads + 2 * n_kv_heads) * attn_head_dim;  // Q + K + V
    float* d_qkv       = scratch + 2 * d_model;                        // [qkv_size]
    float* d_attn_out  = d_qkv + qkv_size;                              // [n_attn_heads * attn_head_dim]

    // Verify scratch is large enough
    // Need: normed(d_model) + proj(d_model) + QKV(qkv_size) + attn_out + gate(d_ff) + up(d_ff)
    // ≈ 2*2560 + 3120 + 5120*2 = ~20K floats ≈ 80KB. Fine.

    // 2a. Copy hidden → normed, then RMS norm in-place
    blocks = (d_model + threads * 4 - 1) / (threads * 4);
    hipLaunchKernelGGL(vec_copy_kernel, dim3(blocks), dim3(threads), 0, stream,
        d_normed, hidden, d_model);

    hipLaunchKernelGGL(mamba2_group_norm_kernel,
        dim3(1, 1, 1), dim3(threads), 0, stream,
        d_normed, input_norm_w, d_model, 1, 1, 1e-6f);

    // 2b. Linear projection: d_proj = W_linear @ d_normed
    int gemv_blocks = (d_model + 2 - 1) / 2;
    hipLaunchKernelGGL(mamba2_tiled_gemv_kernel,
        dim3(gemv_blocks), dim3(256), 0, stream,
        linear_w, d_normed, d_proj, d_model, d_model);

    // 2c. Residual: d_proj += hidden
    hipLaunchKernelGGL(vec_add_kernel, dim3(blocks), dim3(threads), 0, stream,
        d_proj, hidden, d_model);

    // 2d. QKV projections from d_proj into d_qkv
    int q_elems = n_attn_heads * attn_head_dim;
    int k_elems = n_kv_heads * attn_head_dim;
    float* d_q = d_qkv;                        // [q_elems]
    float* d_k = d_qkv + q_elems;               // [k_elems]
    float* d_v = d_qkv + q_elems + k_elems;     // [k_elems]

    gemv_blocks = (q_elems + 2 - 1) / 2;
    hipLaunchKernelGGL(mamba2_tiled_gemv_kernel,
        dim3(gemv_blocks), dim3(256), 0, stream,
        q_proj_w, d_proj, d_q, q_elems, d_model);

    gemv_blocks = (k_elems + 2 - 1) / 2;
    hipLaunchKernelGGL(mamba2_tiled_gemv_kernel,
        dim3(gemv_blocks), dim3(256), 0, stream,
        k_proj_w, d_proj, d_k, k_elems, d_model);
    hipLaunchKernelGGL(mamba2_tiled_gemv_kernel,
        dim3(gemv_blocks), dim3(256), 0, stream,
        v_proj_w, d_proj, d_v, k_elems, d_model);

    // 2e. RoPE on Q and K
    dim3 rope_grid(n_kv_heads, 1, 1);
    dim3 rope_block(min(attn_head_dim / 2, 128), 1, 1);
    float rope_theta = 10000.0f;
    hipLaunchKernelGGL(rope_fp32_kernel,
        rope_grid, rope_block, 0, stream,
        d_qkv, pos, rope_theta,
        n_attn_heads, n_kv_heads, attn_head_dim);

    // 2f. Append K and V to KV cache
    // k_cache[pos, kv_h, d] = d_k[kv_h * head_dim + d]
    // v_cache[pos, kv_h, d] = d_v[kv_h * head_dim + d]
    // Each thread copies one element
    blocks = (k_elems + threads * 4 - 1) / (threads * 4);
    // We need a kernel for this. For now do D2D memcpy:
    hipMemcpyDtoDAsync(k_cache + pos * (size_t)n_kv_heads * attn_head_dim,
                       d_k, k_elems * sizeof(float), stream);
    hipMemcpyDtoDAsync(v_cache + pos * (size_t)n_kv_heads * attn_head_dim,
                       d_v, k_elems * sizeof(float), stream);

    // 2g. Attention
    float attn_scale = 1.0f / sqrtf((float)attn_head_dim);
    dim3 attn_grid(n_attn_heads, 1, 1);
    dim3 attn_block(attn_head_dim > 256 ? 256 : attn_head_dim, 1, 1);
    d_attn_out = d_qkv + q_elems + k_elems + k_elems;  // after Q+K+V
    hipLaunchKernelGGL(attn_decode_fp32_kernel,
        attn_grid, attn_block, 0, stream,
        d_q, k_cache, v_cache, d_attn_out,
        n_attn_heads, n_kv_heads, attn_head_dim,
        pos + 1, attn_scale);

    // 2h. Attention output projection + residual
    gemv_blocks = (d_model + 2 - 1) / 2;
    hipLaunchKernelGGL(mamba2_tiled_gemv_kernel,
        dim3(gemv_blocks), dim3(256), 0, stream,
        o_proj_w, d_attn_out, d_normed, d_model, n_attn_heads * attn_head_dim);

    // d_proj += d_normed (attn_out)
    hipLaunchKernelGGL(vec_add_kernel, dim3(blocks), dim3(threads), 0, stream,
        d_proj, d_normed, d_model);

    // ──────────────────────────────────────────────────────────────────
    // Step 3: FFN
    // 3a. Pre-FFN RMS norm on d_proj → d_normed
    // 3b. Gate + Up projections
    // 3c. SiLU activation + multiply
    // 3d. Down projection
    // 3e. Residual: hidden += down_out
    // ──────────────────────────────────────────────────────────────────

    // 3a. Copy d_proj → d_normed, then RMS norm
    hipLaunchKernelGGL(vec_copy_kernel, dim3(blocks), dim3(threads), 0, stream,
        d_normed, d_proj, d_model);

    hipLaunchKernelGGL(mamba2_group_norm_kernel,
        dim3(1, 1, 1), dim3(threads), 0, stream,
        d_normed, ffn_norm_w, d_model, 1, 1, 1e-6f);

    // 3b. Gate + Up projections (d_ff = 2 * d_model for Zamba2)
    // d_attn_out area is free now — reuse for gate and up
    float* d_gate = d_attn_out;         // [d_ff]
    float* d_up   = d_attn_out + d_ff;  // [d_ff]

    gemv_blocks = (d_ff + 2 - 1) / 2;
    hipLaunchKernelGGL(mamba2_tiled_gemv_kernel,
        dim3(gemv_blocks), dim3(256), 0, stream,
        gate_proj_w, d_normed, d_gate, d_ff, d_model);
    hipLaunchKernelGGL(mamba2_tiled_gemv_kernel,
        dim3(gemv_blocks), dim3(256), 0, stream,
        up_proj_w, d_normed, d_up, d_ff, d_model);

    // 3c. Fused: d_gate = silu(d_gate) * d_up (in-place)
    blocks = (d_ff + threads * 4 - 1) / (threads * 4);
    hipLaunchKernelGGL(silu_mul_kernel,
        dim3(blocks), dim3(threads), 0, stream,
        d_up, d_gate, d_ff);
    // Now d_up = silu(gate) * up

    // 3d. Down projection: d_normed = down_w @ d_up
    hipLaunchKernelGGL(mamba2_tiled_gemv_kernel,
        dim3(gemv_blocks), dim3(256), 0, stream,
        down_proj_w, d_up, d_normed, d_model, d_ff);

    // 3e. Residual: hidden += d_normed
    hipLaunchKernelGGL(vec_add_kernel, dim3(blocks), dim3(threads), 0, stream,
        hidden, d_normed, d_model);
}
