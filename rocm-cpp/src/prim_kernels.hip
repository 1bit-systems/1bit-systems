// Primitive kernels for MLX-free BitNet decode on gfx1151.
//
// Five small kernels that together with Phase 5 GEMV + Phase 4i prefill
// complete the BitNet decode math path without pulling in MLX:
//
//   1. rcpp_quantize_fp16_to_i8  — activation quantization (FP16 -> INT8 + scale)
//   2. rcpp_rmsnorm_fp16          — RMSNorm over the last dim
//   3. rcpp_rope_fp16             — rotary position embedding on Q/K
//   4. rcpp_silu_glu_fp16         — gate * SiLU(up) for SwiGLU FFN
//   5. rcpp_embedding_lookup_fp16 — token id -> FP16 embedding vector
//
// All kernels are single-block-per-output-item. Target: batch=1 decode.
// Batched variants can come later; decode is the primary use case.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cfloat>
#include <cstdint>
#include <cmath>

#include "rocm_cpp/ck_gemm.h"

namespace {

constexpr int BLOCK = 256;
constexpr int WARP  = 32;

// Fast-math sentinel for "smallest representable float" in reductions.
// Compiled with -ffast-math; plain `-INFINITY` is UB under that flag
// (clang diagnostic -Wnan-infinity-disabled). -FLT_MAX is finite and
// preserves the "less than any real input" property for argmax/softmax-max
// seeds. expf(-FLT_MAX) flushes to 0 identically under fast-math.
constexpr float NEG_SENTINEL = -FLT_MAX;

// ============================================================================
// 1. Activation quantization — FP16 [K] -> INT8 [K] + scalar scale
//    scale is computed as max(|x|) / 127 so INT8 values stay in [-127, +127].
// ============================================================================

__global__ void quantize_fp16_to_i8_kernel(
    const __half* __restrict__ x_fp16,
    int8_t*       __restrict__ x_i8,
    float*        __restrict__ out_scale,   // [1]
    int K)
{
    __shared__ float warp_max[BLOCK / WARP];
    __shared__ float scale_s;

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;

    // Block-wide max(|x|) reduction
    float local = 0.0f;
    for (int k = tid; k < K; k += BLOCK) {
        float v = std::fabsf((float)x_fp16[k]);
        if (v > local) local = v;
    }
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) {
        float other = __shfl_xor(local, o);
        if (other > local) local = other;
    }
    if (lane == 0) warp_max[wid] = local;
    __syncthreads();

    if (wid == 0) {
        float v = (lane < BLOCK / WARP) ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) {
            float other = __shfl_xor(v, o);
            if (other > v) v = other;
        }
        if (lane == 0) {
            float s = v / 127.0f;
            if (s < 1e-8f) s = 1e-8f;
            scale_s = s;
            *out_scale = s;
        }
    }
    __syncthreads();

    const float inv = 1.0f / scale_s;
    for (int k = tid; k < K; k += BLOCK) {
        float v = (float)x_fp16[k] * inv;
        int q   = (int)std::round(v);
        if (q >  127) q =  127;
        if (q < -127) q = -127;
        x_i8[k] = (int8_t)q;
    }
}

// ============================================================================
// 2. RMSNorm — y = (x / sqrt(mean(x^2) + eps)) * weight
//    Single block per output vector. Batch=1 decode.
// ============================================================================

__global__ void rmsnorm_fp16_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half*       __restrict__ y,
    float                       eps,
    int K)
{
    __shared__ float warp_sum[BLOCK / WARP];
    __shared__ float rms_inv_s;

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;

    // Sum of squares
    float local = 0.0f;
    for (int k = tid; k < K; k += BLOCK) {
        float v = (float)x[k];
        local += v * v;
    }
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) local += __shfl_xor(local, o);
    if (lane == 0) warp_sum[wid] = local;
    __syncthreads();

    if (wid == 0) {
        float s = (lane < BLOCK / WARP) ? warp_sum[lane] : 0.0f;
        #pragma unroll
        for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) s += __shfl_xor(s, o);
        if (lane == 0) {
            float mean = s / (float)K;
            rms_inv_s = 1.0f / std::sqrtf(mean + eps);
        }
    }
    __syncthreads();

    const float inv = rms_inv_s;
    for (int k = tid; k < K; k += BLOCK) {
        float v = (float)x[k] * inv * (float)weight[k];
        y[k] = __float2half(v);
    }
}

// ============================================================================
// 3. RoPE — apply rotary embedding to [num_heads, head_dim] at position pos.
//    HuggingFace LLaMA / BitNet convention (split-half, NOT interleaved):
//      for i in [0, hd/2):
//        row[i]       <- cos*row[i]       - sin*row[i+hd/2]
//        row[i+hd/2]  <- sin*row[i]_orig  + cos*row[i+hd/2]
//    freq[i] = 1 / theta^(2i / head_dim), angle = pos * freq
//    Both halves share cos/sin per index (HF does `cat(freqs, freqs)`).
//    Single block handles one head; each thread owns one i in [0, hd/2)
//    and writes BOTH row[i] and row[i+hd/2] after reading both originals.
// ============================================================================

__global__ void rope_fp16_kernel(
    __half*  __restrict__ x,               // [num_heads, head_dim]
    int                   pos,
    float                 theta,
    int                   num_heads,
    int                   head_dim)
{
    const int h = blockIdx.x;
    if (h >= num_heads) return;
    const int tid = threadIdx.x;
    const int half_dim = head_dim / 2;

    __half* row = x + (size_t)h * head_dim;
    for (int i = tid; i < half_dim; i += blockDim.x) {
        const float freq = 1.0f / std::powf(theta, 2.0f * (float)i / (float)head_dim);
        const float angle = (float)pos * freq;
        const float c = std::cosf(angle);
        const float s = std::sinf(angle);
        const float x0 = (float)row[i];             // lower half
        const float x1 = (float)row[i + half_dim];  // upper half
        row[i]            = __float2half(c * x0 - s * x1);
        row[i + half_dim] = __float2half(s * x0 + c * x1);
    }
}

// ============================================================================
// 4. SiLU-gated — y[i] = silu(up[i]) * gate[i]
//    silu(z) = z * sigmoid(z) = z / (1 + exp(-z))
// ============================================================================

__global__ void silu_glu_fp16_kernel(
    const __half* __restrict__ up,
    const __half* __restrict__ gate,
    __half*       __restrict__ y,
    int                       N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    const float u = (float)up[tid];
    const float g = (float)gate[tid];
    const float silu = u / (1.0f + std::expf(-u));
    y[tid] = __float2half(silu * g);
}

// ReLU² GLU — BitNet-b1.58 uses relu² as FFN activation (hidden_act="relu2"):
//   y[i] = relu(gate[i])² * up[i]
__global__ void relu2_glu_fp16_kernel(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half*       __restrict__ y,
    int                       N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    const float g = (float)gate[tid];
    const float u = (float)up[tid];
    const float r = g > 0.0f ? g : 0.0f;
    y[tid] = __float2half(r * r * u);
}

// Fused BitNet FFN activation: (relu²(gate) * up) normalized by ffn_sub_norm.
// The un-normalized intermediate on BitNet-b1.58 reaches 1e9, overflowing
// FP16 (max 6.5e4). Keep it in FP32 inside the kernel, divide by the RMS
// scale immediately, then emit the FP16 normalized tensor ready for quant.
__global__ __launch_bounds__(BLOCK)
void relu2_glu_rmsnorm_fp16_kernel(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    const __half* __restrict__ ffn_sub_norm_w,
    __half*       __restrict__ y_out,
    float eps, int N)
{
    __shared__ float warp_sum[BLOCK / WARP];
    __shared__ float rms_inv_s;
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;

    // First pass: compute sum of squares of activation (FP32 throughout).
    float local = 0.0f;
    for (int k = tid; k < N; k += BLOCK) {
        const float g = (float)gate[k];
        const float u = (float)up[k];
        const float r = g > 0.0f ? g : 0.0f;
        const float v = r * r * u;
        local += v * v;
    }
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) local += __shfl_xor(local, o);
    if (lane == 0) warp_sum[wid] = local;
    __syncthreads();
    if (wid == 0) {
        float s = (lane < BLOCK / WARP) ? warp_sum[lane] : 0.0f;
        #pragma unroll
        for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) s += __shfl_xor(s, o);
        if (lane == 0) {
            float mean = s / (float)N;
            rms_inv_s = 1.0f / std::sqrtf(mean + eps);
        }
    }
    __syncthreads();

    // Second pass: recompute activation and emit (v * inv_rms * weight) as FP16.
    const float inv = rms_inv_s;
    for (int k = tid; k < N; k += BLOCK) {
        const float g = (float)gate[k];
        const float u = (float)up[k];
        const float r = g > 0.0f ? g : 0.0f;
        const float v = r * r * u;
        y_out[k] = __float2half(v * inv * (float)ffn_sub_norm_w[k]);
    }
}

// ============================================================================
// 6. Residual add — y[i] += src[i]  (in-place)
// ============================================================================

__global__ void residual_add_fp16_kernel(
    __half* __restrict__ y,
    const __half* __restrict__ src,
    int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    y[tid] = __float2half((float)y[tid] + (float)src[tid]);
}

// FP32 residual accumulator: x_fp32[i] += (float)src_fp16[i]
// Used for the residual stream in deep networks where FP16 precision at
// the accumulated magnitude (±tens of thousands after 30 layers) loses
// too many bits on each tiny delta add.
__global__ void residual_add_fp32_from_fp16_kernel(
    float* __restrict__ x,
    const __half* __restrict__ src,
    int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    x[tid] += (float)src[tid];
}

// RMSNorm with FP32 input, FP16 weight, FP16 output. For use on the
// FP32 residual stream going into a sublayer.
__global__ void rmsnorm_fp32_in_fp16_out_kernel(
    const float* __restrict__ x,
    const __half* __restrict__ weight,
    __half* __restrict__ y,
    float eps, int K)
{
    __shared__ float warp_sum[BLOCK / WARP];
    __shared__ float rms_inv_s;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid = tid / WARP;

    float local = 0.0f;
    for (int k = tid; k < K; k += BLOCK) {
        float v = x[k];
        local += v * v;
    }
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) local += __shfl_xor(local, o);
    if (lane == 0) warp_sum[wid] = local;
    __syncthreads();
    if (wid == 0) {
        float s = (lane < BLOCK / WARP) ? warp_sum[lane] : 0.0f;
        #pragma unroll
        for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) s += __shfl_xor(s, o);
        if (lane == 0) {
            float mean = s / (float)K;
            rms_inv_s = 1.0f / std::sqrtf(mean + eps);
        }
    }
    __syncthreads();
    const float inv = rms_inv_s;
    for (int k = tid; k < K; k += BLOCK) {
        float v = x[k] * inv * (float)weight[k];
        y[k] = __float2half(v);
    }
}

// ============================================================================
// 11. FP16 × FP16 -> FP32 GEMV  (for the LM head on tied-embedding models)
//     W: FP16 [M, K]  row-major (= embedding table viewed as weight)
//     x: FP16 [K]
//     y: FP32 [M]
//
//     For BitNet-2B-4T the LM head is the embedding matrix (tied). This kernel
//     computes y[m] = sum_k W[m,k] * x[k] as FP32 accumulator.
//
//     Simple: one block per row. 128 threads reduce across K.
// ============================================================================

__global__ void fp16_gemv_kernel(
    const __half* __restrict__ W,
    const __half* __restrict__ x,
    float*        __restrict__ y,
    int M, int K)
{
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;

    __shared__ float s_sum[BLOCK / WARP];

    float local = 0.0f;
    const __half* row = W + (size_t)m * K;
    for (int k = tid; k < K; k += BLOCK) {
        local += (float)row[k] * (float)x[k];
    }
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) local += __shfl_xor(local, o);
    if (lane == 0) s_sum[wid] = local;
    __syncthreads();

    if (wid == 0) {
        float v = (lane < BLOCK / WARP) ? s_sum[lane] : 0.0f;
        #pragma unroll
        for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
        if (lane == 0) y[m] = v;
    }
}

// ============================================================================
// 12. FP32 -> FP16 in-place conversion (for chaining ternary GEMV output
//     (FP32) into the next kernel that wants FP16 input).
// ============================================================================

__global__ void fp32_to_fp16_kernel(
    const float* __restrict__ x_fp32,
    __half*      __restrict__ y_fp16,
    int N)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    y_fp16[tid] = __float2half(x_fp32[tid]);
}

// ============================================================================
// 10. Top-k filter — keep only the k largest logits, set the rest to -INF.
//     Caller then runs softmax + multinomial to sample within the top-k.
//     Simple approach: k iterations of block-wide argmax, save + mask.
//     Supports k up to MAX_TOPK.
// ============================================================================

constexpr int MAX_TOPK = 512;

__global__ void top_k_fp32_kernel(
    float* __restrict__ logits,
    int k, int V)
{
    __shared__ int picked_idx[MAX_TOPK];
    __shared__ float picked_val[MAX_TOPK];
    __shared__ float s_val[BLOCK / WARP];
    __shared__ int   s_idx[BLOCK / WARP];

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;

    for (int iter = 0; iter < k; ++iter) {
        float best_v = NEG_SENTINEL;
        int   best_i = 0;
        for (int v = tid; v < V; v += BLOCK) {
            float x = logits[v];
            if (x > best_v) { best_v = x; best_i = v; }
        }
        #pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) {
            float ov = __shfl_xor(best_v, o);
            int   oi = __shfl_xor(best_i, o);
            if (ov > best_v) { best_v = ov; best_i = oi; }
        }
        if (lane == 0) { s_val[wid] = best_v; s_idx[wid] = best_i; }
        __syncthreads();
        if (wid == 0) {
            float v = (lane < BLOCK / WARP) ? s_val[lane] : NEG_SENTINEL;
            int   i = (lane < BLOCK / WARP) ? s_idx[lane] : 0;
            #pragma unroll
            for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) {
                float ov = __shfl_xor(v, o);
                int   oi = __shfl_xor(i, o);
                if (ov > v) { v = ov; i = oi; }
            }
            if (lane == 0) {
                picked_idx[iter] = i;
                picked_val[iter] = v;
                logits[i] = NEG_SENTINEL;   // mask so next argmax skips it
            }
        }
        __syncthreads();
    }

    // Zero everything below top-k
    for (int v = tid; v < V; v += BLOCK) logits[v] = NEG_SENTINEL;
    __syncthreads();

    // Restore top-k values
    if (tid < k) logits[picked_idx[tid]] = picked_val[tid];
}

// ============================================================================
// 8. Softmax (FP32) with optional temperature — in place on logits.
//    y[v] = exp((x[v] - max) / T) / sum
// ============================================================================

__global__ void softmax_fp32_kernel(
    float* __restrict__ logits,
    int V, float inv_T)
{
    __shared__ float s_max;
    __shared__ float s_sum;
    __shared__ float s_val[BLOCK / WARP];

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;

    // 1) max reduction
    float lmax = NEG_SENTINEL;
    for (int v = tid; v < V; v += BLOCK) {
        float x = logits[v] * inv_T;
        if (x > lmax) lmax = x;
    }
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) {
        float ov = __shfl_xor(lmax, o);
        if (ov > lmax) lmax = ov;
    }
    if (lane == 0) s_val[wid] = lmax;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < BLOCK / WARP) ? s_val[lane] : NEG_SENTINEL;
        #pragma unroll
        for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) {
            float ov = __shfl_xor(v, o);
            if (ov > v) v = ov;
        }
        if (lane == 0) s_max = v;
    }
    __syncthreads();

    // 2) exp + sum
    float lsum = 0.0f;
    const float m = s_max;
    for (int v = tid; v < V; v += BLOCK) {
        float e = std::expf(logits[v] * inv_T - m);
        logits[v] = e;   // reuse buffer
        lsum += e;
    }
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) lsum += __shfl_xor(lsum, o);
    if (lane == 0) s_val[wid] = lsum;
    __syncthreads();
    if (wid == 0) {
        float s = (lane < BLOCK / WARP) ? s_val[lane] : 0.0f;
        #pragma unroll
        for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) s += __shfl_xor(s, o);
        if (lane == 0) s_sum = s;
    }
    __syncthreads();

    // 3) normalize
    const float inv = 1.0f / s_sum;
    for (int v = tid; v < V; v += BLOCK) logits[v] *= inv;
}

// ============================================================================
// 9. Multinomial sample — given probs[V] and a uniform r ∈ [0, 1),
//    return the smallest v where cumsum(probs[0..v]) > r.
//    Linear scan on one thread — correct but could be faster (inverse-CDF
//    binary search). Good enough for one-token-per-step decode.
// ============================================================================

__global__ void sample_multinomial_fp32_kernel(
    const float* __restrict__ probs,
    float                     r,
    int* __restrict__ out_idx,
    int V)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float cumu = 0.0f;
    int picked = V - 1;
    for (int v = 0; v < V; ++v) {
        cumu += probs[v];
        if (cumu >= r) { picked = v; break; }
    }
    *out_idx = picked;
}

// ============================================================================
// 7. Argmax sampling — returns index of max element in logits[V]
//    Single block, block-wide reduction.
// ============================================================================

__global__ void argmax_fp32_kernel(
    const float* __restrict__ logits,
    int* __restrict__ out_idx,
    int V)
{
    __shared__ float s_val[BLOCK / WARP];
    __shared__ int   s_idx[BLOCK / WARP];

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;

    float best_v = NEG_SENTINEL;
    int   best_i = 0;
    for (int v = tid; v < V; v += BLOCK) {
        float x = logits[v];
        if (x > best_v) { best_v = x; best_i = v; }
    }
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) {
        float ov = __shfl_xor(best_v, o);
        int   oi = __shfl_xor(best_i, o);
        if (ov > best_v) { best_v = ov; best_i = oi; }
    }
    if (lane == 0) { s_val[wid] = best_v; s_idx[wid] = best_i; }
    __syncthreads();

    if (wid == 0) {
        float v = (lane < BLOCK / WARP) ? s_val[lane] : NEG_SENTINEL;
        int   i = (lane < BLOCK / WARP) ? s_idx[lane] : 0;
        #pragma unroll
        for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) {
            float ov = __shfl_xor(v, o);
            int   oi = __shfl_xor(i, o);
            if (ov > v) { v = ov; i = oi; }
        }
        if (lane == 0) *out_idx = i;
    }
}

// ============================================================================
// 7b. Top-1 argmax over FP16 logits — paired with forward_token_greedy's
//     D→H-skip fast path. Identical reduction structure to argmax_fp32, but
//     reads half-precision input so callers whose final GEMV emits fp16
//     (e.g. a future fp16-output lm_head variant) don't need an extra
//     fp32 cast pass just to pick the argmax. Single block, block-wide
//     reduction, single int32 output.
// ============================================================================

__global__ void argmax_fp16_top1_kernel(
    const __half* __restrict__ logits,
    int* __restrict__ out_idx,
    int V)
{
    __shared__ float s_val[BLOCK / WARP];
    __shared__ int   s_idx[BLOCK / WARP];

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;

    float best_v = NEG_SENTINEL;
    int   best_i = 0;
    for (int v = tid; v < V; v += BLOCK) {
        float x = (float)logits[v];
        if (x > best_v) { best_v = x; best_i = v; }
    }
    #pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) {
        float ov = __shfl_xor(best_v, o);
        int   oi = __shfl_xor(best_i, o);
        if (ov > best_v) { best_v = ov; best_i = oi; }
    }
    if (lane == 0) { s_val[wid] = best_v; s_idx[wid] = best_i; }
    __syncthreads();

    if (wid == 0) {
        float v = (lane < BLOCK / WARP) ? s_val[lane] : NEG_SENTINEL;
        int   i = (lane < BLOCK / WARP) ? s_idx[lane] : 0;
        #pragma unroll
        for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) {
            float ov = __shfl_xor(v, o);
            int   oi = __shfl_xor(i, o);
            if (ov > v) { v = ov; i = oi; }
        }
        if (lane == 0) *out_idx = i;
    }
}

// ============================================================================
// 5. Embedding lookup — y[k] = E[token_id, k]  for k in 0..hidden-1
// ============================================================================

__global__ void embedding_lookup_fp16_kernel(
    const __half* __restrict__ embedding,   // [vocab, hidden]
    int                         token_id,
    __half*       __restrict__ y,            // [hidden]
    int                         hidden)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden) return;
    y[tid] = embedding[(size_t)token_id * hidden + tid];
}

}  // namespace

extern "C" {

rcpp_status_t
rcpp_quantize_fp16_to_i8(const void* x_fp16_dev, void* x_i8_dev,
                         float* scale_dev, int K, void* stream)
{
    if (!x_fp16_dev || !x_i8_dev || !scale_dev || K <= 0) return RCPP_INVALID_ARG;
    dim3 grid(1, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(quantize_fp16_to_i8_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(x_fp16_dev),
                       static_cast<int8_t*>(x_i8_dev),
                       scale_dev, K);
    return RCPP_OK;
}

rcpp_status_t
rcpp_rmsnorm_fp16(const void* x_dev, const void* weight_dev, void* y_dev,
                  float eps, int K, void* stream)
{
    if (!x_dev || !weight_dev || !y_dev || K <= 0) return RCPP_INVALID_ARG;
    dim3 grid(1, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(rmsnorm_fp16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(x_dev),
                       static_cast<const __half*>(weight_dev),
                       static_cast<__half*>(y_dev),
                       eps, K);
    return RCPP_OK;
}

rcpp_status_t
rcpp_rope_fp16(void* x_dev, int pos, float theta,
               int num_heads, int head_dim, void* stream)
{
    if (!x_dev || num_heads <= 0 || head_dim <= 0 || (head_dim & 1)) return RCPP_INVALID_ARG;
    // head_dim/2 is the real thread count (each thread owns one (i, i+hd/2) pair).
    // For BitNet-2B hd=128 → half_dim=64 → one block = 2 wave32 warps, which
    // matches the two coalesced 64-byte halves of the row. Rounding up to a
    // wave32 multiple keeps the tail warp fully masked rather than partial.
    const int half_dim = head_dim / 2;
    int block_threads = ((half_dim + WARP - 1) / WARP) * WARP;
    if (block_threads < WARP)  block_threads = WARP;
    if (block_threads > BLOCK) block_threads = BLOCK;
    dim3 grid(num_heads, 1, 1), block(block_threads, 1, 1);
    hipLaunchKernelGGL(rope_fp16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<__half*>(x_dev),
                       pos, theta, num_heads, head_dim);
    return RCPP_OK;
}

rcpp_status_t
rcpp_silu_glu_fp16(const void* up_dev, const void* gate_dev, void* y_dev,
                   int N, void* stream)
{
    if (!up_dev || !gate_dev || !y_dev || N <= 0) return RCPP_INVALID_ARG;
    dim3 grid((N + BLOCK - 1) / BLOCK, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(silu_glu_fp16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(up_dev),
                       static_cast<const __half*>(gate_dev),
                       static_cast<__half*>(y_dev),
                       N);
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_relu2_glu_fp16(const void* gate_dev, const void* up_dev, void* y_dev,
                    int N, void* stream)
{
    if (!gate_dev || !up_dev || !y_dev || N <= 0) return RCPP_INVALID_ARG;
    dim3 grid((N + BLOCK - 1) / BLOCK, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(relu2_glu_fp16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(gate_dev),
                       static_cast<const __half*>(up_dev),
                       static_cast<__half*>(y_dev),
                       N);
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_relu2_glu_rmsnorm_fp16(const void* gate_dev, const void* up_dev,
                            const void* ffn_sub_norm_dev,
                            void* y_dev, float eps, int N, void* stream)
{
    if (!gate_dev || !up_dev || !ffn_sub_norm_dev || !y_dev || N <= 0)
        return RCPP_INVALID_ARG;
    dim3 grid(1, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(relu2_glu_rmsnorm_fp16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(gate_dev),
                       static_cast<const __half*>(up_dev),
                       static_cast<const __half*>(ffn_sub_norm_dev),
                       static_cast<__half*>(y_dev),
                       eps, N);
    return RCPP_OK;
}

rcpp_status_t
rcpp_embedding_lookup_fp16(const void* embedding_dev, int token_id,
                           void* y_dev, int hidden, void* stream)
{
    if (!embedding_dev || !y_dev || token_id < 0 || hidden <= 0) return RCPP_INVALID_ARG;
    dim3 grid((hidden + BLOCK - 1) / BLOCK, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(embedding_lookup_fp16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(embedding_dev),
                       token_id,
                       static_cast<__half*>(y_dev),
                       hidden);
    return RCPP_OK;
}

rcpp_status_t
rcpp_residual_add_fp16(void* y_dev, const void* src_dev, int N, void* stream)
{
    if (!y_dev || !src_dev || N <= 0) return RCPP_INVALID_ARG;
    dim3 grid((N + BLOCK - 1) / BLOCK, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(residual_add_fp16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<__half*>(y_dev),
                       static_cast<const __half*>(src_dev),
                       N);
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_residual_add_fp32_from_fp16(void* x_fp32_dev, const void* src_fp16_dev,
                                 int N, void* stream)
{
    if (!x_fp32_dev || !src_fp16_dev || N <= 0) return RCPP_INVALID_ARG;
    dim3 grid((N + BLOCK - 1) / BLOCK, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(residual_add_fp32_from_fp16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<float*>(x_fp32_dev),
                       static_cast<const __half*>(src_fp16_dev),
                       N);
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_rmsnorm_fp32_in_fp16_out(const void* x_fp32_dev, const void* weight_dev,
                              void* y_fp16_dev, float eps, int K, void* stream)
{
    if (!x_fp32_dev || !weight_dev || !y_fp16_dev || K <= 0) return RCPP_INVALID_ARG;
    dim3 grid(1, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(rmsnorm_fp32_in_fp16_out_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const float*>(x_fp32_dev),
                       static_cast<const __half*>(weight_dev),
                       static_cast<__half*>(y_fp16_dev),
                       eps, K);
    return RCPP_OK;
}

rcpp_status_t
rcpp_argmax_fp32(const void* logits_dev, void* out_idx_dev, int V, void* stream)
{
    if (!logits_dev || !out_idx_dev || V <= 0) return RCPP_INVALID_ARG;
    dim3 grid(1, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(argmax_fp32_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const float*>(logits_dev),
                       static_cast<int*>(out_idx_dev),
                       V);
    return RCPP_OK;
}

rcpp_status_t
rcpp_argmax_fp16_top1(const void* logits_dev, void* out_idx_dev, int V, void* stream)
{
    if (!logits_dev || !out_idx_dev || V <= 0) return RCPP_INVALID_ARG;
    dim3 grid(1, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(argmax_fp16_top1_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(logits_dev),
                       static_cast<int*>(out_idx_dev),
                       V);
    return RCPP_OK;
}

rcpp_status_t
rcpp_softmax_fp32(void* logits_dev, int V, float temperature, void* stream)
{
    if (!logits_dev || V <= 0 || temperature <= 0.0f) return RCPP_INVALID_ARG;
    dim3 grid(1, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(softmax_fp32_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<float*>(logits_dev),
                       V, 1.0f / temperature);
    return RCPP_OK;
}

rcpp_status_t
rcpp_fp16_gemv(const void* W_dev, const void* x_dev, void* y_dev,
               int M, int K, void* stream)
{
    if (!W_dev || !x_dev || !y_dev || M <= 0 || K <= 0) return RCPP_INVALID_ARG;
    dim3 grid(M, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(fp16_gemv_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(W_dev),
                       static_cast<const __half*>(x_dev),
                       static_cast<float*>(y_dev),
                       M, K);
    return RCPP_OK;
}

rcpp_status_t
rcpp_fp32_to_fp16(const void* x_fp32_dev, void* y_fp16_dev, int N, void* stream)
{
    if (!x_fp32_dev || !y_fp16_dev || N <= 0) return RCPP_INVALID_ARG;
    dim3 grid((N + BLOCK - 1) / BLOCK, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(fp32_to_fp16_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const float*>(x_fp32_dev),
                       static_cast<__half*>(y_fp16_dev),
                       N);
    return RCPP_OK;
}

rcpp_status_t
rcpp_top_k_fp32(void* logits_dev, int k, int V, void* stream)
{
    if (!logits_dev || k <= 0 || V <= 0) return RCPP_INVALID_ARG;
    if (k > MAX_TOPK) return RCPP_UNSUPPORTED;
    if (k >= V) return RCPP_OK;   // no-op: all values survive
    dim3 grid(1, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(top_k_fp32_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<float*>(logits_dev),
                       k, V);
    return RCPP_OK;
}

rcpp_status_t
rcpp_sample_multinomial_fp32(const void* probs_dev, float r,
                             void* out_idx_dev, int V, void* stream)
{
    if (!probs_dev || !out_idx_dev || V <= 0) return RCPP_INVALID_ARG;
    if (r < 0.0f) r = 0.0f;
    if (r >= 1.0f) r = 1.0f - 1e-7f;
    dim3 grid(1, 1, 1), block(1, 1, 1);
    hipLaunchKernelGGL(sample_multinomial_fp32_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const float*>(probs_dev),
                       r,
                       static_cast<int*>(out_idx_dev),
                       V);
    return RCPP_OK;
}

}  // extern "C"
