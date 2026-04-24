// Phase 6 — KV cache attention kernel for batch=1 decode.
//
// Flash-Decoding-style online softmax: Q is small (this token), K/V cache is
// the long context. Single block per Q head, online accumulator over seq_len.
//
// Layouts (all device-resident):
//   Q       : FP16 [num_q_heads, head_dim]                   — current token
//   K_cache : FP16 [seq_len, num_kv_heads, head_dim]         — past K
//   V_cache : FP16 [seq_len, num_kv_heads, head_dim]         — past V
//   out     : FP16 [num_q_heads, head_dim]                   — attention output
//
// GQA: num_q_heads must be a multiple of num_kv_heads. For BitNet-2B-4T this
// is 20/5 = 4. Head h's K/V head index is h / (num_q_heads / num_kv_heads).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cfloat>
#include <cstdint>
#include <cmath>

#include "rocm_cpp/ck_gemm.h"

namespace {

constexpr int BLOCK = 128;
constexpr int WARP  = 32;
constexpr int MAX_HEAD_DIM = 256;
constexpr int O_LOCAL_MAX  = 8;   // fixed per-thread accumulator capacity

// WHY: each kernel below declares `float o_local[O_LOCAL_MAX]` and indexes
// it with `ei` bounded by `elems_per_thread = ceil(head_dim / BLOCK)`. If a
// future diff raises MAX_HEAD_DIM or lowers BLOCK such that the quotient
// exceeds O_LOCAL_MAX, the inner accumulator loops would write past the
// array. Fail loudly at compile time rather than silently corrupt the
// residual. The launcher separately enforces the runtime bound
// `head_dim <= MAX_HEAD_DIM` so this assert fully covers the dynamic case.
static_assert((MAX_HEAD_DIM + BLOCK - 1) / BLOCK <= O_LOCAL_MAX,
              "o_local[O_LOCAL_MAX] would overflow: raise O_LOCAL_MAX, raise "
              "BLOCK, or lower MAX_HEAD_DIM");

// Fast-math sentinel — see prim_kernels.hip for rationale.
// Fast-math makes plain -INFINITY UB; -FLT_MAX is finite and equivalent
// as the initial max for the online-softmax running maximum.
constexpr float NEG_SENTINEL = -FLT_MAX;

__global__ __launch_bounds__(BLOCK)
void kv_cache_attn_decode_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ out,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, float scale)
{
    const int h = blockIdx.x;
    if (h >= num_q_heads) return;

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head   = h / gqa_ratio;

    // Q for this head, loaded into registers.
    __shared__ __half Q_shared[MAX_HEAD_DIM];
    for (int d = tid; d < head_dim; d += BLOCK) {
        Q_shared[d] = Q[(size_t)h * head_dim + d];
    }
    __syncthreads();

    // Online softmax state: running max m, running exp-sum l, output accumulator o[d].
    // Output accumulator held in registers (head_dim up to 256 → each thread holds 2).
    float m = NEG_SENTINEL;
    float l = 0.0f;
    const int elems_per_thread = (head_dim + BLOCK - 1) / BLOCK;
    float o_local[O_LOCAL_MAX] = {0};  // static_assert at namespace scope proves
                                       // elems_per_thread <= O_LOCAL_MAX.

    // Per-warp shared for reductions
    __shared__ float s_max[BLOCK / WARP];
    __shared__ float s_sum[BLOCK / WARP];

    // Iterate over all KV positions, one at a time. Each thread computes a
    // partial dot of Q · K[t] along head_dim (cooperating across the block),
    // then the whole block sees S[t] via reduction.
    for (int t = 0; t < seq_len; ++t) {
        const __half* K_row = K + ((size_t)t * num_kv_heads + kv_head) * head_dim;

        // Cooperative dot product Q · K_row
        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += BLOCK) {
            partial += (float)Q_shared[d] * (float)K_row[d];
        }
        #pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) partial += __shfl_xor(partial, o);
        if (lane == 0) s_sum[wid] = partial;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < BLOCK / WARP) ? s_sum[lane] : 0.0f;
            #pragma unroll
            for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
            if (lane == 0) s_max[0] = v * scale;  // S[t]
        }
        __syncthreads();
        const float s = s_max[0];

        // Online softmax update
        const float m_new = (s > m) ? s : m;
        const float alpha = std::expf(m - m_new);
        const float beta  = std::expf(s - m_new);
        l = l * alpha + beta;
        // Scale existing output accumulator by alpha
        #pragma unroll
        for (int k = 0; k < O_LOCAL_MAX; ++k) o_local[k] *= alpha;
        // Accumulate V[t] weighted by beta
        const __half* V_row = V + ((size_t)t * num_kv_heads + kv_head) * head_dim;
        for (int ei = 0; ei < elems_per_thread; ++ei) {
            int d = tid + ei * BLOCK;
            if (d < head_dim) {
                o_local[ei] += beta * (float)V_row[d];
            }
        }
        m = m_new;
    }

    // Normalize + write out.
    // WHY: degenerate rows (all-masked / empty tile) leave l == 0, which makes
    // 1.0f/l == +inf and poisons the fp16 output with NaN once multiplied by the
    // zero accumulator (0 * inf = NaN). Clamp below a tiny epsilon — the output
    // accumulator is also zero in that case, so inv_l = 0 yields a finite zero
    // row, matching the i8 + FD variants (see kv_cache_attn_i8.hip:128,
    // kv_cache_attn_fd.hip:216).
    const float inv_l = (l > 1e-20f) ? (1.0f / l) : 0.0f;
    for (int ei = 0; ei < elems_per_thread; ++ei) {
        int d = tid + ei * BLOCK;
        if (d < head_dim) {
            out[(size_t)h * head_dim + d] = __float2half(o_local[ei] * inv_l);
        }
    }
}

}  // namespace

extern "C" rcpp_status_t
rcpp_kv_cache_attn_decode(const void* Q_dev, const void* K_dev, const void* V_dev,
                          void* out_dev,
                          int num_q_heads, int num_kv_heads, int head_dim,
                          int seq_len, float scale, void* stream)
{
    if (!Q_dev || !K_dev || !V_dev || !out_dev)          return RCPP_INVALID_ARG;
    if (num_q_heads <= 0 || num_kv_heads <= 0)           return RCPP_INVALID_ARG;
    if (num_q_heads % num_kv_heads != 0)                 return RCPP_INVALID_ARG;
    if (head_dim <= 0 || head_dim > MAX_HEAD_DIM)         return RCPP_INVALID_ARG;
    if (seq_len <= 0)                                     return RCPP_INVALID_ARG;
    dim3 grid(num_q_heads, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(kv_cache_attn_decode_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(Q_dev),
                       static_cast<const __half*>(K_dev),
                       static_cast<const __half*>(V_dev),
                       static_cast<__half*>(out_dev),
                       num_q_heads, num_kv_heads, head_dim, seq_len, scale);
    return RCPP_OK;
}

// ============================================================================
// Prefill attention — multi-token, causal mask.
//   Q: [seq_len, num_q_heads, head_dim]
//   K: [seq_len, num_kv_heads, head_dim]
//   V: [seq_len, num_kv_heads, head_dim]
//   out: [seq_len, num_q_heads, head_dim]
// Each output position t attends only to K/V[0..t]. One block per (t, h).
// ============================================================================

namespace {
__global__ __launch_bounds__(BLOCK)
void kv_cache_attn_prefill_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half*       __restrict__ out,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, float scale)
{
    const int t = blockIdx.x;   // Q position
    const int h = blockIdx.y;   // Q head
    if (t >= seq_len || h >= num_q_heads) return;

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head   = h / gqa_ratio;

    __shared__ __half Q_shared[MAX_HEAD_DIM];
    for (int d = tid; d < head_dim; d += BLOCK) {
        Q_shared[d] = Q[((size_t)t * num_q_heads + h) * head_dim + d];
    }
    __syncthreads();

    float m = NEG_SENTINEL;
    float l = 0.0f;
    const int elems_per_thread = (head_dim + BLOCK - 1) / BLOCK;
    float o_local[O_LOCAL_MAX] = {0};

    __shared__ float s_sum[BLOCK / WARP];
    __shared__ float s_scalar[1];

    // Causal: only attend to t' in 0..t inclusive
    for (int tp = 0; tp <= t; ++tp) {
        const __half* K_row = K + ((size_t)tp * num_kv_heads + kv_head) * head_dim;

        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += BLOCK) {
            partial += (float)Q_shared[d] * (float)K_row[d];
        }
        #pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) partial += __shfl_xor(partial, o);
        if (lane == 0) s_sum[wid] = partial;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < BLOCK / WARP) ? s_sum[lane] : 0.0f;
            #pragma unroll
            for (int o = (BLOCK / WARP) / 2; o > 0; o >>= 1) v += __shfl_xor(v, o);
            if (lane == 0) s_scalar[0] = v * scale;
        }
        __syncthreads();
        const float s = s_scalar[0];

        const float m_new = (s > m) ? s : m;
        const float alpha = std::expf(m - m_new);
        const float beta  = std::expf(s - m_new);
        l = l * alpha + beta;
        #pragma unroll
        for (int k = 0; k < O_LOCAL_MAX; ++k) o_local[k] *= alpha;
        const __half* V_row = V + ((size_t)tp * num_kv_heads + kv_head) * head_dim;
        for (int ei = 0; ei < elems_per_thread; ++ei) {
            int d = tid + ei * BLOCK;
            if (d < head_dim) {
                o_local[ei] += beta * (float)V_row[d];
            }
        }
        m = m_new;
    }

    // WHY: see decode-kernel comment above. In prefill the guard also covers
    // the t == 0 path (1 query attending to 1 key): l stays finite there, but
    // the guard is cheap and uniform across kernels.
    const float inv_l = (l > 1e-20f) ? (1.0f / l) : 0.0f;
    for (int ei = 0; ei < elems_per_thread; ++ei) {
        int d = tid + ei * BLOCK;
        if (d < head_dim) {
            out[((size_t)t * num_q_heads + h) * head_dim + d] = __float2half(o_local[ei] * inv_l);
        }
    }
}
}  // namespace

extern "C" rcpp_status_t
rcpp_kv_cache_attn_prefill(const void* Q_dev, const void* K_dev, const void* V_dev,
                           void* out_dev,
                           int num_q_heads, int num_kv_heads, int head_dim,
                           int seq_len, float scale, void* stream)
{
    if (!Q_dev || !K_dev || !V_dev || !out_dev)          return RCPP_INVALID_ARG;
    if (num_q_heads <= 0 || num_kv_heads <= 0)           return RCPP_INVALID_ARG;
    if (num_q_heads % num_kv_heads != 0)                 return RCPP_INVALID_ARG;
    if (head_dim <= 0 || head_dim > MAX_HEAD_DIM)         return RCPP_INVALID_ARG;
    if (seq_len <= 0)                                     return RCPP_INVALID_ARG;
    dim3 grid(seq_len, num_q_heads, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(kv_cache_attn_prefill_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(Q_dev),
                       static_cast<const __half*>(K_dev),
                       static_cast<const __half*>(V_dev),
                       static_cast<__half*>(out_dev),
                       num_q_heads, num_kv_heads, head_dim, seq_len, scale);
    return RCPP_OK;
}
