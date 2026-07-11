// Phase 6b — INT8 KV cache attention kernel for batch=1 decode + prefill.
//
// Mirrors src/kv_cache_attn.hip (Flash-Decoding online softmax), but K and V
// caches are stored as int8 with a per-(pos, kv_head) fp16 scale. This halves
// the KV DRAM footprint and bandwidth vs the fp16 cache at negligible accuracy
// cost when activations are well-conditioned (RMSNorm + RoPE post-quant).
//
// Layouts (device-resident):
//   Q        : FP16 [num_q_heads, head_dim]                                        — current token
//   K_cache  : INT8 [seq_len, num_kv_heads, head_dim]                              — past K
//   V_cache  : INT8 [seq_len, num_kv_heads, head_dim]                              — past V
//   K_scales : FP16 [seq_len, num_kv_heads]                                        — per-slot/head scale
//   V_scales : FP16 [seq_len, num_kv_heads]                                        — per-slot/head scale
//   out      : FP16 [num_q_heads, head_dim]                                        — attention output
//
// Dequant happens on-the-fly inside the dot product / V accumulation:
//   kf = (float)k_i8 * (float)K_scales[pos * num_kv_heads + kv_head]
//   vf = (float)v_i8 * (float)V_scales[pos * num_kv_heads + kv_head]
//
// The per-(pos, kv_head) scale is uniform across the warp while all threads
// iterate over head_dim for a given (pos, kv_head) pair, so the scale is loaded
// once per position by the whole block (broadcast via shared memory). No
// divergent scale reads.

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

// Fast-math sentinel — see prim_kernels.hip. -FLT_MAX avoids the
// -Wnan-infinity-disabled UB that -INFINITY triggers under -ffast-math.
constexpr float NEG_SENTINEL = -FLT_MAX;

__global__ __launch_bounds__(BLOCK)
void kv_cache_attn_decode_i8_kernel(
    const __half*  __restrict__ Q,
    const int8_t*  __restrict__ K,
    const int8_t*  __restrict__ V,
    const __half*  __restrict__ K_scales,   // [seq_len, num_kv_heads]
    const __half*  __restrict__ V_scales,   // [seq_len, num_kv_heads]
    __half*        __restrict__ out,
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

    __shared__ __half Q_shared[MAX_HEAD_DIM];
    for (int d = tid; d < head_dim; d += BLOCK) {
        Q_shared[d] = Q[(size_t)h * head_dim + d];
    }
    __syncthreads();

    float m = NEG_SENTINEL;
    float l = 0.0f;
    const int elems_per_thread = (head_dim + BLOCK - 1) / BLOCK;
    float o_local[8] = {0};

    // Single-barrier-per-step reduction with double-buffered warp-partial LDS:
    //   iter t writes s_sum[buf], syncthreads, then everyone reads s_sum[buf].
    //   iter t+1 writes s_sum[buf^1], so a fast warp cannot overwrite a slot a
    //   slow warp is still reading in iter t. The __syncthreads() at the top of
    //   iter t+1 ensures slow warps have finished their iter-t read before any
    //   warp proceeds to iter t+1's post-reduce path. Net: 1 barrier/seq-step
    //   instead of the original 3 (scale broadcast + two-stage warp reduce),
    //   which at seq_len=1024 × 20 heads saves ~40k block-wide barriers/call.
    //
    // Scales are loaded directly from global: every lane in the block hits the
    // same 2-byte address, so the hardware serves them as a single cache-line
    // broadcast without an explicit LDS hop.
    constexpr int NWARPS = BLOCK / WARP;
    __shared__ float s_sum[2][NWARPS];
    int buf = 0;

    for (int t = 0; t < seq_len; ++t) {
        const int8_t* K_row = K + ((size_t)t * num_kv_heads + kv_head) * head_dim;
        const int8_t* V_row = V + ((size_t)t * num_kv_heads + kv_head) * head_dim;

        const size_t sc_idx = (size_t)t * num_kv_heads + kv_head;
        const float ks = (float)K_scales[sc_idx];
        const float vs = (float)V_scales[sc_idx];

        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += BLOCK) {
            partial += (float)Q_shared[d] * (float)K_row[d];
        }
        #pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) partial += __shfl_xor(partial, o);
        if (lane == 0) s_sum[buf][wid] = partial;
        __syncthreads();
        float s_acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < NWARPS; ++w) s_acc += s_sum[buf][w];
        const float s = s_acc * ks * scale;
        buf ^= 1;

        const float m_new = (s > m) ? s : m;
        const float alpha = std::expf(m - m_new);
        const float beta  = std::expf(s - m_new);
        l = l * alpha + beta;
        #pragma unroll
        for (int k = 0; k < 8; ++k) o_local[k] *= alpha;
        const float beta_vs = beta * vs;
        for (int ei = 0; ei < elems_per_thread; ++ei) {
            int d = tid + ei * BLOCK;
            if (d < head_dim) {
                o_local[ei] += beta_vs * (float)V_row[d];
            }
        }
        m = m_new;
    }

    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int ei = 0; ei < elems_per_thread; ++ei) {
        int d = tid + ei * BLOCK;
        if (d < head_dim) {
            out[(size_t)h * head_dim + d] = __float2half(o_local[ei] * inv_l);
        }
    }
}

__global__ __launch_bounds__(BLOCK)
void kv_cache_attn_prefill_i8_kernel(
    const __half*  __restrict__ Q,
    const int8_t*  __restrict__ K,
    const int8_t*  __restrict__ V,
    const __half*  __restrict__ K_scales,
    const __half*  __restrict__ V_scales,
    __half*        __restrict__ out,
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
    float o_local[8] = {0};

    // See decode kernel above for the single-barrier-per-step rationale.
    constexpr int NWARPS = BLOCK / WARP;
    __shared__ float s_sum[2][NWARPS];
    int buf = 0;

    for (int tp = 0; tp <= t; ++tp) {
        const int8_t* K_row = K + ((size_t)tp * num_kv_heads + kv_head) * head_dim;
        const int8_t* V_row = V + ((size_t)tp * num_kv_heads + kv_head) * head_dim;

        const size_t sc_idx = (size_t)tp * num_kv_heads + kv_head;
        const float ks = (float)K_scales[sc_idx];
        const float vs = (float)V_scales[sc_idx];

        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += BLOCK) {
            partial += (float)Q_shared[d] * (float)K_row[d];
        }
        #pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) partial += __shfl_xor(partial, o);
        if (lane == 0) s_sum[buf][wid] = partial;
        __syncthreads();
        float s_acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < NWARPS; ++w) s_acc += s_sum[buf][w];
        const float s = s_acc * ks * scale;
        buf ^= 1;

        const float m_new = (s > m) ? s : m;
        const float alpha = std::expf(m - m_new);
        const float beta  = std::expf(s - m_new);
        l = l * alpha + beta;
        #pragma unroll
        for (int k = 0; k < 8; ++k) o_local[k] *= alpha;
        const float beta_vs = beta * vs;
        for (int ei = 0; ei < elems_per_thread; ++ei) {
            int d = tid + ei * BLOCK;
            if (d < head_dim) {
                o_local[ei] += beta_vs * (float)V_row[d];
            }
        }
        m = m_new;
    }

    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int ei = 0; ei < elems_per_thread; ++ei) {
        int d = tid + ei * BLOCK;
        if (d < head_dim) {
            out[((size_t)t * num_q_heads + h) * head_dim + d] = __float2half(o_local[ei] * inv_l);
        }
    }
}

// ----------------------------------------------------------------------------
// Multi-row per-row symmetric int8 quantizer.
// Each block handles one row.
//   in  : FP16 [num_rows, row_len]
//   out : INT8 [num_rows, row_len]
//   scales_fp16 : FP16 [num_rows], scale = max(|x|)/127 (clamped >= 1e-8)
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(BLOCK)
void quantize_fp16_to_i8_rowscale_kernel(
    const __half* __restrict__ in,
    int8_t*       __restrict__ out,
    __half*       __restrict__ scales,
    int num_rows, int row_len)
{
    const int row = blockIdx.x;
    if (row >= num_rows) return;
    const int tid = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;

    __shared__ float warp_max[BLOCK / WARP];
    __shared__ float s_scale;

    const __half* x = in  + (size_t)row * row_len;
    int8_t*       y = out + (size_t)row * row_len;

    float local = 0.0f;
    for (int k = tid; k < row_len; k += BLOCK) {
        float v = std::fabsf((float)x[k]);
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
            s_scale = s;
            scales[row] = __float2half(s);
        }
    }
    __syncthreads();

    const float inv = 1.0f / s_scale;
    for (int k = tid; k < row_len; k += BLOCK) {
        float v = (float)x[k] * inv;
        int q   = (int)std::round(v);
        if (q >  127) q =  127;
        if (q < -127) q = -127;
        y[k] = (int8_t)q;
    }
}

}  // namespace

extern "C" rcpp_status_t
rcpp_kv_cache_attn_decode_i8(const void* Q_dev,
                             const void* K_i8_dev, const void* V_i8_dev,
                             const void* K_scales_fp16_dev, const void* V_scales_fp16_dev,
                             void* out_dev,
                             int num_q_heads, int num_kv_heads, int head_dim,
                             int seq_len, float scale, void* stream)
{
    if (!Q_dev || !K_i8_dev || !V_i8_dev || !K_scales_fp16_dev || !V_scales_fp16_dev || !out_dev)
        return RCPP_INVALID_ARG;
    if (num_q_heads <= 0 || num_kv_heads <= 0)       return RCPP_INVALID_ARG;
    if (num_q_heads % num_kv_heads != 0)             return RCPP_INVALID_ARG;
    if (head_dim <= 0 || head_dim > MAX_HEAD_DIM)    return RCPP_INVALID_ARG;
    if (seq_len <= 0)                                return RCPP_INVALID_ARG;
    dim3 grid(num_q_heads, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(kv_cache_attn_decode_i8_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(Q_dev),
                       static_cast<const int8_t*>(K_i8_dev),
                       static_cast<const int8_t*>(V_i8_dev),
                       static_cast<const __half*>(K_scales_fp16_dev),
                       static_cast<const __half*>(V_scales_fp16_dev),
                       static_cast<__half*>(out_dev),
                       num_q_heads, num_kv_heads, head_dim, seq_len, scale);
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_kv_cache_attn_prefill_i8(const void* Q_dev,
                              const void* K_i8_dev, const void* V_i8_dev,
                              const void* K_scales_fp16_dev, const void* V_scales_fp16_dev,
                              void* out_dev,
                              int num_q_heads, int num_kv_heads, int head_dim,
                              int seq_len, float scale, void* stream)
{
    if (!Q_dev || !K_i8_dev || !V_i8_dev || !K_scales_fp16_dev || !V_scales_fp16_dev || !out_dev)
        return RCPP_INVALID_ARG;
    if (num_q_heads <= 0 || num_kv_heads <= 0)       return RCPP_INVALID_ARG;
    if (num_q_heads % num_kv_heads != 0)             return RCPP_INVALID_ARG;
    if (head_dim <= 0 || head_dim > MAX_HEAD_DIM)    return RCPP_INVALID_ARG;
    if (seq_len <= 0)                                return RCPP_INVALID_ARG;
    dim3 grid(seq_len, num_q_heads, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(kv_cache_attn_prefill_i8_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(Q_dev),
                       static_cast<const int8_t*>(K_i8_dev),
                       static_cast<const int8_t*>(V_i8_dev),
                       static_cast<const __half*>(K_scales_fp16_dev),
                       static_cast<const __half*>(V_scales_fp16_dev),
                       static_cast<__half*>(out_dev),
                       num_q_heads, num_kv_heads, head_dim, seq_len, scale);
    return RCPP_OK;
}

extern "C" rcpp_status_t
rcpp_quantize_fp16_to_i8_rowscale(const void* x_fp16_dev, void* out_i8_dev,
                                  void* scale_fp16_out_dev,
                                  int num_rows, int row_len, void* stream)
{
    if (!x_fp16_dev || !out_i8_dev || !scale_fp16_out_dev) return RCPP_INVALID_ARG;
    if (num_rows <= 0 || row_len <= 0)                     return RCPP_INVALID_ARG;
    dim3 grid(num_rows, 1, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(quantize_fp16_to_i8_rowscale_kernel,
                       grid, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(x_fp16_dev),
                       static_cast<int8_t*>(out_i8_dev),
                       static_cast<__half*>(scale_fp16_out_dev),
                       num_rows, row_len);
    return RCPP_OK;
}
