// Phase 6c-i8 — Split-KV Flash-Decoding attention over INT8 KV storage.
//
// Combines the bandwidth halving of kv_cache_attn_i8.hip (1-byte K/V + per-
// (pos, kv_head) fp16 scale) with the split-KV occupancy recovery of
// kv_cache_attn_fd.hip. The single-block-per-Q-head i8 path saturates ~20
// blocks on a 40-WGP gfx1151 — same starvation that drove the original FD
// port for the fp16 cache. Tile=128 + two-pass online-softmax recovers
// occupancy at long context.
//
// At seq_len=2048 we expect the same 6.78× the fp16 FD port hit at
// equivalent shapes (per project_attention_fd.md), composed multiplicatively
// with the 2× DRAM cut already realised by the i8 storage. Ship-relevant for
// long-context decode (project_i8_kv_unwired.md says I8 is now the default
// router pick for ctx > 512).
//
// This file is purely additive — kv_cache_attn_i8.hip stays in place and the
// new entry point `rcpp_kv_cache_attn_decode_fd_i8` lives alongside the
// single-block decoder so bitnet_decode (and lemond's engine) can opt in
// per-call.
//
// O_LOCAL is templated identically to fd.hip; see that file for the rationale.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cfloat>
#include <cstdint>
#include <cmath>
#include <mutex>

#include "rocm_cpp/ck_gemm.h"

namespace {

constexpr int BLOCK        = 128;
constexpr int WARP         = 32;
constexpr int MAX_HEAD_DIM = 256;
constexpr int TILE         = 128;
constexpr int O_LOCAL_MAX  = (MAX_HEAD_DIM + BLOCK - 1) / BLOCK;
static_assert(O_LOCAL_MAX == 2,
              "FD-i8 launcher dispatch table assumes O_LOCAL_MAX==2; if "
              "MAX_HEAD_DIM or BLOCK changed, extend the switch in the "
              "launcher to cover the new ceil(head_dim/BLOCK).");

constexpr float NEG_SENTINEL = -FLT_MAX;

// Pass 1 — int8 KV storage with per-(pos, kv_head) fp16 scales. Mirrors
// kv_cache_attn_decode_split_kernel<O_LOCAL> in fd.hip; the only material
// changes are K/V load type (int8_t), the per-tile scale broadcast (fp16
// scales[pos*num_kv_heads + kv_head]), and the dequant fold-in done as a
// scalar multiply on the fp32 dot/V accumulators.
template <int O_LOCAL>
__global__ __launch_bounds__(BLOCK)
void kv_cache_attn_decode_split_i8_kernel(
    const __half*  __restrict__ Q,
    const int8_t*  __restrict__ K,
    const int8_t*  __restrict__ V,
    const __half*  __restrict__ K_scales,   // [seq_len, num_kv_heads]
    const __half*  __restrict__ V_scales,   // [seq_len, num_kv_heads]
    float*         __restrict__ partials,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, int num_kv_tiles, float scale)
{
    const int h    = blockIdx.x;
    const int tile = blockIdx.y;
    if (h >= num_q_heads || tile >= num_kv_tiles) return;

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head   = h / gqa_ratio;

    const int t_begin = tile * TILE;
    const int t_end   = min(t_begin + TILE, seq_len);

    __shared__ __half Q_shared[MAX_HEAD_DIM];
    for (int d = tid; d < head_dim; d += BLOCK) {
        Q_shared[d] = Q[(size_t)h * head_dim + d];
    }
    __syncthreads();

    float m = NEG_SENTINEL;
    float l = 0.0f;
    float o_local[O_LOCAL] = {0};

    __shared__ float s_sum[BLOCK / WARP];
    __shared__ float s_scalar[1];

    if (t_begin >= t_end) {
        if (tid == 0) {
            float* out_base = partials + ((size_t)h * num_kv_tiles + tile) * (head_dim + 2);
            out_base[head_dim]     = NEG_SENTINEL;
            out_base[head_dim + 1] = 0.0f;
        }
        for (int d = tid; d < head_dim; d += BLOCK) {
            float* out_base = partials + ((size_t)h * num_kv_tiles + tile) * (head_dim + 2);
            out_base[d] = 0.0f;
        }
        return;
    }

    for (int t = t_begin; t < t_end; ++t) {
        const int8_t* K_row = K + ((size_t)t * num_kv_heads + kv_head) * head_dim;
        const int8_t* V_row = V + ((size_t)t * num_kv_heads + kv_head) * head_dim;

        // Per-(pos, kv_head) scales — identical broadcast pattern as the
        // single-block i8 kernel. All lanes hit the same 2-byte address →
        // hardware coalesces into one cache-line broadcast, no LDS hop.
        const size_t sc_idx = (size_t)t * num_kv_heads + kv_head;
        const float ks = (float)K_scales[sc_idx];
        const float vs = (float)V_scales[sc_idx];

        // Q · dequant(K_row)  — fold ks into the scalar multiply at the end
        // rather than per-element; preserves the int8 → fp16 → fp32 chain.
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
            // Fold ks (K dequant scale) into the softmax score in fp32.
            if (lane == 0) s_scalar[0] = v * ks * scale;
        }
        __syncthreads();
        const float s = s_scalar[0];

        const float m_new = (s > m) ? s : m;
        const float alpha = std::expf(m - m_new);
        const float beta  = std::expf(s - m_new);
        l = l * alpha + beta;

        #pragma unroll
        for (int k = 0; k < O_LOCAL; ++k) o_local[k] *= alpha;

        // V dequant scale folded into the accumulation step.
        const float beta_vs = beta * vs;
        #pragma unroll
        for (int ei = 0; ei < O_LOCAL; ++ei) {
            int d = tid + ei * BLOCK;
            if (d < head_dim) {
                o_local[ei] += beta_vs * (float)V_row[d];
            }
        }
        m = m_new;
    }

    // Emit *unnormalized* partials — pass 2 does the final normalize.
    float* out_base = partials + ((size_t)h * num_kv_tiles + tile) * (head_dim + 2);
    #pragma unroll
    for (int ei = 0; ei < O_LOCAL; ++ei) {
        int d = tid + ei * BLOCK;
        if (d < head_dim) {
            out_base[d] = o_local[ei];
        }
    }
    if (tid == 0) {
        out_base[head_dim]     = m;
        out_base[head_dim + 1] = l;
    }
}

// Pass 2 — identical to fd.hip's reduce. Partials are already in fp32 with
// the per-(pos, kv_head) scales folded into both the softmax score and the
// V-side accumulator, so the reduce only needs to combine (m, l, o) triples
// across tiles. We instantiate per O_LOCAL for the same VGPR reason as
// pass 1.
template <int O_LOCAL>
__global__ __launch_bounds__(BLOCK)
void kv_cache_attn_reduce_i8_kernel(
    const float*  __restrict__ partials,
    __half*       __restrict__ out,
    int num_q_heads, int head_dim, int num_kv_tiles)
{
    const int h = blockIdx.x;
    if (h >= num_q_heads) return;

    const int tid = threadIdx.x;

    float m = NEG_SENTINEL;
    float l = 0.0f;
    float o_local[O_LOCAL] = {0};

    __shared__ float s_ml[2];

    for (int tile = 0; tile < num_kv_tiles; ++tile) {
        const float* in_base =
            partials + ((size_t)h * num_kv_tiles + tile) * (head_dim + 2);

        if (tid == 0) {
            s_ml[0] = in_base[head_dim];
            s_ml[1] = in_base[head_dim + 1];
        }
        __syncthreads();
        const float tile_m = s_ml[0];
        const float tile_l = s_ml[1];
        __syncthreads();

        if (tile_l == 0.0f) continue;

        const float m_new = (tile_m > m) ? tile_m : m;
        const float alpha = std::expf(m - m_new);
        const float beta  = std::expf(tile_m - m_new);
        l = l * alpha + beta * tile_l;

        #pragma unroll
        for (int k = 0; k < O_LOCAL; ++k) o_local[k] *= alpha;

        #pragma unroll
        for (int ei = 0; ei < O_LOCAL; ++ei) {
            int d = tid + ei * BLOCK;
            if (d < head_dim) {
                o_local[ei] += beta * in_base[d];
            }
        }
        m = m_new;
    }

    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    #pragma unroll
    for (int ei = 0; ei < O_LOCAL; ++ei) {
        int d = tid + ei * BLOCK;
        if (d < head_dim) {
            out[(size_t)h * head_dim + d] = __float2half(o_local[ei] * inv_l);
        }
    }
}

// Cached growing-only partials buffer, identical pattern to fd.hip but kept
// SEPARATE so a mixed fp16/i8 workload doesn't thrash a shared cache. Both
// caches grow monotonically with seq_len during a decode stream and are
// freed (or leaked at exit) by the OS.
struct PartialsCacheI8 {
    std::mutex  mu;
    float*      ptr   = nullptr;
    size_t      cap_f = 0;

    float* acquire(size_t need_f) {
        std::lock_guard<std::mutex> g(mu);
        if (need_f <= cap_f) return ptr;
        if (ptr) {
            hipError_t fst = hipFree(ptr);
            (void)fst;
        }
        size_t new_cap = (cap_f == 0) ? need_f : cap_f;
        while (new_cap < need_f) new_cap = new_cap + (new_cap >> 1) + 1;
        void* p = nullptr;
        hipError_t st = hipMalloc(&p, new_cap * sizeof(float));
        if (st != hipSuccess) {
            ptr = nullptr;
            cap_f = 0;
            return nullptr;
        }
        ptr   = static_cast<float*>(p);
        cap_f = new_cap;
        return ptr;
    }
};

PartialsCacheI8& partials_cache_i8() {
    static PartialsCacheI8 c;
    return c;
}

}  // namespace

extern "C" rcpp_status_t
rcpp_kv_cache_attn_decode_fd_i8(const void* Q_dev,
                                const void* K_i8_dev, const void* V_i8_dev,
                                const void* K_scales_fp16_dev,
                                const void* V_scales_fp16_dev,
                                void* out_dev,
                                int num_q_heads, int num_kv_heads, int head_dim,
                                int seq_len, float scale, void* stream)
{
    if (!Q_dev || !K_i8_dev || !V_i8_dev ||
        !K_scales_fp16_dev || !V_scales_fp16_dev || !out_dev)
        return RCPP_INVALID_ARG;
    if (num_q_heads <= 0 || num_kv_heads <= 0)       return RCPP_INVALID_ARG;
    if (num_q_heads % num_kv_heads != 0)             return RCPP_INVALID_ARG;
    if (head_dim <= 0 || head_dim > MAX_HEAD_DIM)    return RCPP_INVALID_ARG;
    if (seq_len <= 0)                                return RCPP_INVALID_ARG;

    const int num_kv_tiles = (seq_len + TILE - 1) / TILE;
    const size_t need_f =
        (size_t)num_q_heads * (size_t)num_kv_tiles * (size_t)(head_dim + 2);
    float* partials = partials_cache_i8().acquire(need_f);
    if (!partials) return RCPP_HIP_ERROR;

    dim3 grid1(num_q_heads, num_kv_tiles, 1), block(BLOCK, 1, 1);
    const int o_local_needed = (head_dim + BLOCK - 1) / BLOCK;
    if (o_local_needed <= 1) {
        hipLaunchKernelGGL(kv_cache_attn_decode_split_i8_kernel<1>,
                           grid1, block, 0, (hipStream_t)stream,
                           static_cast<const __half*>(Q_dev),
                           static_cast<const int8_t*>(K_i8_dev),
                           static_cast<const int8_t*>(V_i8_dev),
                           static_cast<const __half*>(K_scales_fp16_dev),
                           static_cast<const __half*>(V_scales_fp16_dev),
                           partials,
                           num_q_heads, num_kv_heads, head_dim,
                           seq_len, num_kv_tiles, scale);
    } else {
        hipLaunchKernelGGL(kv_cache_attn_decode_split_i8_kernel<2>,
                           grid1, block, 0, (hipStream_t)stream,
                           static_cast<const __half*>(Q_dev),
                           static_cast<const int8_t*>(K_i8_dev),
                           static_cast<const int8_t*>(V_i8_dev),
                           static_cast<const __half*>(K_scales_fp16_dev),
                           static_cast<const __half*>(V_scales_fp16_dev),
                           partials,
                           num_q_heads, num_kv_heads, head_dim,
                           seq_len, num_kv_tiles, scale);
    }

    dim3 grid2(num_q_heads, 1, 1);
    if (o_local_needed <= 1) {
        hipLaunchKernelGGL(kv_cache_attn_reduce_i8_kernel<1>,
                           grid2, block, 0, (hipStream_t)stream,
                           partials,
                           static_cast<__half*>(out_dev),
                           num_q_heads, head_dim, num_kv_tiles);
    } else {
        hipLaunchKernelGGL(kv_cache_attn_reduce_i8_kernel<2>,
                           grid2, block, 0, (hipStream_t)stream,
                           partials,
                           static_cast<__half*>(out_dev),
                           num_q_heads, head_dim, num_kv_tiles);
    }

    return RCPP_OK;
}
