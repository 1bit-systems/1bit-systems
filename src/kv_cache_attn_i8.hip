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
// DECODE kernel design (2026-07-18 rework, see #338 follow-up research):
// split-KV Flash-Decoding (matches src/kv_cache_attn_fd.hip's fp16 approach)
// COMBINED with GQA-group-per-block dequant sharing. Two separate fixes for
// two separate problems, needed together:
//
// 1. GQA-group-per-block: the original layout put one Q-head per block
//    (grid.x = num_q_heads). Under GQA, `gqa_ratio` Q-heads share one
//    KV-head, so `gqa_ratio` blocks each independently re-read *and*
//    re-dequantize the exact same K/V row from global memory every
//    position, even though the raw int8 bytes and scale are identical
//    across the group. This mirrors a bug independently found and fixed
//    upstream for llama.cpp's CUDA/HIP flash-attention tile kernel
//    (Nathanw1014/llama.cpp@strix-halo-fa-fixes, same gfx1151 target):
//    dequant fused into a per-Q-head dot product can never be shared, so
//    it gets paid gqa_ratio times over. Fix: one block per KV-head, one
//    warp per Q-head in that KV-head's group; the whole block loads
//    K_row/V_row into shared memory ONCE per position, each warp then does
//    its own independent dot product / online-softmax update against that
//    shared copy. Q·K itself is NOT shared (each Q-head's query differs
//    and must be scored independently) — only the KV read + dequant is.
//
// 2. But (1) alone *regresses* throughput here: it shrinks the grid from
//    num_q_heads blocks down to num_kv_heads blocks — e.g. 20 -> 5 blocks
//    on BitNet-2B-4T shape, badly under-subscribing gfx1151's 32-40 CUs
//    (measured: 0.80-0.99x vs fp16, worse than the original 1.08-1.33x).
//    This is the exact occupancy problem src/kv_cache_attn_fd.hip's header
//    comment already documents for the fp16 kernel ("catastrophically
//    under-subscribed... ~4-5 GB/s off a ~256 GB/s peak"). Fix: split the
//    seq_len axis into TILE-sized chunks the same way kv_cache_attn_fd.hip
//    does, grid = (num_kv_heads, num_kv_tiles) instead of (num_kv_heads),
//    each block computing an unnormalized partial (o, m, l) for its tile;
//    a second small reduce pass (kv_cache_attn_reduce_i8_kernel, ported
//    from kv_cache_attn_fd.hip's dtype-agnostic reduce — same partials
//    layout [num_q_heads, num_kv_tiles, head_dim+2], reused via source
//    duplication since it's defined in a different translation unit)
//    combines tiles into the final per-Q-head output. At long seq_len this
//    also multiplies occupancy well past the original single-pass kernel's
//    (e.g. seq_len=2048, TILE=32: 5 kv_heads * 64 tiles = 320 blocks vs the
//    original 20), on top of eliminating the redundant KV reads. TILE=32
//    (smaller than kv_cache_attn_fd.hip's fp16 TILE=128) was chosen by
//    sweeping {16, 32, 128} on real hardware via tools/bench_kv_i8.cpp: 32
//    is the best balance — it keeps the short-context case (seq_len=64,
//    otherwise still occupancy-starved at 1 tile) solidly ahead of fp16
//    without giving up much of the much larger long-context win.
//
// Verified bit-for-bit-equivalent-within-tolerance against the pre-rework
// kernel and against the fp16 reference kernel via tools/bench_kv_i8.cpp on
// real gfx1151 hardware; see that tool's output for measured speedup.
//
// PREFILL kernel below is unchanged (still one block per (t, Q-head) pair)
// — same redundancy exists there too, but decode is the hot path for
// interactive/streaming generation and is what this rework targets; fixing
// prefill the same way is a natural follow-up, not done here.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cfloat>
#include <cstdint>
#include <cmath>
#include <mutex>

#include "rocm_cpp/ck_gemm.h"

namespace {

constexpr int BLOCK = 128;
constexpr int WARP  = 32;
constexpr int MAX_HEAD_DIM = 256;
constexpr int TILE  = 32;   // swept {16,32,128} on real hardware — see file header
// Max GQA ratio the decode kernel's static shared-memory layout supports —
// generous enough for every realistic config (BitNet-2B-4T=4, Qwen3-family
// up to 8); rcpp_kv_cache_attn_decode_i8() rejects anything larger rather
// than silently truncating the group.
constexpr int MAX_GQA_RATIO = 16;
constexpr int MAX_BLOCK_DECODE = MAX_GQA_RATIO * WARP;

// Fast-math sentinel — see prim_kernels.hip. -FLT_MAX avoids the
// -Wnan-infinity-disabled UB that -INFINITY triggers under -ffast-math.
constexpr float NEG_SENTINEL = -FLT_MAX;

// Pass 1: one block per (kv_head, kv_tile); one warp per Q-head within that
// kv_head's GQA group. Writes unnormalized partials
// [num_q_heads, num_kv_tiles, head_dim+2] — same layout as
// kv_cache_attn_fd.hip's fp16 split kernel, so the reduce pass below is
// dtype-agnostic and structurally identical to that file's pass 2.
// blockDim.x must be exactly gqa_ratio * WARP, set by the launcher below.
__global__ __launch_bounds__(MAX_BLOCK_DECODE)
void kv_cache_attn_decode_i8_split_kernel(
    const __half*  __restrict__ Q,
    const int8_t*  __restrict__ K,
    const int8_t*  __restrict__ V,
    const __half*  __restrict__ K_scales,   // [seq_len, num_kv_heads]
    const __half*  __restrict__ V_scales,   // [seq_len, num_kv_heads]
    float*         __restrict__ partials,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, int num_kv_tiles, float scale, int gqa_ratio)
{
    const int kv_head = blockIdx.x;
    const int tile     = blockIdx.y;
    if (kv_head >= num_kv_heads || tile >= num_kv_tiles) return;

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP - 1);
    const int wid  = tid / WARP;         // which Q-head within this group
    if (wid >= gqa_ratio) return;
    const int h = kv_head * gqa_ratio + wid;   // global Q-head index

    const int t_begin = tile * TILE;
    const int t_end   = min(t_begin + TILE, seq_len);

    // Each warp stages its own Q-head's query vector.
    __shared__ __half Q_shared[MAX_GQA_RATIO][MAX_HEAD_DIM];
    for (int d = lane; d < head_dim; d += WARP) {
        Q_shared[wid][d] = Q[(size_t)h * head_dim + d];
    }

    // K/V row + scales for the current position, shared by every warp in
    // the block (loaded once per position, cooperatively, by all threads).
    __shared__ int8_t K_shared[MAX_HEAD_DIM];
    __shared__ int8_t V_shared[MAX_HEAD_DIM];
    __shared__ float  ks_shared, vs_shared;
    __syncthreads();  // Q_shared writes visible before first load below

    float m = NEG_SENTINEL;
    float l = 0.0f;
    const int elems_per_thread = (head_dim + WARP - 1) / WARP;
    float o_local[8] = {0};

    // Degenerate tile (past seq_len) — emit neutral partials for this
    // warp's Q-head so the reduce pass skips it cleanly.
    if (t_begin >= t_end) {
        float* out_base = partials + ((size_t)h * num_kv_tiles + tile) * (head_dim + 2);
        if (lane == 0) {
            out_base[head_dim]     = NEG_SENTINEL;
            out_base[head_dim + 1] = 0.0f;
        }
        for (int d = lane; d < head_dim; d += WARP) out_base[d] = 0.0f;
        return;
    }

    for (int t = t_begin; t < t_end; ++t) {
        const int8_t* K_row = K + ((size_t)t * num_kv_heads + kv_head) * head_dim;
        const int8_t* V_row = V + ((size_t)t * num_kv_heads + kv_head) * head_dim;
        const size_t sc_idx = (size_t)t * num_kv_heads + kv_head;

        for (int d = tid; d < head_dim; d += blockDim.x) {
            K_shared[d] = K_row[d];
            V_shared[d] = V_row[d];
        }
        if (tid == 0) {
            ks_shared = (float)K_scales[sc_idx];
            vs_shared = (float)V_scales[sc_idx];
        }
        __syncthreads();

        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += WARP) {
            partial += (float)Q_shared[wid][d] * (float)K_shared[d];
        }
        #pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) partial += __shfl_xor(partial, o);
        const float s = partial * ks_shared * scale;

        const float m_new = (s > m) ? s : m;
        const float alpha = std::expf(m - m_new);
        const float beta  = std::expf(s - m_new);
        l = l * alpha + beta;
        #pragma unroll
        for (int k = 0; k < 8; ++k) o_local[k] *= alpha;
        const float beta_vs = beta * vs_shared;
        for (int ei = 0; ei < elems_per_thread; ++ei) {
            int d = lane + ei * WARP;
            if (d < head_dim) {
                o_local[ei] += beta_vs * (float)V_shared[d];
            }
        }
        m = m_new;

        // Ensure every warp is done reading K_shared/V_shared for this
        // position before the next iteration's cooperative load overwrites
        // them.
        __syncthreads();
    }

    // Emit *unnormalized* partials — pass 2 does the final normalize.
    float* out_base = partials + ((size_t)h * num_kv_tiles + tile) * (head_dim + 2);
    for (int ei = 0; ei < elems_per_thread; ++ei) {
        int d = lane + ei * WARP;
        if (d < head_dim) out_base[d] = o_local[ei];
    }
    if (lane == 0) {
        out_base[head_dim]     = m;
        out_base[head_dim + 1] = l;
    }
}

// Pass 2: one block per Q-head. Structurally identical to
// kv_cache_attn_fd.hip's kv_cache_attn_reduce_kernel (that one has internal
// linkage in a different translation unit, so it's duplicated here rather
// than shared — matches this codebase's existing per-file self-containment
// convention, e.g. the HIP_CHECK macro).
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
    const int elems_per_thread = (head_dim + BLOCK - 1) / BLOCK;
    float o_local[8] = {0};

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
        for (int k = 0; k < 8; ++k) o_local[k] *= alpha;

        for (int ei = 0; ei < elems_per_thread; ++ei) {
            int d = tid + ei * BLOCK;
            if (d < head_dim) o_local[ei] += beta * in_base[d];
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

// Growing-only scratch for pass-1 partials, same pattern as
// kv_cache_attn_fd.hip's PartialsCache (separate instance — different TU).
struct I8PartialsCache {
    std::mutex mu;
    float*     ptr   = nullptr;
    size_t     cap_f = 0;

    float* acquire(size_t need_f) {
        std::lock_guard<std::mutex> g(mu);
        if (need_f <= cap_f) return ptr;
        if (ptr) {
            hipError_t fst = hipFree(ptr);
            (void)fst;  // [[nodiscard]]; best-effort free, see fd.hip
        }
        size_t new_cap = (cap_f == 0) ? need_f : cap_f;
        while (new_cap < need_f) new_cap = new_cap + (new_cap >> 1) + 1;
        void* p = nullptr;
        hipError_t st = hipMalloc(&p, new_cap * sizeof(float));
        if (st != hipSuccess) { ptr = nullptr; cap_f = 0; return nullptr; }
        ptr   = static_cast<float*>(p);
        cap_f = new_cap;
        return ptr;
    }
};

I8PartialsCache& i8_partials_cache() {
    static I8PartialsCache c;
    return c;
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
    const int gqa_ratio = num_q_heads / num_kv_heads;
    if (gqa_ratio > MAX_GQA_RATIO)                   return RCPP_INVALID_ARG;

    const int num_kv_tiles = (seq_len + TILE - 1) / TILE;
    const size_t need_f =
        (size_t)num_q_heads * (size_t)num_kv_tiles * (size_t)(head_dim + 2);
    float* partials = i8_partials_cache().acquire(need_f);
    if (!partials) return RCPP_HIP_ERROR;

    // Pass 1 — fill partials. One block per (kv_head, tile), one warp per
    // Q-head in that kv_head's GQA group.
    dim3 grid1(num_kv_heads, num_kv_tiles, 1), block1(gqa_ratio * WARP, 1, 1);
    hipLaunchKernelGGL(kv_cache_attn_decode_i8_split_kernel,
                       grid1, block1, 0, (hipStream_t)stream,
                       static_cast<const __half*>(Q_dev),
                       static_cast<const int8_t*>(K_i8_dev),
                       static_cast<const int8_t*>(V_i8_dev),
                       static_cast<const __half*>(K_scales_fp16_dev),
                       static_cast<const __half*>(V_scales_fp16_dev),
                       partials,
                       num_q_heads, num_kv_heads, head_dim,
                       seq_len, num_kv_tiles, scale, gqa_ratio);

    // Pass 2 — reduce tiles into final fp16 out. One block per Q-head.
    dim3 grid2(num_q_heads, 1, 1), block2(BLOCK, 1, 1);
    hipLaunchKernelGGL(kv_cache_attn_reduce_i8_kernel,
                       grid2, block2, 0, (hipStream_t)stream,
                       partials,
                       static_cast<__half*>(out_dev),
                       num_q_heads, head_dim, num_kv_tiles);
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
