// Phase 6c — Split-KV Flash-Decoding attention for batch=1 decode.
//
// The baseline kv_cache_attn_decode_kernel assigns one block per Q head (20 on
// BitNet-2B-4T), which is catastrophically under-subscribed on gfx1151 — we
// measured ~4–5 GB/s off a ~256 GB/s LPDDR5 peak. The inner seq_len loop is
// also gated by 2 __syncthreads() per step plus a broadcast reduction.
//
// Flash-Decoding (Dao et al.) fix: split the KV axis into tiles, run one block
// per (q_head, kv_tile) to recover occupancy, then do a tiny second-pass reduce
// across tiles to combine the online-softmax partials.
//
// Pass 1 grid = (num_q_heads, num_kv_tiles)
//   each block computes (partial_out[HD], partial_m, partial_l) for its tile
//   stored in partials[num_q_heads, num_kv_tiles, HD+2]  (HD f32, m f32, l f32)
//
// Pass 2 grid = (num_q_heads)
//   each block scans its num_kv_tiles partials with the same m/l online-softmax
//   recipe and writes fp16 out[num_q_heads, head_dim].
//
// TILE=128 chosen so that:
//   - at seq_len=2048 we have 20*16 = 320 blocks (vs 20 before) — plenty of
//     room for the scheduler to fill gfx1151's 40 WGPs (80 SIMDs).
//   - at seq_len=64 we still split into 1 block-per-head (no regression) but
//     just replay the baseline's work structure.
//   - the partials buffer for a max-supported context (say 8192) stays tiny:
//     8192/128 = 64 tiles * 20 heads * 130 f32 = 666 KB.
//
// Single partials allocation is cached as a static buffer, grown on demand.

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
constexpr int TILE         = 128;    // Flash-Decoding KV-tile size (compile-time)
constexpr int O_LOCAL_MAX  = 8;      // fixed per-thread accumulator capacity

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

// Fast-math sentinel — see prim_kernels.hip. -FLT_MAX avoids the
// -Wnan-infinity-disabled UB that -INFINITY triggers under -ffast-math.
// The pass-2 reduce guards empty tiles with `if (tile_l == 0.0f) continue;`
// so the sentinel's exact value never participates in arithmetic — it's
// written at line 90 purely as a diagnostic marker.
constexpr float NEG_SENTINEL = -FLT_MAX;

// Pass 1: one block per (h, kv_tile). Computes online-softmax *over the tile*.
// Writes [num_q_heads, num_kv_tiles, head_dim+2] scratch. Layout is:
//   partials[h, tile, 0..head_dim-1]  : output accumulator (f32)
//   partials[h, tile, head_dim]       : partial m  (running max)
//   partials[h, tile, head_dim+1]     : partial l  (running sum of exp)
__global__ __launch_bounds__(BLOCK)
void kv_cache_attn_decode_split_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    float*        __restrict__ partials,
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
    const int elems_per_thread = (head_dim + BLOCK - 1) / BLOCK;
    float o_local[O_LOCAL_MAX] = {0};   // static_assert at namespace scope proves
                                        // elems_per_thread <= O_LOCAL_MAX (head_dim
                                        // <= 8*BLOCK = 1024; MAX_HEAD_DIM caps at 256).

    __shared__ float s_sum[BLOCK / WARP];
    __shared__ float s_scalar[1];

    // Handle degenerate tiles (no work — emit neutral partials).
    if (t_begin >= t_end) {
        if (tid == 0) {
            float* out_base = partials + ((size_t)h * num_kv_tiles + tile) * (head_dim + 2);
            out_base[head_dim]     = NEG_SENTINEL;   // partial_m
            out_base[head_dim + 1] = 0.0f;        // partial_l
        }
        for (int d = tid; d < head_dim; d += BLOCK) {
            float* out_base = partials + ((size_t)h * num_kv_tiles + tile) * (head_dim + 2);
            out_base[d] = 0.0f;
        }
        return;
    }

    for (int t = t_begin; t < t_end; ++t) {
        const __half* K_row = K + ((size_t)t * num_kv_heads + kv_head) * head_dim;

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
        const __half* V_row = V + ((size_t)t * num_kv_heads + kv_head) * head_dim;
        for (int ei = 0; ei < elems_per_thread; ++ei) {
            int d = tid + ei * BLOCK;
            if (d < head_dim) {
                o_local[ei] += beta * (float)V_row[d];
            }
        }
        m = m_new;
    }

    // Emit *unnormalized* partials — pass 2 does the final normalize.
    float* out_base = partials + ((size_t)h * num_kv_tiles + tile) * (head_dim + 2);
    for (int ei = 0; ei < elems_per_thread; ++ei) {
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

// Pass 2: one block per q_head. Each block walks its num_kv_tiles partial
// triples (o_tile, m_tile, l_tile) and combines them with the same numerically
// stable online-softmax recipe, then writes a single fp16 output vector.
//
// Small (num_kv_tiles <= BLOCK typically), so a single-warp-or-two
// loop-with-__syncthreads on the running (m, l) is fine. We use a "broadcast
// the tile's m/l via shared memory" pattern — same as pass 1's S[t] broadcast.
__global__ __launch_bounds__(BLOCK)
void kv_cache_attn_reduce_kernel(
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
    float o_local[O_LOCAL_MAX] = {0};

    __shared__ float s_ml[2];  // [tile_m, tile_l] broadcast slot

    for (int tile = 0; tile < num_kv_tiles; ++tile) {
        const float* in_base =
            partials + ((size_t)h * num_kv_tiles + tile) * (head_dim + 2);

        if (tid == 0) {
            s_ml[0] = in_base[head_dim];       // tile_m
            s_ml[1] = in_base[head_dim + 1];   // tile_l
        }
        __syncthreads();
        const float tile_m = s_ml[0];
        const float tile_l = s_ml[1];
        __syncthreads();   // pair with the one above before next iter's write

        // Empty tile → skip (partial_l is 0, partial_m is -INF).
        if (tile_l == 0.0f) continue;

        const float m_new = (tile_m > m) ? tile_m : m;
        const float alpha = std::expf(m - m_new);
        const float beta  = std::expf(tile_m - m_new);
        l = l * alpha + beta * tile_l;

        #pragma unroll
        for (int k = 0; k < O_LOCAL_MAX; ++k) o_local[k] *= alpha;

        for (int ei = 0; ei < elems_per_thread; ++ei) {
            int d = tid + ei * BLOCK;
            if (d < head_dim) {
                o_local[ei] += beta * in_base[d];
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

// ---- Cached intermediate buffer -------------------------------------------
// Growing-only scratch; freed at process exit via hipFree-on-destruct atexit
// would need hipDeviceReset coordination — leave it leaking, OS reclaims.
struct PartialsCache {
    std::mutex  mu;
    float*      ptr   = nullptr;
    size_t      cap_f = 0;   // capacity in floats

    // Returns a device pointer with at least need_f floats of capacity.
    float* acquire(size_t need_f) {
        std::lock_guard<std::mutex> g(mu);
        if (need_f <= cap_f) return ptr;
        if (ptr) {
            // Best-effort free. Growing on every decode step for a monotonic
            // seq_len is the normal path; a failing hipFree here would leak
            // device memory, but we can't meaningfully recover — log and move on.
            hipError_t fst = hipFree(ptr);
            (void)fst;  // [[nodiscard]] on hipError_t; ignored by design
        }
        // Grow with a 1.5x factor to avoid thrashing as seq_len grows
        // monotonically during a decode stream.
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

PartialsCache& partials_cache() {
    static PartialsCache c;
    return c;
}

}  // namespace

extern "C" rcpp_status_t
rcpp_kv_cache_attn_decode_fd(const void* Q_dev, const void* K_dev, const void* V_dev,
                             void* out_dev,
                             int num_q_heads, int num_kv_heads, int head_dim,
                             int seq_len, float scale, void* stream)
{
    if (!Q_dev || !K_dev || !V_dev || !out_dev)          return RCPP_INVALID_ARG;
    if (num_q_heads <= 0 || num_kv_heads <= 0)           return RCPP_INVALID_ARG;
    if (num_q_heads % num_kv_heads != 0)                 return RCPP_INVALID_ARG;
    if (head_dim <= 0 || head_dim > MAX_HEAD_DIM)        return RCPP_INVALID_ARG;
    if (seq_len <= 0)                                    return RCPP_INVALID_ARG;

    const int num_kv_tiles = (seq_len + TILE - 1) / TILE;

    const size_t need_f =
        (size_t)num_q_heads * (size_t)num_kv_tiles * (size_t)(head_dim + 2);
    float* partials = partials_cache().acquire(need_f);
    if (!partials) return RCPP_HIP_ERROR;

    // Pass 1 — fill partials.
    dim3 grid1(num_q_heads, num_kv_tiles, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(kv_cache_attn_decode_split_kernel,
                       grid1, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(Q_dev),
                       static_cast<const __half*>(K_dev),
                       static_cast<const __half*>(V_dev),
                       partials,
                       num_q_heads, num_kv_heads, head_dim,
                       seq_len, num_kv_tiles, scale);

    // Pass 2 — reduce tiles into final fp16 out.
    dim3 grid2(num_q_heads, 1, 1);
    hipLaunchKernelGGL(kv_cache_attn_reduce_kernel,
                       grid2, block, 0, (hipStream_t)stream,
                       partials,
                       static_cast<__half*>(out_dev),
                       num_q_heads, head_dim, num_kv_tiles);

    return RCPP_OK;
}
