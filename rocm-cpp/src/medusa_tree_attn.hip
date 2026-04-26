// medusa_tree_attn.hip — tree-parallel split-KV Flash-Decoding (BODY).
//
// See include/rocm_cpp/medusa.h for the public contract. This file implements
// the Medusa-1 tree-attention kernel as a generalization of the M=1 split-KV
// Flash-Decoding kernel in src/kv_cache_attn_fd.hip.
//
// Design choice: one block per (q_head, kv_tile, m)
// -------------------------------------------------
// The M=1 baseline launches grid (num_q_heads, num_kv_tiles); we add a third
// dim for the candidate index m and run the *same* per-block work that the
// M=1 path runs. This keeps the M=1 launch path bit-exact vs
// rcpp_kv_cache_attn_decode_fd (same reduction order, same shfl pattern,
// same partials layout once you fix m=0).
//
// Trade-off: K is re-read M times (once per candidate). At Medusa-1's
// typical tree_size ≤ 16 and gfx1151's L2 (8 MB LLC + iGPU shares system
// LPDDR5 cache hierarchy) we expect the speculative-tail K traffic to live
// in cache; the committed-prefix K is the bandwidth bulk and L2-resident
// across (m, q_head) neighbors in the grid. A bandwidth-fused variant
// (one block per (q_head, kv_tile) with M outputs in registers) can land
// later — at M=64 head_dim=256 it would require >32 VGPRs/thread per
// candidate's o_local + (m, l) state, pushing register pressure past the
// 256-VGPR budget. v1 ships correctness-first.
//
// Tree mask semantics
// -------------------
//   K_cache occupies positions [0, seq_len + tree_size) in the cache buffer.
//   Committed prefix [0, seq_len)         — every candidate attends, no mask.
//   Speculative tail [seq_len, seq_len+tree_size)
//                                          — candidate m attends to position
//                                            (seq_len + j) iff
//                                            tree_mask[m, j] != 0.
//
//   In the M=1 case the launcher receives tree_mask=nullptr and tree_size=1,
//   AND the cache layout collapses: there is no speculative tail (no extra
//   K rows beyond the committed prefix), so no mask check ever fires. That
//   path is bit-exact vs the M=1 baseline.
//
// gfx1151 / wave32 first-class. No hipBLAS, no Python, no
// hipDeviceSynchronize.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cfloat>
#include <cstdint>
#include <cmath>
#include <mutex>

#include "rocm_cpp/medusa.h"

namespace {

constexpr int BLOCK         = 128;
constexpr int WARP          = 32;
constexpr int MAX_HEAD_DIM  = 256;
constexpr int TILE          = 128;   // KV-tile size, matches kv_cache_attn_fd.
constexpr int MAX_TREE_SIZE = RCPP_MEDUSA_MAX_TREE_SIZE;   // 64

// Same fast-math sentinel reasoning as kv_cache_attn_fd.hip. -FLT_MAX is
// used because -INFINITY under -ffast-math triggers nan-infinity-disabled
// UB; the pass-2 reduce gates on `tile_l == 0.0f` so the sentinel never
// participates in a real arithmetic op.
constexpr float NEG_SENTINEL = -FLT_MAX;

// ── Pass 1 — one block per (q_head, kv_tile, candidate) ───────────────────
// Block writes a partial (output_acc[head_dim], partial_m, partial_l) triple
// into a [tree_size, num_q_heads, num_kv_tiles, head_dim+2] f32 scratch.
__global__ __launch_bounds__(BLOCK)
void medusa_tree_attn_decode_split_kernel(
    const __half*  __restrict__ Q,           // [tree_size, num_q_heads, head_dim]
    const __half*  __restrict__ K,           // [total_kv_len, num_kv_heads, head_dim]
    const __half*  __restrict__ V,           // [total_kv_len, num_kv_heads, head_dim]
    const uint8_t* __restrict__ tree_mask,   // [tree_size, tree_size]  (or nullptr)
    float*         __restrict__ partials,    // [tree_size, num_q_heads, num_kv_tiles, head_dim+2]
    int tree_size, int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len_committed, int total_kv_len, int num_kv_tiles, float scale)
{
    const int h    = blockIdx.x;
    const int tile = blockIdx.y;
    const int m    = blockIdx.z;
    if (h >= num_q_heads || tile >= num_kv_tiles || m >= tree_size) return;

    const int tid       = threadIdx.x;
    const int lane      = tid & (WARP - 1);
    const int wid       = tid / WARP;
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head   = h / gqa_ratio;

    const int t_begin = tile * TILE;
    const int t_end   = min(t_begin + TILE, total_kv_len);

    // ── LDS staging ───────────────────────────────────────────────────────
    // Q_shared : this candidate's Q vector for this q_head (HD halves).
    // mask_row : this candidate's row of the tree mask (tree_size bytes,
    //            or absent in the M=1 case where tree_mask==nullptr).
    __shared__ __half  Q_shared[MAX_HEAD_DIM];
    __shared__ uint8_t mask_row[MAX_TREE_SIZE];
    __shared__ float   s_sum[BLOCK / WARP];
    __shared__ float   s_scalar[1];

    // Load Q[m, h, :] cooperatively. Q layout: [tree_size, num_q_heads,
    // head_dim] row-major. Stride = num_q_heads * head_dim per candidate.
    {
        const size_t q_base =
            (size_t)m * (size_t)num_q_heads * (size_t)head_dim
            + (size_t)h * (size_t)head_dim;
        for (int d = tid; d < head_dim; d += BLOCK) {
            Q_shared[d] = Q[q_base + (size_t)d];
        }
    }

    // Load mask_row = tree_mask[m, :]. Skipped when no mask (M=1 path).
    // tree_size <= MAX_TREE_SIZE = 64, so a single coalesced wave-load
    // covers it; we still loop in case tree_size > BLOCK in some odd
    // future config.
    if (tree_mask) {
        const size_t mask_base = (size_t)m * (size_t)tree_size;
        for (int j = tid; j < tree_size; j += BLOCK) {
            mask_row[j] = tree_mask[mask_base + (size_t)j];
        }
    }
    __syncthreads();

    // Output partials base for THIS (m, h, tile).
    // Scratch layout: [m][h][tile][hd+2] flattened.
    float* const out_base =
        partials
        + ((size_t)m * (size_t)num_q_heads + (size_t)h) * (size_t)num_kv_tiles
              * (size_t)(head_dim + 2)
        + (size_t)tile * (size_t)(head_dim + 2);

    // Degenerate empty tile — emit neutral partials and bail. Same shape
    // as the kv_cache_attn_fd guard.
    if (t_begin >= t_end) {
        if (tid == 0) {
            out_base[head_dim]     = NEG_SENTINEL;
            out_base[head_dim + 1] = 0.0f;
        }
        for (int d = tid; d < head_dim; d += BLOCK) {
            out_base[d] = 0.0f;
        }
        return;
    }

    float m_acc = NEG_SENTINEL;
    float l_acc = 0.0f;
    const int elems_per_thread = (head_dim + BLOCK - 1) / BLOCK;
    float o_local[8] = {0};   // head_dim <= 8*BLOCK = 1024; capped at 256.

    for (int t = t_begin; t < t_end; ++t) {
        // ── Early mask check — skip masked positions whole-tile-wide ──
        //
        // Lifted ABOVE the q·k reduction so masked positions cost zero
        // (no dot, no V load, no online-softmax update). The branch is
        // uniform across the block — `t` is loop-invariant per iteration
        // and mask_row[] is in LDS, identical for every thread — so the
        // `continue` skips every thread together and the inner
        // __syncthreads() pair below stays on uniform control flow.
        if (tree_mask && t >= seq_len_committed) {
            const int j = t - seq_len_committed;
            // j < tree_size by construction (t < total_kv_len = seq_len + tree_size).
            if (mask_row[j] == 0u) continue;
        }

        // ── q · k partial, lane-reduced, then wave-reduced ───────────────
        //
        // Identical reduction shape to kv_cache_attn_decode_split_kernel
        // — same shfl_xor tree, same s_sum staging, same scalar broadcast.
        // Same accumulation order means M=1 launch is bit-exact vs the
        // baseline at the partial level.
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

        // Online-softmax update — bit-identical to the M=1 baseline. By
        // construction we only reach here for unmasked positions (the
        // mask gate at the top of the loop handled the rest), so this
        // is the same arithmetic the single-Q FD kernel performs.
        const float m_new = (s > m_acc) ? s : m_acc;
        const float alpha = std::expf(m_acc - m_new);
        const float beta  = std::expf(s     - m_new);
        l_acc = l_acc * alpha + beta;
        #pragma unroll
        for (int k = 0; k < 8; ++k) o_local[k] *= alpha;

        const __half* V_row = V + ((size_t)t * num_kv_heads + kv_head) * head_dim;
        for (int ei = 0; ei < elems_per_thread; ++ei) {
            int d = tid + ei * BLOCK;
            if (d < head_dim) {
                o_local[ei] += beta * (float)V_row[d];
            }
        }
        m_acc = m_new;
    }

    // Emit unnormalized partials (pass 2 normalizes).
    for (int ei = 0; ei < elems_per_thread; ++ei) {
        int d = tid + ei * BLOCK;
        if (d < head_dim) {
            out_base[d] = o_local[ei];
        }
    }
    if (tid == 0) {
        out_base[head_dim]     = m_acc;
        out_base[head_dim + 1] = l_acc;
    }
}

// ── Pass 2 — one block per (q_head, candidate) ───────────────────────────
// Combines this (m, h)'s num_kv_tiles partial triples into the final fp16
// out vector at [m, h, :]. Same recipe as kv_cache_attn_reduce_kernel.
__global__ __launch_bounds__(BLOCK)
void medusa_tree_attn_reduce_kernel(
    const float* __restrict__ partials,
    __half*      __restrict__ out,
    int tree_size, int num_q_heads, int head_dim, int num_kv_tiles)
{
    const int h = blockIdx.x;
    const int m = blockIdx.y;
    if (h >= num_q_heads || m >= tree_size) return;

    const int tid = threadIdx.x;

    float m_acc = NEG_SENTINEL;
    float l_acc = 0.0f;
    const int elems_per_thread = (head_dim + BLOCK - 1) / BLOCK;
    float o_local[8] = {0};

    __shared__ float s_ml[2];

    const float* const partials_mh =
        partials
        + ((size_t)m * (size_t)num_q_heads + (size_t)h) * (size_t)num_kv_tiles
              * (size_t)(head_dim + 2);

    for (int tile = 0; tile < num_kv_tiles; ++tile) {
        const float* in_base =
            partials_mh + (size_t)tile * (size_t)(head_dim + 2);

        if (tid == 0) {
            s_ml[0] = in_base[head_dim];
            s_ml[1] = in_base[head_dim + 1];
        }
        __syncthreads();
        const float tile_m = s_ml[0];
        const float tile_l = s_ml[1];
        __syncthreads();

        // Empty tile or fully-masked tile both report l == 0; skip.
        if (tile_l == 0.0f) continue;

        const float m_new = (tile_m > m_acc) ? tile_m : m_acc;
        const float alpha = std::expf(m_acc  - m_new);
        const float beta  = std::expf(tile_m - m_new);
        l_acc = l_acc * alpha + beta * tile_l;

        #pragma unroll
        for (int k = 0; k < 8; ++k) o_local[k] *= alpha;

        for (int ei = 0; ei < elems_per_thread; ++ei) {
            int d = tid + ei * BLOCK;
            if (d < head_dim) {
                o_local[ei] += beta * in_base[d];
            }
        }
        m_acc = m_new;
    }

    // Output layout: [tree_size, num_q_heads, head_dim] row-major.
    // Empty/all-masked candidates (l_acc == 0) write a zero vector — the
    // caller's verify path discards those rows.
    const float inv_l = (l_acc > 0.0f) ? (1.0f / l_acc) : 0.0f;
    __half* out_mh =
        out + ((size_t)m * (size_t)num_q_heads + (size_t)h) * (size_t)head_dim;
    for (int ei = 0; ei < elems_per_thread; ++ei) {
        int d = tid + ei * BLOCK;
        if (d < head_dim) {
            out_mh[d] = __float2half(o_local[ei] * inv_l);
        }
    }
}

// ── Cached partials buffer (lazy-grow, mirror of kv_cache_attn_fd.hip) ──
// Independent allocation from the M=1 path so the two kernels can be in
// flight on different streams without contention. Same growing-only
// strategy: monotonic seq_len during a decode stream means amortized O(1)
// reallocs.
struct MedusaPartialsCache {
    std::mutex mu;
    float*     ptr   = nullptr;
    size_t     cap_f = 0;

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
            ptr = nullptr; cap_f = 0;
            return nullptr;
        }
        ptr = static_cast<float*>(p);
        cap_f = new_cap;
        return ptr;
    }
};

MedusaPartialsCache& medusa_partials_cache() {
    static MedusaPartialsCache c;
    return c;
}

}  // namespace

extern "C"
rcpp_status_t
rcpp_medusa_tree_attn_decode_fd(const void* Q_dev,
                                const void* K_dev,
                                const void* V_dev,
                                const void* tree_mask_dev,   // may be nullptr when tree_size == 1
                                void*       out_dev,
                                int         tree_size,
                                int         num_q_heads,
                                int         num_kv_heads,
                                int         head_dim,
                                int         seq_len,         // committed prefix length
                                float       scale,
                                void*       stream)
{
    if (!Q_dev || !K_dev || !V_dev || !out_dev)              return RCPP_INVALID_ARG;
    if (tree_size <= 0 || tree_size > MAX_TREE_SIZE)         return RCPP_INVALID_ARG;
    if (num_q_heads <= 0 || num_kv_heads <= 0)               return RCPP_INVALID_ARG;
    if (num_q_heads % num_kv_heads != 0)                     return RCPP_INVALID_ARG;
    if (head_dim <= 0 || head_dim > MAX_HEAD_DIM)            return RCPP_INVALID_ARG;
    if (seq_len < 0)                                         return RCPP_INVALID_ARG;
    // The tree_mask pointer is only required when tree_size > 1; the M=1
    // path collapses to "no speculative tail, no mask" and bit-matches
    // rcpp_kv_cache_attn_decode_fd. tree_size > 1 with a null mask would
    // be an under-specified call — reject early.
    if (tree_size > 1 && !tree_mask_dev)                     return RCPP_INVALID_ARG;

    // Total KV length = committed prefix + speculative candidates. For
    // tree_size == 1 the speculative tail is empty (the single "candidate"
    // is the previous-step's committed token, already in cache at index
    // seq_len-1 — no new K row needed). For tree_size > 1 the caller
    // staged tree_size additional K rows at positions [seq_len, seq_len+tree_size).
    const int total_kv_len = (tree_size == 1) ? seq_len : (seq_len + tree_size);
    if (total_kv_len <= 0)                                   return RCPP_INVALID_ARG;

    const int num_kv_tiles = (total_kv_len + TILE - 1) / TILE;

    const size_t need_f =
        (size_t)tree_size * (size_t)num_q_heads
        * (size_t)num_kv_tiles * (size_t)(head_dim + 2);
    float* partials = medusa_partials_cache().acquire(need_f);
    if (!partials) return RCPP_HIP_ERROR;

    // ── Pass 1: fill [tree_size, num_q_heads, num_kv_tiles] partials ──
    // Grid laid out (q_head, kv_tile, m): keeps the (q_head, kv_tile)
    // plane contiguous so adjacent blocks share K rows in L2 (gqa_ratio
    // groups several q_heads onto one kv_head, and consecutive kv_tiles
    // step the K stream linearly).
    dim3 grid1(num_q_heads, num_kv_tiles, tree_size);
    dim3 block(BLOCK, 1, 1);
    hipLaunchKernelGGL(medusa_tree_attn_decode_split_kernel,
                       grid1, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(Q_dev),
                       static_cast<const __half*>(K_dev),
                       static_cast<const __half*>(V_dev),
                       static_cast<const uint8_t*>(tree_mask_dev),
                       partials,
                       tree_size, num_q_heads, num_kv_heads, head_dim,
                       seq_len, total_kv_len, num_kv_tiles, scale);

    // ── Pass 2: combine per-(m, h) tiles into final fp16 out ──
    dim3 grid2(num_q_heads, tree_size, 1);
    hipLaunchKernelGGL(medusa_tree_attn_reduce_kernel,
                       grid2, block, 0, (hipStream_t)stream,
                       partials,
                       static_cast<__half*>(out_dev),
                       tree_size, num_q_heads, head_dim, num_kv_tiles);

    return RCPP_OK;
}
