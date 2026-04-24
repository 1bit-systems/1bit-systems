// Split-KV Flash-Decoding with inline PlanarQuant-3 dequant.
//
// Copy-and-diverge from src/kv_cache_attn_fd.hip. The non-rotor path there
// must stay bit-exact; do NOT edit that file.
//
// Differences from the fp16 FD kernel:
//   * K/V are packed uint8 [seq_len, num_kv_heads, head_dim * 3 / 8] (3 bits
//     per scalar, 8 scalars per 3 bytes).
//   * Inner per-position loop unpacks 8 indices, LUT-dequants, THEN applies
//     the inverse Givens rotation (un-rotates the 2-vector per dim pair) to
//     recover the stored fp16 KV scalar. This is the *reverse* of what the
//     requantize kernel did at write time.
//   * No per-token scale. The LUT is scaled by 1/sqrt(head_dim) once per tile
//     into a register file; everything downstream runs in fp32.
//
// Grid shape (identical to non-rotor FD):
//   Pass 1: (num_q_heads, num_kv_tiles)       — BLOCK threads
//   Pass 2: (num_q_heads)                     — BLOCK threads, single reduce
//
// All angles are recomputed on-device from (layer_idx, pair_idx) via the same
// splitmix32 the requantize kernel uses. The caller guarantees layer_idx on
// read matches the one on write, or the math is wrong (deterministic garbage,
// not UB).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cfloat>
#include <cstdint>
#include <cmath>
#include <mutex>

#include "rocm_cpp/ck_gemm.h"
#include "rocm_cpp/kv_rotorquant.h"

namespace {

constexpr int BLOCK        = 128;
constexpr int WARP         = 32;
constexpr int MAX_HEAD_DIM = 256;
constexpr int TILE         = 128;
constexpr int O_LOCAL_MAX  = 8;     // fixed per-thread accumulator capacity

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
constexpr float NEG_SENTINEL = -FLT_MAX;

// splitmix32 — MUST match the generator in kernels/rotorquant_pack.hip byte
// for byte. If these drift, the inverse rotation decodes to junk.
__device__ inline uint32_t splitmix32_dev(uint32_t x) {
    x += 0x9E3779B9u;
    x = (x ^ (x >> 16)) * 0x85EBCA6Bu;
    x = (x ^ (x >> 13)) * 0xC2B2AE35u;
    x =  x ^ (x >> 16);
    return x;
}

__device__ inline float pq3_pair_angle_dev(int layer_idx, int pair_idx) {
    uint32_t h = splitmix32_dev((uint32_t)layer_idx * 0x9E3779B9u
                                + (uint32_t)pair_idx);
    const float inv = 1.0f / 4294967296.0f;
    return (float)h * inv * 6.2831853071795864769f;
}

// Dequant one scalar: 3-bit index -> scaled LUT value.
// lut_scaled already contains RCPP_PQ3_LUT[k] * (1/sqrt(head_dim)).
__device__ inline float dequant_idx(uint32_t idx, const float* lut_scaled) {
    return lut_scaled[idx & 7u];
}

// Fetch 8 indices from a packed (3-byte) triple. We read the triple via two
// byte loads + one shift assemble; unaligned u32 loads into global would be
// tempting but head_dim*3/8 isn't u32-aligned in general.
__device__ inline uint32_t load_packed_triple(const uint8_t* p) {
    // Three adjacent bytes -> 24 bits in a u32 (upper 8 bits zero).
    const uint32_t b0 = (uint32_t)p[0];
    const uint32_t b1 = (uint32_t)p[1];
    const uint32_t b2 = (uint32_t)p[2];
    return b0 | (b1 << 8) | (b2 << 16);
}

// Unpack 8 indices from 24 bits: idx[k] = (packed >> (3*k)) & 7.
// Used inline by the kernel below.

// -- Pass 1 kernel -----------------------------------------------------------
// One block per (q_head, kv_tile). Same online-softmax shape as the fp16 FD
// kernel; only the per-position K/V read changes.
__global__ __launch_bounds__(BLOCK)
void kv_cache_attn_decode_split_pq3_kernel(
    const __half*  __restrict__ Q,
    const uint8_t* __restrict__ K_idx,          // packed 3-bit
    const uint8_t* __restrict__ V_idx,
    float*         __restrict__ partials,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, int num_kv_tiles, int layer_idx, float scale)
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

    // -- Prep Q in shared (fp16) --------------------------------------------
    __shared__ __half Q_shared[MAX_HEAD_DIM];
    for (int d = tid; d < head_dim; d += BLOCK) {
        Q_shared[d] = Q[(size_t)h * head_dim + d];
    }

    // -- Prep scaled LUT + per-pair rotation tables in shared ---------------
    // lut_scaled[8]                 : RCPP_PQ3_LUT * (1/sqrt(head_dim))
    // pair_cs[head_dim/2][2]        : (cos, sin) per dim pair
    //
    // These are constant for the life of the kernel and tile-uniform, so we
    // precompute once under __syncthreads and all lanes read broadcast.
    __shared__ float s_lut_scaled[8];
    __shared__ float s_pair_cs[MAX_HEAD_DIM / 2][2];

    if (tid < 8) {
        // Device-local LUT copy — keep in sync with RCPP_PQ3_LUT in
        // src/kv_rotorquant.cpp. HIP forbids reading a __host__ const from
        // a __global__, so we inline the values.
        const float LUT_H[8] = {
            -2.1519f, -1.3439f, -0.7560f, -0.2451f,
             0.2451f,  0.7560f,  1.3439f,  2.1519f,
        };
        const float inv_sd = 1.0f / sqrtf((float)head_dim);
        s_lut_scaled[tid] = LUT_H[tid] * inv_sd;
    }
    const int pairs = head_dim >> 1;
    for (int i = tid; i < pairs; i += BLOCK) {
        const float theta = pq3_pair_angle_dev(layer_idx, i);
        float s, c;
        __sincosf(theta, &s, &c);
        s_pair_cs[i][0] = c;
        s_pair_cs[i][1] = s;
    }
    __syncthreads();

    float m = NEG_SENTINEL;
    float l = 0.0f;
    const int elems_per_thread = (head_dim + BLOCK - 1) / BLOCK;
    float o_local[O_LOCAL_MAX] = {0};

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

    const size_t row_bytes = (size_t)(head_dim * 3 / 8);

    // Helper: read one fp32 dequanted+unrotated scalar at dimension `d` from
    // the packed row at `row`. Structured as two helpers (for K and V) to
    // keep the hot path inlined.
    //
    // NOTE: unpack is per-scalar (one idx). For a full pair (d even) we also
    // want idx[d+1]. Callers fetch both and unrotate the pair together.

    // Online softmax over the tile's positions.
    for (int t = t_begin; t < t_end; ++t) {
        const uint8_t* K_row = K_idx + (size_t)(t * num_kv_heads + kv_head) * row_bytes;

        // ---- Q . dequant(K_row) ------------------------------------------
        float partial = 0.0f;
        // We iterate on dim pairs (aligned with rotation pairs). Each thread
        // handles pairs at stride BLOCK/2 in pair-space (covers head_dim/2
        // pairs with head_dim-size stride BLOCK in dim-space, same coalescing
        // footprint as the fp16 kernel's d-stride).
        for (int p = tid; p < pairs; p += BLOCK) {
            const int d0 = p * 2;
            const int d1 = d0 + 1;

            // Figure out which packed byte-triple holds (d0, d1). A triple
            // covers 8 dims so (d0, d1) are ALWAYS in the same triple when
            // d0 is even — which it is by construction.
            const int g       = d0 >> 3;                  // triple index
            const int in_trip = d0 & 7;                   // 0, 2, 4, or 6
            const uint32_t triple = load_packed_triple(K_row + g * 3);
            const uint32_t idx0 = (triple >> (in_trip * 3))     & 7u;
            const uint32_t idx1 = (triple >> ((in_trip + 1) * 3)) & 7u;
            const float k0q = dequant_idx(idx0, s_lut_scaled);   // rotated
            const float k1q = dequant_idx(idx1, s_lut_scaled);

            // Inverse Givens: stored (k0q, k1q) = R(theta) * (k0, k1).
            // Recover (k0, k1) = R(-theta) * (k0q, k1q) = (c*k0q + s*k1q, -s*k0q + c*k1q).
            const float c = s_pair_cs[p][0];
            const float s = s_pair_cs[p][1];
            const float k0 =  c * k0q + s * k1q;
            const float k1 = -s * k0q + c * k1q;

            partial += (float)Q_shared[d0] * k0
                     + (float)Q_shared[d1] * k1;
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
        const float s_score = s_scalar[0];

        const float m_new = (s_score > m) ? s_score : m;
        const float alpha = expf(m - m_new);
        const float beta  = expf(s_score - m_new);
        l = l * alpha + beta;
        #pragma unroll
        for (int k = 0; k < O_LOCAL_MAX; ++k) o_local[k] *= alpha;

        // ---- beta * dequant(V_row) accumulated into o_local --------------
        // ONE lane per pair does triple-load + dequant; the odd partner pulls
        // the dequant results via intra-wave shfl. Saves ~2× vgpr + ALU on the
        // V-loop (was: both d0 and d1 lanes did the full dequant+unrotate then
        // dropped half via a ternary). Pairs always span adjacent lanes within
        // a wave on wave32 for head_dim ≤ 128, so shfl stays intra-wave.
        const uint8_t* V_row = V_idx + (size_t)(t * num_kv_heads + kv_head) * row_bytes;
        for (int ei = 0; ei < elems_per_thread; ++ei) {
            const int d = tid + ei * BLOCK;
            if (d >= head_dim) continue;
            const int p       = d >> 1;
            const int d0      = p * 2;
            const int g       = d0 >> 3;
            const int in_trip = d0 & 7;

            float v0q = 0.0f, v1q = 0.0f;
            if ((d & 1) == 0) {
                const uint32_t triple = load_packed_triple(V_row + g * 3);
                const uint32_t idx0   = (triple >> (in_trip * 3))       & 7u;
                const uint32_t idx1   = (triple >> ((in_trip + 1) * 3)) & 7u;
                v0q = dequant_idx(idx0, s_lut_scaled);
                v1q = dequant_idx(idx1, s_lut_scaled);
            }
            // Both d0 and d1 lanes read from the even (d0) lane.
            const int partner = (tid & 31) & ~1;
            v0q = __shfl(v0q, partner);
            v1q = __shfl(v1q, partner);

            const float c  = s_pair_cs[p][0];
            const float sn = s_pair_cs[p][1];
            const float v_scalar = (d == d0)
                ?  c  * v0q + sn * v1q
                : -sn * v0q + c  * v1q;
            o_local[ei] += beta * v_scalar;
        }

        m = m_new;
    }

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

// -- Pass 2 kernel (identical to fp16 variant; duplicated to avoid a cross-TU
// include from the non-rotor file, since it's file-scoped `static`/anon).
__global__ __launch_bounds__(BLOCK)
void kv_cache_attn_reduce_pq3_kernel(
    const float* __restrict__ partials,
    __half*      __restrict__ out,
    int num_q_heads, int head_dim, int num_kv_tiles)
{
    const int h = blockIdx.x;
    if (h >= num_q_heads) return;

    const int tid = threadIdx.x;

    float m = NEG_SENTINEL;
    float l = 0.0f;
    const int elems_per_thread = (head_dim + BLOCK - 1) / BLOCK;
    float o_local[O_LOCAL_MAX] = {0};

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
        const float alpha = expf(m - m_new);
        const float beta  = expf(tile_m - m_new);
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

// Growing-only scratch buffer for the pass-1 partials. Duplicated here (not
// shared with the fp16 FD cache) so the two paths can run interleaved without
// cross-contamination.
struct PartialsCachePQ3 {
    std::mutex mu;
    float*     ptr   = nullptr;
    size_t     cap_f = 0;

    float* acquire(size_t need_f) {
        std::lock_guard<std::mutex> g(mu);
        if (need_f <= cap_f) return ptr;
        if (ptr) {
            hipError_t fst = hipFree(ptr);
            (void)fst;  // [[nodiscard]] on hipError_t; ignored by design
        }
        size_t new_cap = (cap_f == 0) ? need_f : cap_f;
        while (new_cap < need_f) new_cap = new_cap + (new_cap >> 1) + 1;
        void* p = nullptr;
        if (hipMalloc(&p, new_cap * sizeof(float)) != hipSuccess) {
            ptr = nullptr; cap_f = 0; return nullptr;
        }
        ptr = static_cast<float*>(p);
        cap_f = new_cap;
        return ptr;
    }
};

PartialsCachePQ3& partials_cache_pq3() {
    static PartialsCachePQ3 c;
    return c;
}

}  // namespace

extern "C" int rcpp_pq3_fd_decode_launch(
    const void* Q_dev, const void* K_idx_dev, const void* V_idx_dev,
    void* out_dev,
    int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, int layer_idx, float scale, void* stream)
{
    if (!Q_dev || !K_idx_dev || !V_idx_dev || !out_dev) return 1;
    if (num_q_heads <= 0 || num_kv_heads <= 0)          return 1;
    if (num_q_heads % num_kv_heads != 0)                return 1;
    if (head_dim <= 0 || head_dim > MAX_HEAD_DIM)       return 1;
    if ((head_dim & 7) != 0)                            return 1;   // need %8
    if (seq_len <= 0)                                   return 1;

    const int num_kv_tiles = (seq_len + TILE - 1) / TILE;
    const size_t need_f =
        (size_t)num_q_heads * (size_t)num_kv_tiles * (size_t)(head_dim + 2);
    float* partials = partials_cache_pq3().acquire(need_f);
    if (!partials) return 2;

    dim3 grid1(num_q_heads, num_kv_tiles, 1), block(BLOCK, 1, 1);
    hipLaunchKernelGGL(kv_cache_attn_decode_split_pq3_kernel,
                       grid1, block, 0, (hipStream_t)stream,
                       static_cast<const __half*>(Q_dev),
                       static_cast<const uint8_t*>(K_idx_dev),
                       static_cast<const uint8_t*>(V_idx_dev),
                       partials,
                       num_q_heads, num_kv_heads, head_dim,
                       seq_len, num_kv_tiles, layer_idx, scale);

    dim3 grid2(num_q_heads, 1, 1);
    hipLaunchKernelGGL(kv_cache_attn_reduce_pq3_kernel,
                       grid2, block, 0, (hipStream_t)stream,
                       partials,
                       static_cast<__half*>(out_dev),
                       num_q_heads, head_dim, num_kv_tiles);

    return 0;
}
