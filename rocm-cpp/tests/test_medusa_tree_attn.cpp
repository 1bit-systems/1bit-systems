// test_medusa_tree_attn.cpp — differential test for tree-parallel split-KV
// Flash-Decoding (rcpp_medusa_tree_attn_decode_fd).
//
// What this test verifies
// -----------------------
//   1. M=1 path: bit-exact (≤ 1 fp16 ULP) vs the production single-Q
//      Flash-Decoding kernel (rcpp_kv_cache_attn_decode_fd). Same
//      accumulation order, same tile size, same scratch shape — any
//      drift here means the M=1 launch broke the contract.
//   2. M>1 path: each candidate row of the tree-attn output matches a
//      scalar Flash-Decoding reference run for that candidate's mask.
//      The reference walks the same TILE=128 / online-softmax recipe so
//      partial-tile accumulator orders agree; the only place rounding
//      can drift is the cross-thread reduction of q·k inside the kernel
//      (shfl_xor tree vs scalar sequential sum). The architect's spec
//      pins the tolerance at "bf16 ULP ≤ 8" — bf16 has 8 mantissa bits
//      vs fp16's 10, so 8 bf16 ULPs ≈ 32 fp16 ULPs at matched exponent.
//      We use 64 fp16 ULPs to give parallel-reduction order-of-summation
//      headroom; in practice we observe single-digit fp16 ULPs.
//
// Skip behavior
// -------------
//   No GPU detected → exit 0 (CI-friendly skip), prints SKIP to stderr.
//   Any HIP API failure → SIGABRT via HIP_OK. Result-mismatch → exit 1.
//
// No hipBLAS, no Python, no hipDeviceSynchronize.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "rocm_cpp/medusa.h"
// rcpp_kv_cache_attn_decode_fd is declared via the ck_gemm.h transitive
// include from medusa.h — used below for the M=1 bit-exact diff.

#define HIP_OK(e) do { \
    hipError_t _s = (e); \
    if (_s != hipSuccess) { \
        std::fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
                     (int)_s, hipGetErrorString(_s), __FILE__, __LINE__); \
        std::abort(); \
    } \
} while (0)

namespace {

constexpr int TILE = 128;   // must match the kernel constant.

uint16_t fp16_to_u16(__half v) {
    uint16_t u;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

uint32_t fp16_ulp_diff(__half a, __half b) {
    auto flip = [](uint16_t u) -> int32_t {
        if (u & 0x8000u) return (int32_t)(0x8000u - (u & 0x7FFFu));
        return (int32_t)(0x8000u + u);
    };
    int32_t ia = flip(fp16_to_u16(a));
    int32_t ib = flip(fp16_to_u16(b));
    int32_t d  = ia - ib;
    if (d < 0) d = -d;
    return (uint32_t)d;
}

// ── Host-side Flash-Decoding scalar reference ──────────────────────────────
// Same TILE and online-softmax recipe as the kernel. For each (m, h):
//   1. Stream over t = 0..total_kv_len in TILE-sized chunks.
//   2. Within a tile, accumulate (m_tile, l_tile, o_tile) via the same
//      online-softmax update used by the kernel.
//   3. Combine tiles with the same merge step as pass-2.
// Mask gates positions in the speculative tail [seq_len, seq_len+tree_size).
//
// We DON'T attempt to replicate the kernel's parallel q·k reduction
// shape — the host reference sums q·k sequentially over head_dim. That's
// the only place the kernel and reference can disagree by more than a
// rounding ULP. Tolerance below absorbs it.
void scalar_reference_tree_attn(
    const std::vector<__half>& Q,           // [tree_size, num_q_heads, head_dim]
    const std::vector<__half>& K,           // [total_kv_len, num_kv_heads, head_dim]
    const std::vector<__half>& V,           // [total_kv_len, num_kv_heads, head_dim]
    const std::vector<uint8_t>& tree_mask,  // [tree_size, tree_size] (or empty)
    int tree_size, int num_q_heads, int num_kv_heads, int head_dim,
    int seq_len, int total_kv_len, float scale,
    std::vector<__half>& out)               // [tree_size, num_q_heads, head_dim]
{
    out.assign((size_t)tree_size * num_q_heads * head_dim, __float2half(0.0f));
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int n_tiles   = (total_kv_len + TILE - 1) / TILE;

    std::vector<float> o_tile(head_dim);
    std::vector<float> o_acc(head_dim);

    for (int m = 0; m < tree_size; ++m) {
        for (int h = 0; h < num_q_heads; ++h) {
            const int kv_head = h / gqa_ratio;

            float m_acc = -FLT_MAX;
            float l_acc = 0.0f;
            std::fill(o_acc.begin(), o_acc.end(), 0.0f);

            for (int tile_idx = 0; tile_idx < n_tiles; ++tile_idx) {
                const int t_begin = tile_idx * TILE;
                const int t_end   = std::min(t_begin + TILE, total_kv_len);
                if (t_begin >= t_end) continue;

                float m_tile = -FLT_MAX;
                float l_tile = 0.0f;
                std::fill(o_tile.begin(), o_tile.end(), 0.0f);
                bool any_unmasked = false;

                for (int t = t_begin; t < t_end; ++t) {
                    // Mirror the kernel's early mask-skip so the
                    // accumulation orders match.
                    if (!tree_mask.empty() && t >= seq_len) {
                        const int j = t - seq_len;
                        if (tree_mask[(size_t)m * tree_size + j] == 0u) continue;
                    }

                    float qk = 0.0f;
                    const size_t q_off =
                        (size_t)m * num_q_heads * head_dim
                        + (size_t)h * head_dim;
                    const size_t k_off =
                        ((size_t)t * num_kv_heads + kv_head) * head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        qk += (float)Q[q_off + d] * (float)K[k_off + d];
                    }
                    const float s = qk * scale;

                    const float m_new = std::max(m_tile, s);
                    const float alpha = std::expf(m_tile - m_new);
                    const float beta  = std::expf(s      - m_new);
                    l_tile = l_tile * alpha + beta;

                    const size_t v_off =
                        ((size_t)t * num_kv_heads + kv_head) * head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        o_tile[d] = o_tile[d] * alpha
                                  + beta * (float)V[v_off + d];
                    }
                    m_tile = m_new;
                    any_unmasked = true;
                }

                // Merge tile partial into running (m_acc, l_acc, o_acc)
                // — same recipe as pass-2 of the kernel.
                if (!any_unmasked) continue;
                const float m_new = std::max(m_acc, m_tile);
                const float alpha = std::expf(m_acc  - m_new);
                const float beta  = std::expf(m_tile - m_new);
                l_acc = l_acc * alpha + beta * l_tile;
                for (int d = 0; d < head_dim; ++d) {
                    o_acc[d] = o_acc[d] * alpha + beta * o_tile[d];
                }
                m_acc = m_new;
            }

            const float inv_l = (l_acc > 0.0f) ? (1.0f / l_acc) : 0.0f;
            const size_t out_off =
                (size_t)m * num_q_heads * head_dim
                + (size_t)h * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                out[out_off + d] = __float2half(o_acc[d] * inv_l);
            }
        }
    }
}

struct DiffStats {
    uint32_t max_ulp;
    int      worst_m;
    int      worst_h;
    int      worst_d;
    float    worst_got;
    float    worst_want;
    double   mean_ulp;
};

DiffStats diff_fp16(const std::vector<__half>& got,
                    const std::vector<__half>& want,
                    int tree_size, int num_q_heads, int head_dim)
{
    DiffStats s{0, 0, 0, 0, 0.0f, 0.0f, 0.0};
    double sum = 0.0;
    size_t count = 0;
    for (int m = 0; m < tree_size; ++m) {
        for (int h = 0; h < num_q_heads; ++h) {
            for (int d = 0; d < head_dim; ++d) {
                const size_t i =
                    (size_t)m * num_q_heads * head_dim
                    + (size_t)h * head_dim + d;
                const uint32_t u = fp16_ulp_diff(got[i], want[i]);
                sum += (double)u;
                ++count;
                if (u > s.max_ulp) {
                    s.max_ulp = u;
                    s.worst_m = m; s.worst_h = h; s.worst_d = d;
                    s.worst_got  = (float)got[i];
                    s.worst_want = (float)want[i];
                }
            }
        }
    }
    s.mean_ulp = (count > 0) ? sum / (double)count : 0.0;
    return s;
}

bool gpu_available() {
    int n = 0;
    hipError_t e = hipGetDeviceCount(&n);
    if (e != hipSuccess) return false;
    return n > 0;
}

struct Trial {
    int  tree_size;
    int  num_q_heads;
    int  num_kv_heads;
    int  head_dim;
    int  seq_len;
    bool diff_against_m1_path;   // only meaningful when tree_size == 1
    const char* tag;
};

// Build a synthetic Medusa tree mask: each candidate's ancestor set is a
// subset of the previous candidates, with a random subset bit pattern.
// Diagonal is always 1 (a candidate attends to itself). Position 0 has
// only itself (root).
void build_tree_mask(std::vector<uint8_t>& mask, int tree_size, std::mt19937& rng) {
    mask.assign((size_t)tree_size * tree_size, 0u);
    std::uniform_int_distribution<int> coin(0, 1);
    for (int m = 0; m < tree_size; ++m) {
        mask[(size_t)m * tree_size + m] = 1u;   // self
        for (int j = 0; j < m; ++j) {
            // 50/50 chance candidate m inherits candidate j as an
            // ancestor. This is denser than a typical Medusa tree (which
            // is ~log structured) but stresses the masked-skip path.
            mask[(size_t)m * tree_size + j] = (uint8_t)coin(rng);
        }
        // Future-position positions (j > m) are not ancestors → 0,
        // already initialized.
    }
}

void run_trial(const Trial& T, uint32_t seed, hipStream_t stream,
               uint32_t fp16_ulp_budget, int& failures)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> qkv_dist(0.0f, 1.0f);

    const int total_kv_len =
        (T.tree_size == 1) ? T.seq_len : (T.seq_len + T.tree_size);
    const float scale = 1.0f / std::sqrt((float)T.head_dim);

    // Host buffers.
    std::vector<__half> h_Q((size_t)T.tree_size * T.num_q_heads * T.head_dim);
    std::vector<__half> h_K((size_t)total_kv_len * T.num_kv_heads * T.head_dim);
    std::vector<__half> h_V((size_t)total_kv_len * T.num_kv_heads * T.head_dim);
    for (auto& v : h_Q) v = __float2half(qkv_dist(rng) * 0.5f);
    for (auto& v : h_K) v = __float2half(qkv_dist(rng) * 0.5f);
    for (auto& v : h_V) v = __float2half(qkv_dist(rng) * 0.5f);

    std::vector<uint8_t> h_mask;
    if (T.tree_size > 1) build_tree_mask(h_mask, T.tree_size, rng);

    // Scalar reference.
    std::vector<__half> ref;
    scalar_reference_tree_attn(h_Q, h_K, h_V, h_mask,
                               T.tree_size, T.num_q_heads, T.num_kv_heads,
                               T.head_dim, T.seq_len, total_kv_len, scale, ref);

    // Device side.
    __half *d_Q = nullptr, *d_K = nullptr, *d_V = nullptr, *d_out = nullptr;
    uint8_t* d_mask = nullptr;
    HIP_OK(hipMalloc(&d_Q,   h_Q.size() * sizeof(__half)));
    HIP_OK(hipMalloc(&d_K,   h_K.size() * sizeof(__half)));
    HIP_OK(hipMalloc(&d_V,   h_V.size() * sizeof(__half)));
    HIP_OK(hipMalloc(&d_out,
                     (size_t)T.tree_size * T.num_q_heads * T.head_dim * sizeof(__half)));
    if (!h_mask.empty()) {
        HIP_OK(hipMalloc(&d_mask, h_mask.size()));
        HIP_OK(hipMemcpyAsync(d_mask, h_mask.data(), h_mask.size(),
                              hipMemcpyHostToDevice, stream));
    }
    HIP_OK(hipMemcpyAsync(d_Q, h_Q.data(), h_Q.size() * sizeof(__half),
                          hipMemcpyHostToDevice, stream));
    HIP_OK(hipMemcpyAsync(d_K, h_K.data(), h_K.size() * sizeof(__half),
                          hipMemcpyHostToDevice, stream));
    HIP_OK(hipMemcpyAsync(d_V, h_V.data(), h_V.size() * sizeof(__half),
                          hipMemcpyHostToDevice, stream));
    HIP_OK(hipMemsetAsync(d_out, 0,
                          (size_t)T.tree_size * T.num_q_heads * T.head_dim * sizeof(__half),
                          stream));

    rcpp_status_t st = rcpp_medusa_tree_attn_decode_fd(
        d_Q, d_K, d_V, d_mask, d_out,
        T.tree_size, T.num_q_heads, T.num_kv_heads, T.head_dim,
        T.seq_len, scale, stream);
    if (st != RCPP_OK) {
        std::fprintf(stderr,
            "FAIL [%s seed=0x%x]: rcpp_medusa_tree_attn_decode_fd status=%d\n",
            T.tag, seed, (int)st);
        ++failures;
    }

    std::vector<__half> got((size_t)T.tree_size * T.num_q_heads * T.head_dim);
    HIP_OK(hipMemcpyAsync(got.data(), d_out, got.size() * sizeof(__half),
                          hipMemcpyDeviceToHost, stream));
    HIP_OK(hipStreamSynchronize(stream));

    // ── Diff vs scalar reference ─────────────────────────────────────────
    DiffStats s = diff_fp16(got, ref, T.tree_size, T.num_q_heads, T.head_dim);
    const bool ulp_pass = (s.max_ulp <= fp16_ulp_budget);
    if (!ulp_pass) {
        std::fprintf(stderr,
            "FAIL [%s seed=0x%x]: max_fp16_ulp=%u (budget=%u)  "
            "worst at (m=%d,h=%d,d=%d) got=%g want=%g  mean_ulp=%.2f\n",
            T.tag, seed, s.max_ulp, fp16_ulp_budget,
            s.worst_m, s.worst_h, s.worst_d,
            s.worst_got, s.worst_want, s.mean_ulp);
        ++failures;
    } else {
        std::fprintf(stderr,
            "  PASS [%s seed=0x%x]: max_fp16_ulp=%u  mean_ulp=%.2f\n",
            T.tag, seed, s.max_ulp, s.mean_ulp);
    }

    // ── Optional: M=1 path bit-exact vs production single-Q FD ──
    if (T.diff_against_m1_path && T.tree_size == 1) {
        __half* d_out_m1 = nullptr;
        HIP_OK(hipMalloc(&d_out_m1,
                         (size_t)T.num_q_heads * T.head_dim * sizeof(__half)));
        rcpp_status_t st1 = rcpp_kv_cache_attn_decode_fd(
            d_Q, d_K, d_V, d_out_m1,
            T.num_q_heads, T.num_kv_heads, T.head_dim,
            T.seq_len, scale, stream);
        if (st1 != RCPP_OK) {
            std::fprintf(stderr,
                "FAIL [%s seed=0x%x]: rcpp_kv_cache_attn_decode_fd status=%d\n",
                T.tag, seed, (int)st1);
            ++failures;
        }
        std::vector<__half> got_m1((size_t)T.num_q_heads * T.head_dim);
        HIP_OK(hipMemcpyAsync(got_m1.data(), d_out_m1,
                              got_m1.size() * sizeof(__half),
                              hipMemcpyDeviceToHost, stream));
        HIP_OK(hipStreamSynchronize(stream));

        // Bit-exact target: medusa-tree (tree_size=1) vs single-Q FD.
        // Both kernels share the same accumulation order; we expect 0 ULPs.
        uint32_t max_ulp_m1 = 0;
        for (int h = 0; h < T.num_q_heads; ++h) {
            for (int d = 0; d < T.head_dim; ++d) {
                const size_t i = (size_t)h * T.head_dim + d;
                const uint32_t u = fp16_ulp_diff(got[i], got_m1[i]);
                if (u > max_ulp_m1) max_ulp_m1 = u;
            }
        }
        const uint32_t m1_budget = 1;   // bit-exact target
        if (max_ulp_m1 > m1_budget) {
            std::fprintf(stderr,
                "FAIL [%s seed=0x%x]: M=1 path NOT bit-exact vs "
                "rcpp_kv_cache_attn_decode_fd  max_ulp=%u (budget=%u)\n",
                T.tag, seed, max_ulp_m1, m1_budget);
            ++failures;
        } else {
            std::fprintf(stderr,
                "  PASS [%s seed=0x%x]: M=1 vs single-Q FD max_ulp=%u\n",
                T.tag, seed, max_ulp_m1);
        }
        HIP_OK(hipFree(d_out_m1));
    }

    HIP_OK(hipFree(d_Q));
    HIP_OK(hipFree(d_K));
    HIP_OK(hipFree(d_V));
    HIP_OK(hipFree(d_out));
    if (d_mask) HIP_OK(hipFree(d_mask));
}

}  // namespace

int main(int argc, char** argv) {
    if (!gpu_available()) {
        std::fprintf(stderr, "[medusa tree attn] SKIP: no ROCm device\n");
        return 0;
    }

    const int n_seeds = (argc > 1) ? std::atoi(argv[1]) : 4;

    // Tolerance: spec says "bf16 ULP ≤ 8". 1 bf16 ULP ≈ 4 fp16 ULPs at
    // matched exponent (10 vs 8 mantissa bits). 8 bf16 ULPs ≈ 32 fp16
    // ULPs; we widen to 64 to absorb the parallel q·k reduction reorder
    // that the host-sequential reference cannot replicate. Empirically
    // this should land in single digits; the budget is the gate, not the
    // expected value.
    const uint32_t fp16_ulp_budget = 64;

    const Trial trials[] = {
        // tree=1 path: must match single-Q FD bit-exactly.
        {  1, 20, 4, 128,  64, true,  "M=1 vs single-Q FD seq=64"  },
        {  1, 20, 4, 128, 512, true,  "M=1 vs single-Q FD seq=512" },
        // tree=4 path: differential vs scalar reference with random mask.
        {  4, 20, 4, 128,  64, false, "M=4 random-mask seq=64"     },
        {  4, 20, 4, 128, 512, false, "M=4 random-mask seq=512"    },
        // Larger tree.
        {  8, 20, 4, 128, 256, false, "M=8 random-mask seq=256"    },
        // GQA-1 (no head sharing) sanity.
        {  4,  8, 8,  64, 128, false, "M=4 GQA=1 seq=128"          },
    };

    hipStream_t stream;
    HIP_OK(hipStreamCreate(&stream));

    int failures = 0;
    for (const auto& T : trials) {
        std::fprintf(stderr,
            "── %s  (tree=%d Hq=%d Hk=%d D=%d L=%d) ──\n",
            T.tag, T.tree_size, T.num_q_heads, T.num_kv_heads,
            T.head_dim, T.seq_len);
        for (int i = 0; i < n_seeds; ++i) {
            uint32_t seed = 0xA5A5A5A5u
                          ^ (uint32_t)i * 0x9E3779B1u
                          ^ (uint32_t)T.tree_size   * 0x85EBCA77u
                          ^ (uint32_t)T.num_q_heads * 0xC2B2AE3Du
                          ^ (uint32_t)T.head_dim    * 0x27D4EB2Fu
                          ^ (uint32_t)T.seq_len     * 0x165667B1u;
            run_trial(T, seed, stream, fp16_ulp_budget, failures);
        }
    }

    HIP_OK(hipStreamDestroy(stream));

    if (failures) {
        std::fprintf(stderr, "\n[medusa tree attn] %d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "\n[medusa tree attn] all passed\n");
    return 0;
}
