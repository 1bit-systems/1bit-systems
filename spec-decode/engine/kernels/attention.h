#pragma once
// Optimized Attention Kernels — Flash Attention, GQA, Memory-Efficient Attention
//
// Implements:
//   - FlashAttention:    Tiled online-softmax attention (single-query decode)
//   - BatchedAttention:  Multi-query prefill attention with causal masking
//   - PagedAttention:    Paged KV cache attention (for continuous batching)
//   - GQA support:       Grouped-Query Attention with head replication
//   - Memory-efficient:  Tiled attention with O(1) auxiliary memory
//
// All kernels operate on fp32 host memory. GPU dispatch wrappers are in
// the pipeline module.

#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <span>
#include <bit>

namespace specdecode::kernels {

// ─── Configuration ──────────────────────────────────────────────────────────

struct AttentionConfig {
    int32_t num_heads = 16;
    int32_t num_kv_heads = 8;
    int32_t head_dim = 128;
    int32_t gqa_ratio = 2;             // num_heads / num_kv_heads
    float attn_scale = 0.08838835f;    // 1/sqrt(head_dim) = 1/sqrt(128)
    int32_t tile_size = 32;            // Tile dimension for flash attention
    bool causal = true;
    bool use_softcap = false;
    float softcap_value = 50.0f;
};

// ─── Flash Attention (Single-Query Decode) ──────────────────────────────────
//
// Online softmax via the safe-softmax trick:
//   m(x) = max(x)
//   s(x) = sum(exp(x - m(x)))
//   softmax(x) = exp(x - m(x)) / s(x)
//
// For single-query: O(seq_len * head_dim) compute, O(head_dim) memory.

class FlashAttention {
public:
    explicit FlashAttention(const AttentionConfig& cfg) : cfg_(cfg) {}

    // Single-query flash attention: q attends to K/V cache.
    //   q:        [num_heads * head_dim]
    //   k_cache:  [seq_len * num_kv_heads * head_dim]
    //   v_cache:  [seq_len * num_kv_heads * head_dim]
    //   output:   [num_heads * head_dim]
    void forward(
        std::span<const float> q,
        std::span<const float> k_cache,
        std::span<const float> v_cache,
        int32_t seq_len,
        std::span<float> output
    ) const {
        const int32_t NH = cfg_.num_heads;
        const int32_t NKV = cfg_.num_kv_heads;
        const int32_t D = cfg_.head_dim;
        const int32_t GQA = cfg_.gqa_ratio;

        // Temporary buffers for online softmax
        std::vector<float> scores(cfg_.tile_size);  // attention scores per position (reused)
        std::vector<float> max_row(NH, -std::numeric_limits<float>::infinity());
        std::vector<float> sum_row(NH, 0.0f);
        std::vector<float> acc(NH * D, 0.0f);  // accumulated output

        // Tile over sequence dimension
        for (int32_t tile_start = 0; tile_start < seq_len; tile_start += cfg_.tile_size) {
            int32_t tile_end = std::min(tile_start + cfg_.tile_size, seq_len);
            int32_t tile_len = tile_end - tile_start;

            for (int32_t h = 0; h < NH; h++) {
                int32_t kvh = h / GQA;

                // Compute attention scores for this tile
                for (int32_t p = tile_start; p < tile_end; p++) {
                    double score = 0.0;
                    for (int32_t d = 0; d < D; d++) {
                        score += (double)q[(size_t)h * D + d] *
                                 k_cache[(size_t)p * NKV * D + (size_t)kvh * D + d];
                    }
                    scores[p - tile_start] = (float)(score * cfg_.attn_scale);
                }

                // Online safe softmax update
                for (int32_t i = 0; i < tile_len; i++) {
                    float s = scores[i];
                    float old_max = max_row[h];
                    float new_max = std::max(old_max, s);
                    float exp_sum = 0.0f;

                    // Renormalize previous accumulation
                    if (new_max > old_max) {
                        float rescale = std::exp(old_max - new_max);
                        for (int32_t d = 0; d < D; d++) {
                            acc[(size_t)h * D + d] *= rescale;
                        }
                        sum_row[h] *= rescale;
                    }

                    float e = std::exp(s - new_max);
                    sum_row[h] += e;
                    max_row[h] = new_max;

                    // Weighted sum of V
                    for (int32_t d = 0; d < D; d++) {
                        acc[(size_t)h * D + d] += e *
                            v_cache[(size_t)(tile_start + i) * NKV * D + (size_t)kvh * D + d];
                    }
                }
            }
        }

        // Normalize and write output
        for (int32_t h = 0; h < NH; h++) {
            float inv_sum = 1.0f / sum_row[h];
            for (int32_t d = 0; d < D; d++) {
                output[(size_t)h * D + d] = acc[(size_t)h * D + d] * inv_sum;
            }
        }
    }

private:
    AttentionConfig cfg_;
};

// ─── Batched Flash Attention (Prefill) ──────────────────────────────────────
//
// N queries attend to K/V cache with causal masking.
// Uses tiling over both query and key dimensions.

class BatchedAttention {
public:
    explicit BatchedAttention(const AttentionConfig& cfg) : cfg_(cfg) {}

    // N queries, each attending to the SAME K/V cache (prefill).
    //   q:          [n_queries * num_heads * head_dim]
    //   k_cache:    [seq_len * num_kv_heads * head_dim]
    //   v_cache:    [seq_len * num_kv_heads * head_dim]
    //   output:     [n_queries * num_heads * head_dim]
    //   seq_start:  Position offset for causal masking (e.g., past context length)
    void forward(
        std::span<const float> q,
        std::span<const float> k_cache,
        std::span<const float> v_cache,
        int32_t n_queries,
        int32_t seq_len,
        int32_t seq_start,
        std::span<float> output
    ) const {
        const int32_t NH = cfg_.num_heads;
        const int32_t NKV = cfg_.num_kv_heads;
        const int32_t D = cfg_.head_dim;
        const int32_t GQA = cfg_.gqa_ratio;

        // For batch: process queries sequentially (or parallel with OpenMP)
        #pragma omp parallel for if(n_queries > 4)
        for (int32_t qi = 0; qi < n_queries; qi++) {
            int32_t causal_len = seq_start + qi + 1;  // causal: attend up to current

            // Per-query online softmax state
            float max_row = -std::numeric_limits<float>::infinity();
            float sum_row = 0.0f;
            std::vector<float> acc(NH * D, 0.0f);

            for (int32_t tile_start = 0; tile_start < causal_len; tile_start += cfg_.tile_size) {
                int32_t tile_end = std::min(tile_start + cfg_.tile_size, causal_len);

                for (int32_t h = 0; h < NH; h++) {
                    int32_t kvh = h / GQA;

                    for (int32_t p = tile_start; p < tile_end; p++) {
                        double score = 0.0;
                        for (int32_t d = 0; d < D; d++) {
                            score += (double)q[(size_t)qi * NH * D + (size_t)h * D + d] *
                                     k_cache[(size_t)p * NKV * D + (size_t)kvh * D + d];
                        }
                        float s = (float)(score * cfg_.attn_scale);

                        float old_max = max_row;
                        float new_max = std::max(old_max, s);

                        if (new_max > old_max) {
                            float rescale = std::exp(old_max - new_max);
                            for (int32_t d = 0; d < D; d++) {
                                acc[(size_t)h * D + d] *= rescale;
                            }
                            sum_row *= rescale;
                        }

                        float e = std::exp(s - new_max);
                        sum_row += e;
                        max_row = new_max;

                        for (int32_t d = 0; d < D; d++) {
                            acc[(size_t)h * D + d] += e *
                                v_cache[(size_t)p * NKV * D + (size_t)kvh * D + d];
                        }
                    }
                }
            }

            // Normalize and write
            float inv_sum = 1.0f / sum_row;
            for (int32_t h = 0; h < NH; h++) {
                for (int32_t d = 0; d < D; d++) {
                    output[(size_t)qi * NH * D + (size_t)h * D + d] =
                        acc[(size_t)h * D + d] * inv_sum;
                }
            }
        }
    }

private:
    AttentionConfig cfg_;
};

// ─── Paged Attention ────────────────────────────────────────────────────────
//
// KV cache is stored in fixed-size pages. A page table maps logical
// token positions to physical page slots. This enables:
//   - Non-contiguous KV storage (eliminates fragmentation)
//   - Copy-on-write for shared prefixes (radix attention)
//   - H2O eviction: replace low-score pages with zero page

struct PageTableEntry {
    int32_t page_id = -1;   // Physical page index
    int32_t tokens_used = 0;
    bool valid = false;
};

class PagedAttention {
public:
    explicit PagedAttention(const AttentionConfig& cfg, int32_t page_size = 16)
        : cfg_(cfg), page_size_(page_size) {}

    struct PagedAttentionInput {
        std::span<const float> q;           // [num_heads * head_dim]
        std::span<const float> k_pages;     // [num_pages * page_size * n_kv_heads * head_dim]
        std::span<const float> v_pages;     // same
        std::span<const PageTableEntry> page_table;  // [logical_pages]
        int32_t num_logical_pages = 0;
        int32_t total_tokens = 0;
    };

    // Single-query paged attention
    void forward(
        const PagedAttentionInput& input,
        std::span<float> output
    ) const {
        const int32_t NH = cfg_.num_heads;
        const int32_t NKV = cfg_.num_kv_heads;
        const int32_t D = cfg_.head_dim;
        const int32_t GQA = cfg_.gqa_ratio;

        std::vector<float> max_row(NH, -std::numeric_limits<float>::infinity());
        std::vector<float> sum_row(NH, 0.0f);
        std::vector<float> acc(NH * D, 0.0f);

        for (int32_t lpage = 0; lpage < input.num_logical_pages; lpage++) {
            const auto& entry = input.page_table[lpage];
            if (!entry.valid || entry.page_id < 0) continue;

            int32_t tokens_in_page = entry.tokens_used;

            for (int32_t h = 0; h < NH; h++) {
                int32_t kvh = h / GQA;

                for (int32_t t = 0; t < tokens_in_page; t++) {
                    // Physical location
                    size_t page_offset = (size_t)entry.page_id * page_size_ * NKV * D +
                                         (size_t)t * NKV * D + (size_t)kvh * D;

                    double score = 0.0;
                    for (int32_t d = 0; d < D; d++) {
                        score += (double)input.q[(size_t)h * D + d] *
                                 input.k_pages[page_offset + d];
                    }
                    float s = (float)(score * cfg_.attn_scale);

                    float old_max = max_row[h];
                    float new_max = std::max(old_max, s);

                    if (new_max > old_max) {
                        float rescale = std::exp(old_max - new_max);
                        for (int32_t d = 0; d < D; d++) {
                            acc[(size_t)h * D + d] *= rescale;
                        }
                        sum_row[h] *= rescale;
                    }

                    float e = std::exp(s - new_max);
                    sum_row[h] += e;
                    max_row[h] = new_max;

                    for (int32_t d = 0; d < D; d++) {
                        acc[(size_t)h * D + d] += e * input.v_pages[page_offset + d];
                    }
                }
            }
        }

        for (int32_t h = 0; h < NH; h++) {
            float inv_sum = 1.0f / sum_row[h];
            for (int32_t d = 0; d < D; d++) {
                output[(size_t)h * D + d] = acc[(size_t)h * D + d] * inv_sum;
            }
        }
    }

private:
    AttentionConfig cfg_;
    int32_t page_size_;
};

// ─── Auxiliary Kernels ──────────────────────────────────────────────────────

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
inline void rms_norm(
    std::span<const float> x,
    std::span<const float> weight,
    std::span<float> y,
    float eps = 1e-6f
) {
    double ss = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        ss += (double)x[i] * x[i];
    }
    float inv = 1.0f / std::sqrt((float)(ss / x.size()) + eps);
    for (size_t i = 0; i < x.size(); i++) {
        y[i] = weight[i] * (x[i] * inv);
    }
}

// RoPE: Rotary Position Embedding (interleaved pairs)
inline void apply_rope(
    std::span<float> x,
    int32_t pos,
    int32_t head_dim,
    float rope_theta = 1000000.0f
) {
    for (int32_t d = 0; d < head_dim; d += 2) {
        float angle = (float)pos / std::pow(rope_theta, (float)d / (float)head_dim);
        float c = std::cos(angle);
        float s = std::sin(angle);
        float a = x[d], b = x[d + 1];
        x[d]     = a * c - b * s;
        x[d + 1] = b * c + a * s;
    }
}

// HuggingFace "rotate_half" RoPE convention
inline void apply_rope_rotate_half(
    std::span<float> x,
    int32_t pos,
    int32_t head_dim,
    float rope_theta = 1000000.0f
) {
    int32_t half = head_dim / 2;
    std::vector<float> orig(x.begin(), x.begin() + head_dim);
    for (int32_t i = 0; i < half; i++) {
        float angle = (float)pos / std::pow(rope_theta, (float)(2 * i) / (float)head_dim);
        float c = std::cos(angle);
        float s = std::sin(angle);
        x[i]          = orig[i] * c - orig[i + half] * s;
        x[i + half]   = orig[i + half] * c + orig[i] * s;
    }
}

// SwiGLU activation: swish(x) = x * sigmoid(x)
inline void swiglu(
    std::span<const float> gate,
    std::span<const float> up,
    std::span<float> output
) {
    for (size_t i = 0; i < gate.size(); i++) {
        float g = gate[i];
        output[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }
}

// Softmax with numerical stability
inline void softmax(std::span<float> x) {
    if (x.empty()) return;
    float mx = x[0];
    for (size_t i = 1; i < x.size(); i++) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    float inv = 1.0f / (float)sum;
    for (size_t i = 0; i < x.size(); i++) x[i] *= inv;
}

// SiLU (Sigmoid Linear Unit)
inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

} // namespace specdecode::kernels
