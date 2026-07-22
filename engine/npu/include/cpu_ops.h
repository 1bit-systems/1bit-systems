#pragma once
/// CPU-side math operations for the NPU inference engine.
/// These run on the host CPU for operations the NPU xclbin cannot accelerate
/// (attention softmax, RoPE, RMS norm, activation functions, LM head).
/// Ported from engine/npu/src/cpu_ops.zig
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <algorithm>

// ─── Constants ───────────────────────────────────────────────────
constexpr float F32_MAX = FLT_MAX;

// ─── RMS Normalization ───────────────────────────────────────────

/// In-place RMS normalization: x[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
/// Sets NaN/Inf inputs to 0 before computing.
inline void rmsNorm(float* x, size_t n, const float* weight, float eps) {
    if (n == 0) return;

    // Clean NaN/Inf
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) x[i] = 0.0f;
    }

    // Sum of squares (double precision for stability)
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_sq += (double)x[i] * (double)x[i];
    }

    double mean = sum_sq / (double)n;
    double inv_rms = 1.0 / std::sqrt(mean + (double)eps);

    for (size_t i = 0; i < n; ++i) {
        x[i] = (float)((double)x[i] * inv_rms * (double)weight[i]);
    }
}

// ─── RoPE (Rotary Position Embedding) ────────────────────────────

/// Apply rotary position embedding to a single head's query/key vector.
/// x is [hd] interleaved pairs. rc/rs are precomputed tables at offset pos*hd.
inline void rope(float* x, size_t hd, size_t pos, const float* rc, const float* rs) {
    size_t hd2 = hd / 2;
    size_t base = pos * hd;
    for (size_t d = 0; d < hd2; ++d) {
        float a = x[d];
        float b = x[d + hd2];
        float c = rc[base + d];
        float s = rs[base + d];
        x[d] = a * c - b * s;
        x[d + hd2] = a * s + b * c;
    }
}

/// Precompute RoPE cos/sin tables for up to max_pos positions.
/// Returns (rc, rs) vectors. Caller must free.
inline void precomputeRoPE(float*& rc, float*& rs, size_t hd, size_t max_pos, float theta) {
    float hd2_f = (float)hd;
    rc = new float[max_pos * hd];
    rs = new float[max_pos * hd];

    for (size_t pos = 0; pos < max_pos; ++pos) {
        float pf = (float)pos;
        for (size_t d = 0; d < hd / 2; ++d) {
            float df = (float)d;
            float angle = pf / std::pow(theta, 2.0f * df / hd2_f);
            float c = std::cos(angle);
            float s = std::sin(angle);
            rc[pos * hd + d] = c;
            rs[pos * hd + d] = s;
            // Mirror for second half
            rc[pos * hd + hd / 2 + d] = c;
            rs[pos * hd + hd / 2 + d] = s;
        }
    }
}

// ─── Softmax (numerically stable) ────────────────────────────────

/// In-place softmax with numeric stability (max subtraction).
inline void softmax(float* scores, size_t n) {
    if (n == 0) return;

    // Clean NaN/Inf and find max
    float mx = -F32_MAX;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(scores[i])) scores[i] = -F32_MAX;
        if (scores[i] > mx) mx = scores[i];
    }

    // If all -inf (masked), return uniform
    if (mx <= -F32_MAX / 2.0f) {
        float inv_n = 1.0f / (float)n;
        for (size_t i = 0; i < n; ++i) scores[i] = inv_n;
        return;
    }

    // exp(x - max) and sum
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float diff = scores[i] - mx;
        double e = (diff < -80.0f) ? 0.0 : std::exp((double)diff);
        scores[i] = (float)e;
        sum += e;
    }

    // Normalize
    if (sum <= 0.0) {
        float inv_n = 1.0f / (float)n;
        for (size_t i = 0; i < n; ++i) scores[i] = inv_n;
    } else {
        float inv_sum = (float)(1.0 / sum);
        for (size_t i = 0; i < n; ++i) scores[i] *= inv_sum;
    }
}

// ─── Attention operations ────────────────────────────────────────

/// Compute Q*K^T attention scores for a single head.
/// q: [hd], k_cache: [cl * nkv * hd], scores out: [cl]
inline void attentionQK(const float* q, const float* k_cache, float* scores,
                        size_t cl, size_t nkv, size_t kvh, size_t hd, size_t max_pos) {
    float scale = 1.0f / std::sqrt((float)hd);
    size_t k_offset = kvh * hd;

    for (size_t p = 0; p < cl; ++p) {
        if (max_pos > 0 && p >= max_pos) {
            scores[p] = -F32_MAX;
            continue;
        }
        double s = 0.0;
        const float* kp = &k_cache[p * nkv * hd + k_offset];
        for (size_t d = 0; d < hd; ++d) {
            s += (double)q[d] * (double)kp[d];
        }
        scores[p] = (float)(s * (double)scale);
    }
}

/// Compute weighted sum of values by softmax scores.
inline void attentionPV(const float* scores, const float* v_cache, float* output,
                        size_t cl, size_t nkv, size_t kvh, size_t hd) {
    size_t v_offset = kvh * hd;

    for (size_t d = 0; d < hd; ++d) {
        double acc = 0.0;
        for (size_t p = 0; p < cl; ++p) {
            acc += (double)scores[p] * (double)v_cache[p * nkv * hd + v_offset + d];
        }
        output[d] = (float)acc;
    }
}

/// Full single-head attention: scores -> softmax -> weighted sum.
inline void attentionHead(const float* q, const float* k_cache, const float* v_cache,
                          float* output, size_t cl, size_t hh, size_t nh, size_t nkv,
                          size_t hd, size_t max_pos) {
    size_t gqa = nh / nkv;
    size_t kvh = hh / gqa;

    // Scratch scores (max 4096 positions)
    constexpr size_t MAX_CTX = 4096;
    float scores_buf[MAX_CTX];
    if (cl > MAX_CTX) throw std::runtime_error("Context too long for CPU attention");

    attentionQK(q, k_cache, scores_buf, cl, nkv, kvh, hd, max_pos);
    softmax(scores_buf, cl);
    attentionPV(scores_buf, v_cache, output, cl, nkv, kvh, hd);
}

// ─── Activation functions ────────────────────────────────────────

/// SiLU (Sigmoid Linear Unit): x / (1 + exp(-x))
inline float silu(float x) {
    if (!std::isfinite(x)) return (x > 0) ? x : 0.0f;
    return x / (1.0f + std::exp(-x));
}

/// Find maximum absolute value in an array (for dynamic INT8 scaling).
inline float findMaxAbs(const float* x, size_t n) {
    float amax = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (std::isfinite(x[i])) {
            float a = std::abs(x[i]);
            if (a > amax) amax = a;
        }
    }
    return (amax < 1e-12f) ? 1.0f : amax;
}

/// Set NaN/Inf values to 0.
inline void clipAndClean(float* x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i])) x[i] = 0.0f;
    }
}

// ─── LM Head (vocabulary projection) ─────────────────────────────

/// Compute logits = hidden @ embeddings^T
/// embeddings is [nv * h] row-major, hidden is [h], logits out is [nv]
inline void lmHead(const float* hidden, const float* embeddings, float* logits,
                   size_t nv, size_t h) {
    for (size_t n = 0; n < nv; ++n) {
        double s = 0.0;
        const float* emb_row = &embeddings[n * h];
        for (size_t i = 0; i < h; ++i) {
            s += (double)hidden[i] * (double)emb_row[i];
        }
        logits[n] = (float)s;
    }
}

/// Compute logits, softmax, find top-k tokens.
inline void lmHeadTopK(const float* hidden, const float* embeddings, uint32_t* top_ids,
                       uint32_t k, size_t nv, size_t h) {
    constexpr size_t MAX_VOCAB = 200000;
    float logits_buf[MAX_VOCAB];
    if (nv > MAX_VOCAB) throw std::runtime_error("Vocabulary too large for CPU LM head");

    float* logits = logits_buf;

    // Compute logits
    lmHead(hidden, embeddings, logits, nv, h);

    // Softmax with max subtraction
    float mx = -F32_MAX;
    for (size_t i = 0; i < nv; ++i) {
        if (logits[i] > mx) mx = logits[i];
    }
    double sum = 0.0;
    for (size_t i = 0; i < nv; ++i) {
        float diff = logits[i] - mx;
        double e = (diff < -80.0f) ? 0.0 : std::exp((double)diff);
        logits[i] = (float)e;
        sum += e;
    }
    if (sum <= 0.0) {
        float inv_nv = 1.0f / (float)nv;
        for (size_t i = 0; i < nv; ++i) logits[i] = inv_nv;
    } else {
        float inv_sum = (float)(1.0 / sum);
        for (size_t i = 0; i < nv; ++i) logits[i] *= inv_sum;
    }

    // Greedy: pick highest probability
    uint32_t best_id = 0;
    float best_p = -1.0f;
    for (size_t i = 0; i < nv; ++i) {
        if (logits[i] > best_p) {
            best_p = logits[i];
            best_id = (uint32_t)i;
        }
    }
    top_ids[0] = best_id;

    // Top-k (simple O(n*k) selection for small k)
    if (nv > 4096) throw std::runtime_error("Vocab too large for CPU top-k");
    bool used[4096] = {false};
    size_t k_s = std::min((size_t)k, nv);
    for (size_t ki = 0; ki < k_s; ++ki) {
        float best = -1.0f;
        size_t best_idx = 0;
        for (size_t i = 0; i < nv; ++i) {
            if (!used[i] && logits[i] > best) {
                best = logits[i];
                best_idx = i;
            }
        }
        top_ids[ki] = (uint32_t)best_idx;
        used[best_idx] = true;
    }
}
