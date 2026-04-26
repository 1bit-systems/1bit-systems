#include "onebit/core/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace onebit::core {

// -------- FastRand --------

std::uint64_t FastRand::next_u64() noexcept
{
    // xorshift64* — close enough to fastrand::Rng for our reproducibility
    // needs (we don't need bit-for-bit parity with the Rust crate, only
    // determinism within a process).
    state_ ^= state_ << 13;
    state_ ^= state_ >> 7;
    state_ ^= state_ << 17;
    return state_ * 0x2545F4914F6CDD1DULL;
}

float FastRand::next_f32() noexcept
{
    // Top 24 bits → [0, 1) float.
    const std::uint64_t bits = next_u64() >> 40; // 24 bits
    return static_cast<float>(bits) / static_cast<float>(1u << 24);
}

// -------- Sampler --------

Sampler::Sampler(SamplerConfig cfg) noexcept
    : cfg_(cfg), rng_(cfg.seed)
{}

void Sampler::set_config(SamplerConfig cfg) noexcept
{
    if (cfg.seed != cfg_.seed) {
        rng_ = FastRand{cfg.seed};
    }
    cfg_ = cfg;
}

std::expected<TokenId, HaloError>
Sampler::greedy(std::span<const float> logits) noexcept
{
    if (logits.empty()) {
        return std::unexpected(HaloError::sampler("empty logits"));
    }
    std::size_t best_idx = 0;
    float       best_val = logits[0];
    for (std::size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_idx = i;
        }
    }
    return static_cast<TokenId>(best_idx);
}

std::expected<TokenId, HaloError>
Sampler::sample(std::span<float> logits, std::span<const TokenId> recent)
{
    const std::size_t v = logits.size();
    if (v == 0) {
        return std::unexpected(HaloError::sampler("empty logits"));
    }

    // Greedy fast path (matches C++ temperature <= 0 branch).
    if (cfg_.temperature <= 0.0f) {
        return greedy(logits);
    }

    // Repetition penalty.
    if (std::fabs(cfg_.rep_penalty - 1.0f) > std::numeric_limits<float>::epsilon()
        && cfg_.rep_last_n > 0) {
        const std::size_t n     = recent.size();
        const std::size_t last  = static_cast<std::size_t>(cfg_.rep_last_n);
        const std::size_t start = (n > last) ? (n - last) : 0;
        for (std::size_t i = start; i < n; ++i) {
            const TokenId id = recent[i];
            if (id >= 0 && static_cast<std::size_t>(id) < v) {
                float& l = logits[static_cast<std::size_t>(id)];
                l = (l > 0.0f) ? (l / cfg_.rep_penalty)
                               : (l * cfg_.rep_penalty);
            }
        }
    }

    // Top-k: mask everything below the k-th largest to -inf.
    const std::size_t top_k = static_cast<std::size_t>(cfg_.top_k);
    if (top_k > 0 && top_k < v) {
        scratch_.assign(logits.begin(), logits.end());
        const std::size_t pivot = v - top_k;
        std::nth_element(
            scratch_.begin(), scratch_.begin() + pivot, scratch_.end(),
            [](float a, float b) { return a < b; });
        const float thresh = scratch_[pivot];
        for (float& l : logits) {
            if (l < thresh) {
                l = -std::numeric_limits<float>::infinity();
            }
        }
    }

    // Softmax with temperature.
    float max_logit = -std::numeric_limits<float>::infinity();
    for (float l : logits) {
        if (l > max_logit) max_logit = l;
    }
    const float inv_temp = 1.0f / cfg_.temperature;
    double sum = 0.0;
    for (float& l : logits) {
        const float e = std::exp((l - max_logit) * inv_temp);
        l   = e;
        sum += static_cast<double>(e);
    }
    const float inv = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 1.0f;
    for (float& l : logits) {
        l *= inv;
    }

    // Top-p (nucleus).
    if (cfg_.top_p > 0.0f && cfg_.top_p < 1.0f) {
        index_scratch_.resize(v);
        std::iota(index_scratch_.begin(), index_scratch_.end(), 0u);
        std::sort(index_scratch_.begin(), index_scratch_.end(),
                  [&logits](std::uint32_t a, std::uint32_t b) {
                      return logits[a] > logits[b];
                  });
        float       csum   = 0.0f;
        std::size_t cutoff = v;
        for (std::size_t i = 0; i < v; ++i) {
            csum += logits[index_scratch_[i]];
            if (csum >= cfg_.top_p) {
                cutoff = i + 1;
                break;
            }
        }
        for (std::size_t i = cutoff; i < v; ++i) {
            logits[index_scratch_[i]] = 0.0f;
        }
        float keep_sum = 0.0f;
        for (std::size_t i = 0; i < cutoff; ++i) {
            keep_sum += logits[index_scratch_[i]];
        }
        if (keep_sum > 0.0f) {
            const float s = 1.0f / keep_sum;
            for (std::size_t i = 0; i < cutoff; ++i) {
                logits[index_scratch_[i]] *= s;
            }
        }
    }

    // Multinomial draw.
    const float r = rng_.next_f32();
    float       acc = 0.0f;
    for (std::size_t i = 0; i < v; ++i) {
        acc += logits[i];
        if (acc >= r) {
            return static_cast<TokenId>(i);
        }
    }
    return static_cast<TokenId>(v - 1);
}

} // namespace onebit::core
