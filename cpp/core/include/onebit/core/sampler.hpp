#pragma once

#include "onebit/core/error.hpp"
#include "onebit/core/types.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <vector>

namespace onebit::core {

// Sampling hyperparameters.
//
// Defaults are conservative: temperature == 0 enables greedy argmax,
// identical to rcpp_argmax_fp32 on device.
struct SamplerConfig {
    float           temperature = 0.0f;
    std::uint32_t   top_k       = 0;
    float           top_p       = 1.0f;
    float           rep_penalty = 1.0f;
    std::uint32_t   rep_last_n  = 64;
    std::uint64_t   seed        = 0xC0FFEE;
};

// Lightweight LCG matching `fastrand::Rng` semantics closely enough for
// reproducibility within a single process. Emits f32 in [0, 1).
class FastRand {
public:
    explicit FastRand(std::uint64_t seed) noexcept : state_(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    [[nodiscard]] std::uint64_t next_u64() noexcept;
    [[nodiscard]] float         next_f32() noexcept;

private:
    std::uint64_t state_;
};

// Stateful sampler — owns the RNG so draws are reproducible across turns.
class Sampler {
public:
    explicit Sampler(SamplerConfig cfg) noexcept;

    [[nodiscard]] const SamplerConfig& config() const noexcept { return cfg_; }
    void set_config(SamplerConfig cfg) noexcept;

    // Greedy argmax — returns the token id with the largest logit.
    // Identical to rcpp_argmax_fp32. Independent of sampler state.
    [[nodiscard]] static std::expected<TokenId, HaloError>
    greedy(std::span<const float> logits) noexcept;

    // Full sampler path (temperature > 0).
    //
    // `logits` is mutated in-place (rep-penalty + masking + softmax) so
    // callers needing raw logits for logging should clone first.
    // `recent` is the full decode history; only the last rep_last_n
    // entries are read.
    [[nodiscard]] std::expected<TokenId, HaloError>
    sample(std::span<float> logits, std::span<const TokenId> recent);

private:
    SamplerConfig             cfg_;
    FastRand                  rng_;
    std::vector<float>        scratch_;
    std::vector<std::uint32_t> index_scratch_;
};

} // namespace onebit::core
