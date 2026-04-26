// onebit::onnx::generate — token-at-a-time decode loop over an OGA graph.
//
// Mirror of the Rust crate's GenerateRequest / GenerateResponse pair.
// Greedy + temperature-scaled top-k sampling; no beam, no nucleus,
// no streaming surface. Single request at a time — the server above
// us serializes.

#pragma once

#include <cstdint>
#include <expected>
#include <optional>
#include <string>

#include "onebit/onnx/error.hpp"
#include "onebit/onnx/session.hpp"

namespace onebit::onnx {

struct GenerateRequest {
    std::string                  prompt;
    std::size_t                  max_new_tokens{16};
    float                        temperature{0.0f};   // 0 == greedy
    std::optional<std::uint32_t> top_k{};
    std::optional<std::uint64_t> seed{};

    // Convenience: deterministic greedy.
    [[nodiscard]] static GenerateRequest greedy(std::string prompt,
                                                std::size_t  max_new_tokens) {
        return GenerateRequest{
            .prompt = std::move(prompt),
            .max_new_tokens = max_new_tokens,
            .temperature = 0.0f,
            .top_k = std::nullopt,
            .seed = std::nullopt,
        };
    }
};

struct GenerateResponse {
    std::string  text;
    std::size_t  prompt_tokens{};
    std::size_t  completion_tokens{};
    std::uint64_t wall_ms{};
    float        tokens_per_second{};
};

// Run `req` against `session`. Caller must pass a session built via
// `Session::load` (not `load_config_only`); `load_config_only` sessions
// return ErrorKind::SessionInit.
//
// Free function rather than a method to keep `Session` decoupled from
// the decode loop — alternate decode policies (speculative, draft) can
// plug in side-by-side without growing the class surface.
[[nodiscard]] std::expected<GenerateResponse, Error>
generate(Session& session, const GenerateRequest& req);

// --- internals exposed for testing ---------------------------------------

namespace detail {
// Argmax over a flat logits row.
[[nodiscard]] std::size_t argmax(const float* logits, std::size_t n) noexcept;

// xorshift64 → uniform float in [0, 1).
[[nodiscard]] float next_f32(std::uint64_t& state) noexcept;

// Sampling primitive. Mutates `logits` in place (temperature scale +
// top-k mask + softmax). Returns the sampled token id.
[[nodiscard]] std::int64_t sample_next(float*                       logits,
                                       std::size_t                  vocab,
                                       float                        temperature,
                                       std::optional<std::uint32_t> top_k,
                                       std::uint64_t&               rng);
} // namespace detail

} // namespace onebit::onnx
