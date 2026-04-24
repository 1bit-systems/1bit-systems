// 1bit.cpp — host-side sampler interface.
//
// C++20 mirror of `crates/1bit-router/src/sampler/mod.rs` plus
// `crates/1bit-router/src/sampler/cpu.rs`. The Rust implementation moves
// sampling onto a rayon Zen5 worker via `flume::bounded(1)` handoff; this
// pass lands just the interface + a `GreedySampler` that argmaxes on the
// calling thread. Top-k / top-p are declared but throw
// `std::runtime_error("not yet wired")` until the follow-up pass lands the
// CPU worker pool and the flume-equivalent bounded handoff.

#ifndef ONEBIT_CPP_SAMPLER_HPP
#define ONEBIT_CPP_SAMPLER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace onebit::cpp {

// Runtime-polymorphic sampler. The router holds a `std::unique_ptr<Sampler>`
// so a request can pick its sampler per-call without the cost of switching
// out the whole backend. All samplers take a logits view over the full vocab
// and return a token id.
//
// Thread-safety: samplers are single-request objects. A parallel CPU lane
// will instantiate one per worker thread; do not share across threads.
class Sampler {
public:
    virtual ~Sampler() = default;

    // Sample a token id from `logits` over a vocabulary of `logits.size()`
    // entries. Implementations MUST be deterministic given the same input
    // (modulo the RNG seed a future `MultinomialSampler` will take as a
    // ctor param). Greedy is deterministic by definition.
    virtual std::int32_t sample(std::span<const float> logits) = 0;

    // Name for logs / JSON metrics. No heap alloc per call — return a
    // static `string_view` pointing at a string literal.
    virtual const char* name() const noexcept = 0;
};

// Argmax over the logits. O(vocab); no heap alloc; branchless on the inner
// comparison. The only sampler implemented this pass.
class GreedySampler final : public Sampler {
public:
    std::int32_t sample(std::span<const float> logits) override;
    const char* name() const noexcept override { return "greedy"; }
};

// Top-k truncation followed by softmax + multinomial draw. Throws
// `std::runtime_error("not yet wired")` until the follow-up pass.
class TopKSampler final : public Sampler {
public:
    TopKSampler(std::size_t k, float temperature, std::uint64_t seed) noexcept
        : k_(k), temperature_(temperature), seed_(seed) {}

    std::int32_t sample(std::span<const float> logits) override;
    const char* name() const noexcept override { return "top-k"; }

private:
    std::size_t   k_;
    float         temperature_;
    std::uint64_t seed_;
};

// Top-p (nucleus) truncation followed by softmax + multinomial draw. Throws
// `std::runtime_error("not yet wired")` until the follow-up pass.
class TopPSampler final : public Sampler {
public:
    TopPSampler(float p, float temperature, std::uint64_t seed) noexcept
        : p_(p), temperature_(temperature), seed_(seed) {}

    std::int32_t sample(std::span<const float> logits) override;
    const char* name() const noexcept override { return "top-p"; }

private:
    float         p_;
    float         temperature_;
    std::uint64_t seed_;
};

// Convenience factory — one call site in `router.cpp` so any future
// sampler added here shows up everywhere at once.
std::unique_ptr<Sampler> make_greedy();
std::unique_ptr<Sampler> make_top_k(std::size_t k, float temperature, std::uint64_t seed);
std::unique_ptr<Sampler> make_top_p(float p, float temperature, std::uint64_t seed);

}  // namespace onebit::cpp

#endif  // ONEBIT_CPP_SAMPLER_HPP
