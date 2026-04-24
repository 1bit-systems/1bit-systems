// 1bit.cpp — host-side sampler (impl).
//
// Only `GreedySampler` is wired this pass. Top-k / top-p throw until the
// CPU-lane commit lands. The Rust source of truth is
// `crates/1bit-router/src/sampler/cpu.rs` (top-k partitioned selection +
// temperature softmax + nucleus renorm + multinomial draw).

#include "onebit_cpp/sampler.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace onebit::cpp {

std::int32_t GreedySampler::sample(std::span<const float> logits) {
    if (logits.empty()) {
        throw std::runtime_error("GreedySampler::sample: empty logits");
    }
    std::int32_t best_idx = 0;
    float best_val = logits[0];
    // Scalar loop. The next pass vectorizes with a 16-way SIMD argmax on
    // Zen5 (AVX-512 VNNI / BF16 available). The Rust side currently uses
    // a scalar `.enumerate().max_by` too — parity first, perf after.
    const std::size_t n = logits.size();
    for (std::size_t i = 1; i < n; ++i) {
        const float v = logits[i];
        if (v > best_val) {
            best_val = v;
            best_idx = static_cast<std::int32_t>(i);
        }
    }
    return best_idx;
}

std::int32_t TopKSampler::sample(std::span<const float> /*logits*/) {
    throw std::runtime_error(
        "TopKSampler::sample: not yet wired — scheduled for the sampler "
        "CPU-lane port (crates/1bit-router/src/sampler/cpu.rs)");
}

std::int32_t TopPSampler::sample(std::span<const float> /*logits*/) {
    throw std::runtime_error(
        "TopPSampler::sample: not yet wired — scheduled for the sampler "
        "CPU-lane port (crates/1bit-router/src/sampler/cpu.rs)");
}

std::unique_ptr<Sampler> make_greedy() {
    return std::make_unique<GreedySampler>();
}

std::unique_ptr<Sampler>
make_top_k(std::size_t k, float temperature, std::uint64_t seed) {
    return std::make_unique<TopKSampler>(k, temperature, seed);
}

std::unique_ptr<Sampler>
make_top_p(float p, float temperature, std::uint64_t seed) {
    return std::make_unique<TopPSampler>(p, temperature, seed);
}

}  // namespace onebit::cpp
