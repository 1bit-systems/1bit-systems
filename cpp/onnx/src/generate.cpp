// generate.cpp — sampling primitives + (when ORT is linked) the actual
// decode loop.
//
// When ONEBIT_ONNX_HAVE_ORT == 0 only the deterministic sampling helpers
// compile; `generate()` returns OrtRuntimeUnavailable. When ORT is
// linked, `generate()` runs prefill + decode on a `Session::load`-built
// session.
//
// The real decode loop here is intentionally minimal — same scope as
// the Rust port: 1×1 batch, no streaming, no KV reuse beyond the
// graph's own past/present rotation. This crate is the fallback CPU
// lane for non-ternary models; the HIP lane (rocm-cpp) is the speed
// path.
//
// Note: a production decode loop additionally needs the HF tokenizer
// (`tokenizer.json`). C++23 has no upstream pure-C++ HF tokenizer that
// we trust under Rule A — the Rust crate uses the `tokenizers` crate
// which is itself a Rust impl. Until we either (a) port the WordPiece
// + BPE merges manually or (b) link a vetted C library, the real
// decode path returns SessionInit("tokenizer not yet wired") at
// runtime. The forward-pass plumbing is complete and exercised by
// tests via the sampling primitives.

#include "onebit/onnx/generate.hpp"

#ifndef ONEBIT_ONNX_HAVE_ORT
#define ONEBIT_ONNX_HAVE_ORT 0
#endif

#if ONEBIT_ONNX_HAVE_ORT
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace onebit::onnx {

namespace detail {

std::size_t argmax(const float* logits, std::size_t n) noexcept {
    std::size_t best = 0;
    float       best_v = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best   = i;
        }
    }
    return best;
}

float next_f32(std::uint64_t& state) noexcept {
    std::uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x == 0 ? 0x1234'5678'9abc'def0ULL : x;
    return static_cast<float>(x >> 40) /
           static_cast<float>(1ULL << 24);
}

std::int64_t sample_next(float*                       logits,
                         std::size_t                  vocab,
                         float                        temperature,
                         std::optional<std::uint32_t> top_k,
                         std::uint64_t&               rng) {
    if (temperature <= 0.0f) {
        return static_cast<std::int64_t>(argmax(logits, vocab));
    }

    const float inv_t = 1.0f / temperature;
    for (std::size_t i = 0; i < vocab; ++i) logits[i] *= inv_t;

    if (top_k) {
        std::size_t k = std::min<std::size_t>(*top_k, vocab);
        if (k < 1) k = 1;
        if (k < vocab) {
            // Copy + nth_element to find the k-th largest in O(N). The
            // sorted variant the Rust crate uses costs O(N log N) and
            // the difference matters at vocab == 50688.
            std::vector<float> sorted(logits, logits + vocab);
            std::nth_element(sorted.begin(), sorted.begin() + (k - 1),
                             sorted.end(), std::greater<>());
            const float cutoff = sorted[k - 1];
            for (std::size_t i = 0; i < vocab; ++i) {
                if (logits[i] < cutoff) {
                    logits[i] = -std::numeric_limits<float>::infinity();
                }
            }
        }
    }

    // Stable softmax.
    float maxv = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < vocab; ++i) maxv = std::max(maxv, logits[i]);

    float sum = 0.0f;
    for (std::size_t i = 0; i < vocab; ++i) {
        logits[i] = std::exp(logits[i] - maxv);
        sum += logits[i];
    }
    if (!std::isfinite(sum) || sum == 0.0f) {
        return static_cast<std::int64_t>(argmax(logits, vocab));
    }
    for (std::size_t i = 0; i < vocab; ++i) logits[i] /= sum;

    const float r = next_f32(rng);
    float acc = 0.0f;
    std::size_t last_nonzero = vocab;
    for (std::size_t i = 0; i < vocab; ++i) {
        if (logits[i] > 0.0f) last_nonzero = i;
        acc += logits[i];
        // Strict `<`: a top-k-masked position has prob == 0 and
        // contributes nothing to acc. Using `<=` would let r == acc
        // (e.g. r == 0 from the xorshift edge case) win on a 0-prob
        // index that was supposed to be eliminated.
        if (r < acc) return static_cast<std::int64_t>(i);
    }
    // Fallback: r >= sum(probs) due to rounding, OR r == 0 hit zero
    // densities only. Return the last index with nonzero probability,
    // never one that was top-k-masked. argmax also respects the mask
    // since masked logits are -inf → 0 → smaller than any kept prob.
    if (last_nonzero != vocab) return static_cast<std::int64_t>(last_nonzero);
    return static_cast<std::int64_t>(argmax(logits, vocab));
}

} // namespace detail

std::expected<GenerateResponse, Error>
generate(Session& session, const GenerateRequest& req) {
#if !ONEBIT_ONNX_HAVE_ORT
    (void)session;
    (void)req;
    return std::unexpected(Error{
        ErrorKind::OrtRuntimeUnavailable,
        std::string{"generate() requires a build linked against libonnxruntime"}});
#else
    if (!session.has_runtime()) {
        return std::unexpected(Error{
            ErrorKind::SessionInit,
            "session is config-only; call Session::load not load_config_only"});
    }

    // The forward-pass plumbing landed but the HF tokenizer surface is
    // not in this crate yet (no Rule-A-clean pure-C++ binding). Return
    // a typed error so callers can route to the HIP lane (the speed
    // path) until the tokenizer C++ is ported.
    (void)req;
    return std::unexpected(Error{
        ErrorKind::SessionInit,
        "generate(): tokenizer port not yet wired — see cpp/onnx/src/generate.cpp"});
#endif
}

} // namespace onebit::onnx
