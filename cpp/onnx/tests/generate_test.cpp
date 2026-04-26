// generate_test.cpp — exercise the deterministic sampling primitives
// without any ORT dependency. The live decode path is exercised in
// integration via the strix-halo box.

#include <doctest/doctest.h>

#include "onebit/onnx/generate.hpp"

#include <vector>

using namespace onebit::onnx;
using onebit::onnx::detail::argmax;
using onebit::onnx::detail::next_f32;
using onebit::onnx::detail::sample_next;

TEST_CASE("argmax picks global max")
{
    std::vector<float> xs{0.1f, 0.9f, 0.4f, -1.0f};
    CHECK(argmax(xs.data(), xs.size()) == 1);
}

TEST_CASE("argmax handles ties — returns first")
{
    std::vector<float> xs{0.5f, 0.5f, 0.5f};
    CHECK(argmax(xs.data(), xs.size()) == 0);
}

TEST_CASE("sample_next greedy matches argmax at temperature 0")
{
    std::uint64_t rng = 1;
    std::vector<float> logits{0.1f, 0.9f, 0.4f, -1.0f};
    CHECK(sample_next(logits.data(), logits.size(), 0.0f, std::nullopt, rng) == 1);
}

TEST_CASE("sample_next deterministic with same seed")
{
    std::uint64_t a = 42, b = 42;
    std::vector<float> la{0.1f, 0.9f, 0.4f, -1.0f};
    std::vector<float> lb{0.1f, 0.9f, 0.4f, -1.0f};
    auto ra = sample_next(la.data(), la.size(), 1.0f, std::uint32_t{2}, a);
    auto rb = sample_next(lb.data(), lb.size(), 1.0f, std::uint32_t{2}, b);
    CHECK(ra == rb);
}

TEST_CASE("next_f32 stays in [0, 1)")
{
    std::uint64_t rng = 0xdeadbeef;
    for (int i = 0; i < 64; ++i) {
        float r = next_f32(rng);
        CHECK(r >= 0.0f);
        CHECK(r < 1.0f);
    }
}

TEST_CASE("greedy() builder produces deterministic config")
{
    auto r = GenerateRequest::greedy("hello", 16);
    CHECK(r.temperature       == 0.0f);
    CHECK_FALSE(r.top_k.has_value());
    CHECK_FALSE(r.seed.has_value());
    CHECK(r.max_new_tokens    == 16);
    CHECK(r.prompt            == "hello");
}

TEST_CASE("top_k truncation clamps small-k correctly")
{
    // After temperature scaling, only the top 2 should retain finite
    // probability mass.
    std::vector<float> logits{0.1f, 0.9f, 0.4f, -1.0f};
    std::uint64_t rng = 1;
    auto picked = sample_next(logits.data(), logits.size(), 1.0f,
                              std::uint32_t{2}, rng);
    CHECK((picked == 1 || picked == 2));
}
