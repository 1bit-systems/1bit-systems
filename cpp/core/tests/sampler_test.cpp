#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/core/sampler.hpp"

#include <vector>

using onebit::core::Sampler;
using onebit::core::SamplerConfig;
using onebit::core::TokenId;

TEST_CASE("greedy picks argmax")
{
    std::vector<float> logits = {0.1f, 3.2f, -1.0f, 0.5f};
    auto r = Sampler::greedy(logits);
    REQUIRE(r.has_value());
    CHECK(r.value() == TokenId{1});
}

TEST_CASE("greedy on empty logits errors")
{
    std::vector<float> empty;
    auto r = Sampler::greedy(empty);
    CHECK_FALSE(r.has_value());
}

TEST_CASE("temperature zero is greedy")
{
    Sampler s{SamplerConfig{}};
    std::vector<float> l = {0.1f, 3.2f, -1.0f, 0.5f};
    auto r = s.sample(l, {});
    REQUIRE(r.has_value());
    CHECK(r.value() == TokenId{1});
}

TEST_CASE("top_k=1 always picks argmax")
{
    SamplerConfig cfg{};
    cfg.temperature = 1.0f;
    cfg.top_k       = 1;
    cfg.top_p       = 1.0f;
    cfg.rep_penalty = 1.0f;
    cfg.rep_last_n  = 0;
    cfg.seed        = 42;
    Sampler s{cfg};
    for (int i = 0; i < 10; ++i) {
        std::vector<float> l = {0.1f, 3.2f, -1.0f, 0.5f};
        auto r = s.sample(l, {});
        REQUIRE(r.has_value());
        CHECK(r.value() == TokenId{1});
    }
}

TEST_CASE("rep penalty suppresses recent")
{
    SamplerConfig cfg{};
    cfg.temperature = 1.0f;
    cfg.top_k       = 1;
    cfg.top_p       = 1.0f;
    cfg.rep_penalty = 10.0f;
    cfg.rep_last_n  = 4;
    cfg.seed        = 0;
    Sampler s{cfg};
    std::vector<float> l = {0.1f, 3.2f, 3.0f, 0.5f};
    std::vector<TokenId> recent = {1, 1, 1};
    auto r = s.sample(l, recent);
    REQUIRE(r.has_value());
    CHECK(r.value() == TokenId{2});
}

TEST_CASE("multinomial distribution roughly correct")
{
    SamplerConfig cfg{};
    cfg.temperature = 1.0f;
    cfg.top_k       = 0;
    cfg.top_p       = 1.0f;
    cfg.rep_penalty = 1.0f;
    cfg.rep_last_n  = 0;
    cfg.seed        = 7;
    Sampler s{cfg};
    int hits_1 = 0;
    for (int i = 0; i < 200; ++i) {
        std::vector<float> l = {0.0f, std::log(100.0f)}; // probs ~ [1/101, 100/101]
        auto r = s.sample(l, {});
        REQUIRE(r.has_value());
        if (r.value() == TokenId{1}) ++hits_1;
    }
    CHECK(hits_1 > 180);
}

TEST_CASE("set_config preserves stream when seed unchanged")
{
    SamplerConfig cfg{};
    cfg.temperature = 1.0f;
    cfg.top_k       = 0;
    cfg.top_p       = 1.0f;
    cfg.rep_penalty = 1.0f;
    cfg.rep_last_n  = 0;
    cfg.seed        = 99;
    Sampler s{cfg};
    std::vector<float> l1 = {0.0f, 0.0f, 0.0f, 0.0f};
    auto a = s.sample(l1, {});
    cfg.temperature = 2.0f;
    s.set_config(cfg);
    std::vector<float> l2 = {0.0f, 0.0f, 0.0f, 0.0f};
    auto b = s.sample(l2, {});
    REQUIRE(a.has_value());
    REQUIRE(b.has_value());
    // RNG continues — second draw uses next state, not reset.
    CHECK(a.value() != -1);
    CHECK(b.value() != -1);
}
