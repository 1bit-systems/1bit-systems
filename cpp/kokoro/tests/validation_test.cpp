// validation_test.cpp — input validation primitives. No engine
// construction needed.

#include <doctest/doctest.h>

#include "onebit/kokoro/engine.hpp"

#include <cmath>

using namespace onebit::kokoro;

TEST_CASE("validate_text rejects empty")
{
    auto r = detail::validate_text("");
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().kind() == ErrorKind::InvalidText);
}

TEST_CASE("validate_text rejects whitespace-only")
{
    for (auto* s : {" ", "\t", "\n", "  \t\n  "}) {
        auto r = detail::validate_text(s);
        REQUIRE_FALSE(r.has_value());
        CHECK(r.error().kind() == ErrorKind::InvalidText);
    }
}

TEST_CASE("validate_text accepts ordinary input")
{
    auto r = detail::validate_text("hello world");
    CHECK(r.has_value());
}

TEST_CASE("validate_text rejects interior NUL")
{
    using namespace std::literals;
    auto r = detail::validate_text("hello\0world"sv);
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().kind() == ErrorKind::InvalidPath);
}

TEST_CASE("validate_voice rejects empty")
{
    auto r = detail::validate_voice("");
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().kind() == ErrorKind::InvalidVoice);
}

TEST_CASE("validate_voice accepts non-empty")
{
    auto r = detail::validate_voice("af_bella");
    CHECK(r.has_value());
}

TEST_CASE("validate_speed rejects out-of-range / non-finite")
{
    const float bad[] = {0.0f, -1.0f, 4.5f, std::nanf(""),
                         std::numeric_limits<float>::infinity()};
    for (float s : bad) {
        auto r = detail::validate_speed(s);
        REQUIRE_FALSE(r.has_value());
        CHECK(r.error().kind() == ErrorKind::InvalidSpeed);
        // Speed payload preserved for non-NaN cases. NaN compares
        // unequal to itself, so just check kind for that one.
        if (std::isfinite(s)) {
            CHECK(r.error().speed() == s);
        }
    }
}

TEST_CASE("validate_speed accepts canonical rate")
{
    CHECK(detail::validate_speed(1.0f).has_value());
    CHECK(detail::validate_speed(0.5f).has_value());
    CHECK(detail::validate_speed(4.0f).has_value());  // boundary inclusive
}

TEST_CASE("kSpeedMin/Max constants stable")
{
    CHECK(kSpeedMin == 0.0f);
    CHECK(kSpeedMax == 4.0f);
}
