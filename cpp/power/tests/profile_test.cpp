// One TU defines main; the rest just include the header.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/power/profile.hpp"

#include <filesystem>
#include <fstream>

using onebit::power::Profiles;

TEST_CASE("Profiles::parse — inline balanced table")
{
    constexpr const char* src = R"(
[balanced]
stapm_limit = 55000
fast_limit  = 80000
slow_limit  = 70000
tctl_temp   = 90
)";
    auto r = Profiles::parse(src);
    REQUIRE(r.ok());
    const auto& ps = r.value();
    REQUIRE(ps.size() == 1);
    const auto* b = ps.get("balanced");
    REQUIRE(b != nullptr);
    CHECK(b->stapm_limit       == 55000U);
    CHECK(b->fast_limit        == 80000U);
    CHECK(b->slow_limit        == 70000U);
    CHECK(b->tctl_temp         == 90U);
    CHECK_FALSE(b->vrm_current.has_value());
}

TEST_CASE("Profiles::parse — multiple profiles, names() ordered")
{
    constexpr const char* src = R"(
[quiet]
stapm_limit = 30000

[max]
stapm_limit = 120000
)";
    auto r = Profiles::parse(src);
    REQUIRE(r.ok());
    const auto names = r.value().names();
    REQUIRE(names.size() == 2);
    // std::map gives sorted iteration: "max" < "quiet"
    CHECK(names[0] == "max");
    CHECK(names[1] == "quiet");
}

TEST_CASE("Profiles::get — missing profile returns nullptr")
{
    auto r = Profiles::parse("[quiet]\nstapm_limit = 30000\n");
    REQUIRE(r.ok());
    CHECK(r.value().get("nope") == nullptr);
}

TEST_CASE("Profiles::parse — unknown knob is rejected")
{
    constexpr const char* src = R"(
[balanced]
stapm_limit = 55000
mystery_knob = 7
)";
    auto r = Profiles::parse(src);
    REQUIRE_FALSE(r.ok());
    CHECK(r.status().code == onebit::power::Error::ParseError);
    CHECK(r.status().message.find("mystery_knob") != std::string::npos);
}

TEST_CASE("Profiles::parse — non-integer knob is rejected")
{
    constexpr const char* src = R"(
[balanced]
stapm_limit = "fifty-five-watts"
)";
    auto r = Profiles::parse(src);
    REQUIRE_FALSE(r.ok());
    CHECK(r.status().code == onebit::power::Error::ParseError);
}

TEST_CASE("Profiles::parse — negative knob rejected (must fit u32)")
{
    constexpr const char* src = R"(
[balanced]
stapm_limit = -1
)";
    auto r = Profiles::parse(src);
    REQUIRE_FALSE(r.ok());
    CHECK(r.status().code == onebit::power::Error::ParseError);
}

TEST_CASE("Profiles::load — round-trip via tempfile")
{
    namespace fs = std::filesystem;
    fs::path p = fs::temp_directory_path() /
                 ("onebit-power-profile-test-" + std::to_string(::getpid()) + ".toml");
    {
        std::ofstream f{p};
        REQUIRE(f.is_open());
        f << "[balanced]\nstapm_limit = 55000\nfast_limit = 80000\n";
    }
    auto r = Profiles::load(p.string());
    fs::remove(p);
    REQUIRE(r.ok());
    REQUIRE(r.value().get("balanced") != nullptr);
    CHECK(r.value().get("balanced")->stapm_limit == 55000U);
}

TEST_CASE("Profiles::load — missing path is IoError")
{
    auto r = Profiles::load("/nonexistent/abs/path/profiles.toml");
    REQUIRE_FALSE(r.ok());
    CHECK(r.status().code == onebit::power::Error::IoError);
}
