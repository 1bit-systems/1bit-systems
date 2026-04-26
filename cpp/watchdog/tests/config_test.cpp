#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/watchdog/config.hpp"

using onebit::watchdog::Manifest;
using onebit::watchdog::WatchKind;

TEST_CASE("parses_github_entry")
{
    constexpr auto src = R"(
        [watch.qwen3-tts]
        kind       = "github"
        repo       = "khimaros/qwen3-tts.cpp"
        branch     = "main"
        soak_hours = 24
        on_merge   = [["cmake", "--build", "."]]
        notify     = "discord:halo-updates"
    )";
    auto m = Manifest::from_toml(src);
    REQUIRE(m.has_value());
    CHECK(m->watch.size() == 1);
    auto it = m->watch.find("qwen3-tts");
    REQUIRE(it != m->watch.end());
    CHECK(it->second.repo == "khimaros/qwen3-tts.cpp");
    CHECK(it->second.soak_hours == 24);
    CHECK(it->second.on_merge.size() == 1);
    CHECK(it->second.on_merge[0].size() == 3);
    CHECK(it->second.kind == WatchKind::Github);
    REQUIRE(it->second.branch.has_value());
    CHECK(*it->second.branch == "main");
}

TEST_CASE("parses_huggingface_entry")
{
    constexpr auto src = R"(
        [watch.wan]
        kind    = "huggingface"
        repo    = "Wan-AI/Wan2.2-TI2V-5B"
        on_bump = [["ssh", "runpod", "requant"]]
        notify  = "discord:halo-updates"
    )";
    auto m = Manifest::from_toml(src);
    REQUIRE(m.has_value());
    auto it = m->watch.find("wan");
    REQUIRE(it != m->watch.end());
    CHECK(it->second.soak_hours == 24); // default applied
    CHECK(it->second.kind == WatchKind::Huggingface);
    CHECK(it->second.on_bump.size() == 1);
}

TEST_CASE("ignores_other_sections")
{
    constexpr auto src = R"(
        [component.core]
        description = "core"
        [model.foo]
        description = "m"
        [watch.x]
        kind = "github"
        repo = "a/b"
        notify = ""
    )";
    auto m = Manifest::from_toml(src);
    REQUIRE(m.has_value());
    CHECK(m->watch.size() == 1);
}

TEST_CASE("empty_watch_section_is_ok")
{
    auto m = Manifest::from_toml("[component.foo]\ndescription = \"x\"\n");
    REQUIRE(m.has_value());
    CHECK(m->watch.empty());
}

TEST_CASE("rejects_malformed_toml")
{
    onebit::watchdog::ManifestError err{};
    auto m = Manifest::from_toml("not = valid = toml = ===", &err);
    CHECK_FALSE(m.has_value());
    CHECK(err == onebit::watchdog::ManifestError::ParseFailed);
}

TEST_CASE("watchkind_string_round_trip")
{
    CHECK(onebit::watchdog::to_string(WatchKind::Github)      == "github");
    CHECK(onebit::watchdog::to_string(WatchKind::Huggingface) == "huggingface");
}
