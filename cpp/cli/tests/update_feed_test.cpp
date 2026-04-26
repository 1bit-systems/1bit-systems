#include <doctest/doctest.h>

#include "onebit/cli/update.hpp"

using namespace onebit::cli;

TEST_CASE("is_newer — basic semver order")
{
    CHECK(is_newer("0.3.0", "0.2.9"));
    CHECK(is_newer("0.3.1", "0.3.0"));
    CHECK_FALSE(is_newer("0.3.0", "0.3.0"));
    CHECK_FALSE(is_newer("0.2.9", "0.3.0"));
}

TEST_CASE("is_newer — release sorts above rc pre-release")
{
    CHECK(is_newer("0.3.0", "0.3.0-rc1"));
    CHECK_FALSE(is_newer("0.3.0-rc1", "0.3.0"));
}

TEST_CASE("parse_feed — accepts the documented schema")
{
    constexpr const char* kJson = R"({
        "latest": "0.3.0",
        "channels": { "stable": "0.3.0" },
        "releases": [
          { "version": "0.3.0",
            "date": "2026-04-26",
            "min_compatible": "0.1.0",
            "notes": "first feed cut",
            "artifacts": [
              { "platform": "x86_64-linux-gnu",
                "kind": "appimage",
                "url": "https://1bit.systems/dl/0.3.0/1bit.AppImage",
                "sha256": "deadbeef",
                "minisign_sig": "untrusted comment ...",
                "primary": true } ]
          }
        ]
    })";
    auto feed = parse_feed(kJson);
    REQUIRE(feed.has_value());
    CHECK(feed->latest == "0.3.0");
    CHECK(feed->channels.size() == 1);
    REQUIRE(feed->releases.size() == 1);
    CHECK(feed->releases.front().artifacts.front().sha256 == "deadbeef");
}

TEST_CASE("classify_check — UpToDate when current == feed.latest")
{
    constexpr const char* kJson =
        R"({"latest":"0.3.0","releases":[{"version":"0.3.0","artifacts":[]}]})";
    auto feed = parse_feed(kJson);
    REQUIRE(feed.has_value());
    auto outcome = classify_check(*feed, "0.3.0");
    CHECK(std::holds_alternative<CheckUpToDate>(outcome));
    CHECK(exit_code_for(outcome) == 0);
}

TEST_CASE("classify_check — Available when feed.latest > current AND artifact matches")
{
    constexpr const char* kJson = R"({
        "latest":"0.3.0",
        "releases":[
          {"version":"0.3.0","artifacts":[
             {"platform":"x86_64-linux-gnu","kind":"appimage","url":"x","sha256":""},
             {"platform":"aarch64-linux-gnu","kind":"appimage","url":"y","sha256":""}
          ]}
        ]
    })";
    auto feed = parse_feed(kJson);
    REQUIRE(feed.has_value());
    auto outcome = classify_check(*feed, "0.0.0");
    // current_platform() varies; on the build host it should match one
    // of the artifacts. Either way `Available` or `UpToDate` is a
    // legal classification — make the test machine-agnostic.
    if (current_platform() == "x86_64-linux-gnu" ||
        current_platform() == "aarch64-linux-gnu")
    {
        CHECK(std::holds_alternative<CheckAvailable>(outcome));
        CHECK(exit_code_for(outcome) == 1);
    }
}
