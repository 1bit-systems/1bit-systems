#include <doctest/doctest.h>

#include "onebit/cli/install_tracker.hpp"

#include <filesystem>
#include <fstream>

using namespace onebit::cli;

TEST_CASE("tracker is empty by default")
{
    InstallTracker t;
    CHECK(t.empty());
    CHECK(t.size() == 0);
}

TEST_CASE("record + actions snapshot")
{
    InstallTracker t;
    t.record(ActionEnabledUnit{"strix-landing.service"});
    t.record(ActionCopiedFile{"/tmp/foo"});
    auto a = t.actions();
    CHECK(a.size() == 2);
    CHECK(t.size() == 2);
    CHECK_FALSE(t.empty());
}

TEST_CASE("anchor10 — best_effort_revert removes copied files in LIFO order")
{
    namespace fs = std::filesystem;
    const auto td = fs::temp_directory_path() / "onebit_cli_tracker";
    fs::create_directories(td);
    const auto a = td / "a.conf";
    const auto b = td / "b.conf";
    const auto c = td / "c.conf";
    std::ofstream(a) << 'a';
    std::ofstream(b) << 'b';
    std::ofstream(c) << 'c';
    REQUIRE(fs::exists(a));
    REQUIRE(fs::exists(b));
    REQUIRE(fs::exists(c));

    InstallTracker t;
    t.record(ActionCopiedFile{a});
    t.record(ActionCopiedFile{b});
    t.record(ActionCopiedFile{c});

    t.best_effort_revert();
    CHECK_FALSE(fs::exists(a));
    CHECK_FALSE(fs::exists(b));
    CHECK_FALSE(fs::exists(c));
    CHECK(t.empty());

    fs::remove_all(td);
}

TEST_CASE("anchor10 — empty tracker revert is a no-op")
{
    InstallTracker t;
    t.best_effort_revert();    // must not crash
    CHECK(t.empty());
}
