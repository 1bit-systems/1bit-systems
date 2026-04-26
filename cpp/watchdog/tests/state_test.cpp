#include <doctest/doctest.h>

#include "onebit/watchdog/state.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <unistd.h>

using onebit::watchdog::Clock;
using onebit::watchdog::EntryState;
using onebit::watchdog::State;
using onebit::watchdog::Transition;

TEST_CASE("no_change_when_sha_matches_merged")
{
    State s;
    // observe to seed last_seen, then mark_merged so last_merged_sha == abc
    s.observe("x", "abc", 24);
    s.mark_merged("x", Clock::now());
    auto t = s.observe("x", "abc", 24);
    CHECK(t.kind == Transition::Kind::NoChange);
}

TEST_CASE("seen_new_on_first_divergence")
{
    State s;
    auto t = s.observe("x", "abc", 24);
    CHECK(t.kind == Transition::Kind::SeenNew);
}

TEST_CASE("soaking_before_dwell_elapses")
{
    State s;
    s.observe("x", "abc", 24);
    auto t = s.observe("x", "abc", 24);
    CHECK(t.kind == Transition::Kind::Soaking);
}

TEST_CASE("soak_complete_after_window")
{
    State s;
    // First observation arms first_seen_at at t=now-25h
    auto t0 = Clock::now() - std::chrono::hours(25);
    s.observe("x", "abc", 24, t0);
    auto t = s.observe("x", "abc", 24); // now
    CHECK(t.kind == Transition::Kind::SoakComplete);
}

TEST_CASE("reset_clears_dwell_clock")
{
    State s;
    s.observe("x", "abc", 24);
    s.reset("x");
    auto t = s.observe("x", "abc", 24);
    CHECK(t.kind == Transition::Kind::SeenNew);
}

TEST_CASE("mark_merged_records_sha_and_clears_dwell")
{
    State s;
    s.observe("x", "abc", 24);
    s.mark_merged("x", Clock::now());
    const auto& e = s.entries().at("x");
    REQUIRE(e.last_merged_sha.has_value());
    CHECK(*e.last_merged_sha == "abc");
    CHECK_FALSE(e.first_seen_at.has_value());
}

TEST_CASE("save_then_load_round_trip")
{
    auto tmp = std::filesystem::temp_directory_path() /
               ("onebit-watchdog-state-" +
                std::to_string(::getpid()) + ".json");
    {
        State s;
        s.observe("x", "deadbeef", 12);
        s.mark_merged("x", Clock::now());
        s.observe("y", "cafebabe", 12);
        REQUIRE(s.save(tmp.string()));
    }
    auto loaded = State::load(tmp.string());
    REQUIRE(loaded.has_value());
    CHECK(loaded->entries().size() == 2);
    auto it = loaded->entries().find("x");
    REQUIRE(it != loaded->entries().end());
    REQUIRE(it->second.last_merged_sha.has_value());
    CHECK(*it->second.last_merged_sha == "deadbeef");
    std::filesystem::remove(tmp);
}

TEST_CASE("iso8601_round_trip_keeps_microsecond_precision")
{
    using namespace std::chrono;
    auto t = Clock::now();
    auto s = onebit::watchdog::to_iso8601(t);
    auto t2 = onebit::watchdog::from_iso8601(s);
    REQUIRE(t2.has_value());
    auto delta = duration_cast<microseconds>(t - *t2).count();
    if (delta < 0) delta = -delta;
    CHECK(delta < 2); // sub-microsecond drift OK
}
