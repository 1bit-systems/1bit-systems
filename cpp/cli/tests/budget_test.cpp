#include <doctest/doctest.h>

#include "onebit/cli/budget.hpp"

using namespace onebit::cli;

TEST_CASE("parse_meminfo_kib returns kB for known keys")
{
    constexpr const char* kReal =
        "MemTotal:       131000000 kB\n"
        "MemAvailable:   120000000 kB\n";
    auto t = parse_meminfo_kib(kReal, "MemTotal");
    REQUIRE(t.has_value());
    CHECK(*t == 131'000'000);

    auto a = parse_meminfo_kib(kReal, "MemAvailable");
    REQUIRE(a.has_value());
    CHECK(*a == 120'000'000);

    auto missing = parse_meminfo_kib(kReal, "Nope");
    CHECK_FALSE(missing.has_value());
}

TEST_CASE("fmt_bytes — picks reasonable units")
{
    CHECK(fmt_bytes(0) == "0 B");
    CHECK(fmt_bytes(1023) == "1023 B");
    CHECK(fmt_bytes(1024 * 1024).starts_with("1.0 MB"));
    CHECK(fmt_bytes(3ULL * 1024 * 1024 * 1024).starts_with("3.0 GB"));
}

TEST_CASE("looks_like_halo_service — heuristic covers known names")
{
    CHECK(looks_like_halo_service("bitnet_decode"));
    CHECK(looks_like_halo_service("whisper-server"));
    CHECK(looks_like_halo_service("halo-landing"));
    CHECK(looks_like_halo_service("1bit-mcp"));
    CHECK_FALSE(looks_like_halo_service("chrome"));
    CHECK_FALSE(looks_like_halo_service("kwin_wayland"));
}

TEST_CASE("budget_for_next_model — floors at 0 when ram less than reserve")
{
    BudgetSnapshot s;
    s.gtt_total     = 64ULL * 1024 * 1024 * 1024;
    s.gtt_used      = 60ULL * 1024 * 1024 * 1024;
    s.mem_total     = 128ULL * 1024 * 1024 * 1024;
    s.mem_available = 1ULL * 1024 * 1024 * 1024;  // 1 GB < 4 GB reserve
    CHECK(s.budget_for_next_model() == 0);
}

TEST_CASE("budget_for_next_model — returns min(GTT free, RAM avail - reserve)")
{
    BudgetSnapshot s;
    s.gtt_total     = 64ULL * 1024 * 1024 * 1024;
    s.gtt_used      = 24ULL * 1024 * 1024 * 1024;     // 40 GB free
    s.mem_total     = 128ULL * 1024 * 1024 * 1024;
    s.mem_available = 32ULL * 1024 * 1024 * 1024;     // 32 - 4 = 28 GB
    CHECK(s.budget_for_next_model() == 28ULL * 1024 * 1024 * 1024);
}
