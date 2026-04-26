#include <doctest/doctest.h>

#include "onebit/cli/burnin.hpp"

#include <atomic>
#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace onebit::cli;

namespace {

std::atomic<std::uint64_t> g_seq{0};

std::filesystem::path write_jsonl(const std::vector<std::string>& lines)
{
    namespace fs = std::filesystem;
    const auto p = fs::temp_directory_path() /
        ("onebit_cli_burnin_" + std::to_string(::getpid()) + "_" +
         std::to_string(g_seq.fetch_add(1)) + ".jsonl");
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    for (const auto& l : lines) out << l << '\n';
    return p;
}

std::string row(int i, std::string_view ts, std::uint32_t idx, bool pass,
                std::uint32_t off, std::string_view snippet,
                std::string_view v1, std::string_view v2)
{
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        R"({"ts":"%s","prompt_idx":%u,"prompt_snippet":"%s","prefix_match_chars":%u,"full_match":%s,"v1_ms":%d,"v2_ms":%d,"v1_text":"%s","v2_text":"%s"})",
        std::string(ts).c_str(), idx, std::string(snippet).c_str(), off,
        pass ? "true" : "false", 100 + i, 100 + i,
        std::string(v1).c_str(), std::string(v2).c_str());
    return buf;
}

}  // namespace

TEST_CASE("compute_stats on 7-pass / 3-fail fixture reports 70%")
{
    auto p = write_jsonl({
        row(0, "2026-04-20T00:00:00Z", 0, true,  10, "capital of France", " Paris.", " Paris."),
        row(1, "2026-04-20T00:10:00Z", 1, true,  12, "2+2", " 4.", " 4."),
        row(2, "2026-04-20T01:00:00Z", 7, false, 0,  "gold symbol", "1", "0"),
        row(3, "2026-04-20T02:00:00Z", 7, false, 0,  "gold symbol", "1", "0"),
        row(4, "2026-04-20T03:00:00Z", 2, true,  20, "planets",  " Jupiter", " Jupiter"),
        row(5, "2026-04-20T04:00:00Z", 7, false, 0,  "gold symbol", "1", "0"),
        row(6, "2026-04-20T05:00:00Z", 3, true,  5,  "Hamlet",   " wrote", " wrote"),
        row(7, "2026-04-20T06:00:00Z", 4, true,  8,  "horror",   " silent", " silent"),
        row(8, "2026-04-20T07:00:00Z", 5, true,  3,  "clouds",   " drift",  " drift"),
        row(9, "2026-04-20T08:00:00Z", 6, true,  9,  "poem",     " soft",   " soft"),
    });
    auto rows = load_rows(p);
    REQUIRE(rows.has_value());
    CHECK(rows->size() == 10);
    auto s = compute_stats(*rows);
    CHECK(s.total == 10);
    CHECK(s.pass  == 7);
    CHECK(s.fail  == 3);
    CHECK(s.pct   == doctest::Approx(70.0));
    std::filesystem::remove(p);
}

TEST_CASE("compute_drift returns the highest-failing prompt first")
{
    auto p = write_jsonl({
        row(0, "2026-04-20T00:00:00Z", 7, false, 0, "x", "1", "0"),
        row(1, "2026-04-20T00:01:00Z", 7, false, 0, "x", "1", "0"),
        row(2, "2026-04-20T00:02:00Z", 7, false, 0, "x", "1", "0"),
        row(3, "2026-04-20T00:03:00Z", 9, false, 5, "y", "a", "b"),
    });
    auto rows = load_rows(p);
    REQUIRE(rows.has_value());
    auto d = compute_drift(*rows, 10);
    REQUIRE(d.size() == 2);
    CHECK(d[0].prompt_idx == 7);
    CHECK(d[0].fail_count == 3);
    CHECK(d[0].typical_offset == 0);
    std::filesystem::remove(p);
}

TEST_CASE("filter_since cuts strictly by ISO8601 lex compare")
{
    auto p = write_jsonl({
        row(0, "2026-04-20T00:00:00Z", 0, true, 1, "a", "x", "x"),
        row(1, "2026-04-20T05:00:00Z", 1, true, 1, "b", "y", "y"),
        row(2, "2026-04-20T08:00:00Z", 2, true, 1, "c", "z", "z"),
    });
    auto rows = load_rows(p);
    REQUIRE(rows.has_value());
    auto sliced = filter_since(*rows, "2026-04-20T05:00:00Z");
    CHECK(sliced.size() == 2);
    auto all = filter_since(*rows, "2020-01-01T00:00:00Z");
    CHECK(all.size() == 3);
    auto none = filter_since(*rows, "2030-01-01T00:00:00Z");
    CHECK(none.empty());
    std::filesystem::remove(p);
}

TEST_CASE("load_rows skips blank and malformed lines")
{
    auto p = write_jsonl({
        row(0, "2026-04-20T00:00:00Z", 0, true, 1, "a", "x", "x"),
        "",
        "not json at all",
        row(1, "2026-04-20T00:01:00Z", 1, false, 0, "b", "y", "z"),
    });
    auto rows = load_rows(p);
    REQUIRE(rows.has_value());
    CHECK(rows->size() == 2);
    std::filesystem::remove(p);
}
