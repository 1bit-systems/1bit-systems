#include <doctest/doctest.h>

#include "onebit/cli/rollback.hpp"

using namespace onebit::cli;

namespace {

struct FakeSnapper : Snapper {
    bool                       avail = true;
    std::vector<SnapperEntry>  rows;
    std::vector<std::uint32_t> rolled;

    bool                                          available() override { return avail; }
    std::expected<std::vector<SnapperEntry>, Error> list() override { return rows; }
    std::expected<void, Error>                     rollback(std::uint32_t n) override
    {
        rolled.push_back(n);
        return {};
    }
};

SnapperEntry mk(std::uint32_t n, std::string desc)
{
    return SnapperEntry{n, std::move(desc)};
}

}  // namespace

TEST_CASE("snapper_absent — diagnostic + no rollback invocation")
{
    FakeSnapper s;
    s.avail = false;
    auto rc = run_with_snapper(s, std::nullopt, true);
    REQUIRE_FALSE(rc.has_value());
    CHECK(rc.error().kind == ErrorKind::PreconditionFailed);
    CHECK(s.rolled.empty());
}

TEST_CASE("auto_pick — takes highest .halo-preinstall number")
{
    FakeSnapper s;
    s.rows = {
        mk(3,  "boot"),
        mk(6,  "7.00 with claude .halo-preinstall"),
        mk(11, "random snapshot"),
        mk(14, "pre-install .halo-preinstall"),
        mk(17, "manual"),
    };
    auto rc = run_with_snapper(s, std::nullopt, true);
    REQUIRE(rc.has_value());
    REQUIRE(s.rolled.size() == 1);
    CHECK(s.rolled.front() == 14);
}

TEST_CASE("auto_pick — bails when no .halo-preinstall candidate")
{
    FakeSnapper s;
    s.rows = { mk(3, "boot"), mk(6, "manual") };
    auto rc = run_with_snapper(s, std::nullopt, true);
    REQUIRE_FALSE(rc.has_value());
    CHECK(rc.error().kind == ErrorKind::NotFound);
    CHECK(s.rolled.empty());
}

TEST_CASE("explicit number overrides auto_pick")
{
    FakeSnapper s;
    s.rows = {
        mk(6,  "7.00 with claude .halo-preinstall"),
        mk(14, "pre-install .halo-preinstall"),
    };
    auto rc = run_with_snapper(s, std::optional<std::uint32_t>{6}, true);
    REQUIRE(rc.has_value());
    REQUIRE(s.rolled.size() == 1);
    CHECK(s.rolled.front() == 6);
}

TEST_CASE("parse_snapper_list — picks up table-shaped rows")
{
    constexpr const char* kSample =
        "# | Type   | Pre # | Date                     | User | Cleanup  | Description           | Userdata\n"
        "--+--------+-------+--------------------------+------+----------+-----------------------+---------\n"
        "0 | single |       | 2026-04-18 10:00:00 UTC  | root |          | current               |\n"
        "6 | single |       | 2026-04-18 10:01:00 UTC  | root | number   | 7.00 with claude .halo-preinstall |\n"
        "14 | pre   |       | 2026-04-22 02:00:00 UTC  | root |          | pre-install .halo-preinstall       |\n";
    auto parsed = parse_snapper_list(kSample);
    bool saw6  = false, saw14 = false;
    for (const auto& e : parsed) {
        if (e.number == 6  && e.description.find(HALO_PREINSTALL_LABEL) != std::string::npos) saw6  = true;
        if (e.number == 14 && e.description.find(HALO_PREINSTALL_LABEL) != std::string::npos) saw14 = true;
    }
    CHECK(saw6);
    CHECK(saw14);
}
