#include <doctest/doctest.h>

#include <unistd.h>

#include "onebit/power/ec.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>

using onebit::power::CurveDir;
using onebit::power::EcBackend;
using onebit::power::parse_curve_csv;

namespace fs = std::filesystem;

namespace {

struct FakeSysfs {
    fs::path root;

    FakeSysfs()
    {
        root = fs::temp_directory_path() /
               ("onebit-power-ec-test-" +
                std::to_string(::getpid()) + "-" +
                std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        fs::create_directories(root);

        for (int i = 1; i <= 3; ++i) {
            fs::path fan = root / ("fan" + std::to_string(i));
            fs::create_directories(fan);
            std::ofstream{fan / "rpm"}            << "0\n";
            std::ofstream{fan / "mode"}           << "curve\n";
            std::ofstream{fan / "level"}          << "0\n";
            std::ofstream{fan / "rampup_curve"}   << "60,70,83,95,97\n";
            std::ofstream{fan / "rampdown_curve"} << "40,50,80,94,96\n";
        }

        fs::create_directories(root / "temp1");
        std::ofstream{root / "temp1" / "temp"} << "42\n";

        fs::create_directories(root / "apu");
        std::ofstream{root / "apu" / "power_mode"} << "balanced\n";
    }

    ~FakeSysfs()
    {
        std::error_code ec;
        fs::remove_all(root, ec);
    }

    FakeSysfs(const FakeSysfs&)            = delete;
    FakeSysfs& operator=(const FakeSysfs&) = delete;
};

} // namespace

TEST_CASE("EcBackend::available true when apu/power_mode exists")
{
    FakeSysfs s;
    EcBackend be{s.root};
    CHECK(be.available());
}

TEST_CASE("EcBackend::available false on empty tempdir")
{
    auto p = fs::temp_directory_path() /
             ("onebit-power-empty-" + std::to_string(::getpid()));
    fs::create_directories(p);
    EcBackend be{p};
    CHECK_FALSE(be.available());
    fs::remove_all(p);
}

TEST_CASE("EcBackend reads temp_c and power_mode")
{
    FakeSysfs s;
    EcBackend be{s.root};
    auto t = be.temp_c();
    REQUIRE(t.ok());
    CHECK(t.value() == 42);
    auto m = be.power_mode();
    REQUIRE(m.ok());
    CHECK(m.value() == "balanced");
}

TEST_CASE("EcBackend::fan parses curve")
{
    FakeSysfs s;
    EcBackend be{s.root};
    auto f = be.fan(1);
    REQUIRE(f.ok());
    CHECK(f.value().id == 1);
    CHECK(f.value().mode == "curve");
    REQUIRE(f.value().rampup.size() == 5);
    CHECK(f.value().rampup[0] == 60);
    CHECK(f.value().rampup[4] == 97);
}

TEST_CASE("EcBackend::set_power_mode round-trip")
{
    FakeSysfs s;
    EcBackend be{s.root};
    REQUIRE(be.set_power_mode("performance").ok());
    auto m = be.power_mode();
    REQUIRE(m.ok());
    CHECK(m.value() == "performance");
}

TEST_CASE("EcBackend::set_power_mode rejects unknown mode")
{
    FakeSysfs s;
    EcBackend be{s.root};
    auto r = be.set_power_mode("turbo");
    CHECK_FALSE(r.ok());
    CHECK(r.code == onebit::power::Error::InvalidArgument);
}

TEST_CASE("EcBackend::fan rejects out-of-range id")
{
    FakeSysfs s;
    EcBackend be{s.root};
    CHECK_FALSE(be.fan(0).ok());
    CHECK_FALSE(be.fan(4).ok());
}

TEST_CASE("EcBackend::set_fan_level clamps to 0..=5")
{
    FakeSysfs s;
    EcBackend be{s.root};
    CHECK(be.set_fan_level(1, 3).ok());
    CHECK_FALSE(be.set_fan_level(1, 6).ok());
}

TEST_CASE("EcBackend::set_fan_curve writes csv and reads back")
{
    FakeSysfs s;
    EcBackend be{s.root};
    REQUIRE(be.set_fan_curve(2, CurveDir::Rampup, {30, 40, 60, 80, 95}).ok());
    auto f = be.fan(2);
    REQUIRE(f.ok());
    REQUIRE(f.value().rampup.size() == 5);
    CHECK(f.value().rampup[0] == 30);
    CHECK(f.value().rampup[4] == 95);
}

TEST_CASE("EcBackend::snapshot gathers all fans + temp + mode")
{
    FakeSysfs s;
    EcBackend be{s.root};
    auto snap = be.snapshot();
    CHECK(snap.fans.size() == 3);
    CHECK(snap.temp_c == 42);
    CHECK(snap.power_mode.has_value());
    CHECK(*snap.power_mode == "balanced");
}

TEST_CASE("parse_curve_csv tolerates trailing newline + spaces")
{
    auto v = parse_curve_csv(" 60, 70 ,83,95 , 97 \n");
    REQUIRE(v.size() == 5);
    CHECK(v[0] == 60);
    CHECK(v[4] == 97);
}

TEST_CASE("parse_curve_csv silently drops bad tokens")
{
    auto v = parse_curve_csv("60,oops,83,95,97");
    // Bad token simply skipped — same tolerance as Rust filter_map+parse.
    CHECK(v.size() == 4);
}
