#include <doctest/doctest.h>

#include "onebit/cli/preflight.hpp"

#include <string>

using namespace onebit::cli;

namespace {

struct FakeProbe : SystemProbe {
    std::string  kernel    = "6.18.22-1-cachyos-lts";
    bool         rocminfo  = true;
    bool         systemd   = true;
    std::uint64_t disk_gb  = 256;
    std::uint64_t ram_gb   = 128;

    std::string  kernel_release()  override { return kernel; }
    bool         rocminfo_ok()     override { return rocminfo; }
    bool         systemd_user_ok() override { return systemd; }
    std::uint64_t disk_free_gb()   override { return disk_gb; }
    std::uint64_t ram_total_gb()   override { return ram_gb; }
};

}  // namespace

TEST_CASE("kernel 6.18 lts passes")
{
    FakeProbe p;
    auto out = gate_kernel(p);
    CHECK(std::holds_alternative<PreflightPass>(out));
}

TEST_CASE("kernel 7.x fails with kernel-too-new diagnostic")
{
    FakeProbe p;
    p.kernel = "7.0.0-arch1-1";
    auto out = gate_kernel(p);
    REQUIRE(std::holds_alternative<PreflightFail>(out));
    const auto& err = std::get<PreflightFail>(out).err;
    CHECK(err.what.find("Kernel") != std::string::npos);
}

TEST_CASE("kernel unparseable is Skip not Fail")
{
    FakeProbe p;
    p.kernel = "unknown";
    CHECK(std::holds_alternative<PreflightSkip>(gate_kernel(p)));
}

TEST_CASE("disk below 10 GB fails")
{
    FakeProbe p;
    p.disk_gb = 3;
    CHECK(std::holds_alternative<PreflightFail>(gate_disk(p)));
}

TEST_CASE("ram between 64 and 128 is Skip with note")
{
    FakeProbe p;
    p.ram_gb = 96;
    CHECK(std::holds_alternative<PreflightSkip>(gate_ram(p)));
}

TEST_CASE("ram below 64 fails")
{
    FakeProbe p;
    p.ram_gb = 32;
    auto outcome = gate_ram(p);                          // own the value
    REQUIRE(std::holds_alternative<PreflightFail>(outcome));
    const auto& err = std::get<PreflightFail>(outcome).err;
    CHECK(err.expected.find("64") != std::string::npos);
}

TEST_CASE("rocm missing fails")
{
    FakeProbe p;
    p.rocminfo = false;
    CHECK(std::holds_alternative<PreflightFail>(gate_rocm(p)));
}

TEST_CASE("run_all preserves order kernel/rocm/disk/ram/systemd")
{
    FakeProbe p;
    auto rs = run_all(p);
    REQUIRE(rs.size() == 5);
    CHECK(std::string(rs[0].name) == "kernel");
    CHECK(std::string(rs[1].name) == "rocm");
    CHECK(std::string(rs[2].name) == "disk");
    CHECK(std::string(rs[3].name) == "ram");
    CHECK(std::string(rs[4].name) == "systemd");
}

TEST_CASE("systemd absent is Skip not Fail")
{
    FakeProbe p;
    p.systemd = false;
    auto rs = run_all(p);
    CHECK(std::holds_alternative<PreflightSkip>(rs.back().outcome));
}
