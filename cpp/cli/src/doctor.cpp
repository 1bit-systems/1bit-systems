#include "onebit/cli/doctor.hpp"

#include "onebit/cli/http.hpp"
#include "onebit/cli/proc.hpp"
#include "onebit/cli/service.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>

namespace onebit::cli {

namespace {

[[nodiscard]] std::string read_file_str(const std::filesystem::path& p)
{
    std::ifstream f(p);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' ||
                          s.back() == ' '  || s.back() == '\t')) s.pop_back();
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

[[nodiscard]] std::filesystem::path
read_link_basename(const std::filesystem::path& link)
{
    std::error_code ec;
    auto target = std::filesystem::read_symlink(link, ec);
    if (ec) return {};
    return target.filename();
}

}  // namespace

ProbeResult npu_probe(const std::filesystem::path& root)
{
    const auto accel_dir = root / "sys" / "class" / "accel";
    std::error_code ec;
    if (!std::filesystem::is_directory(accel_dir, ec)) {
        return {DoctorOutcome::Warn,
                "no /sys/class/accel — XDNA2 NPU absent or driver unloaded"};
    }
    struct Hit { std::string name, id; };
    std::vector<Hit> found;
    for (const auto& ent : std::filesystem::directory_iterator(accel_dir, ec)) {
        const auto dev = ent.path() / "device";
        const auto vendor = read_file_str(dev / "vendor");
        const auto device = read_file_str(dev / "device");
        if (vendor == "0x1022" && !device.empty()) {
            found.push_back({ent.path().filename().string(), device});
        }
    }
    const bool mod_loaded =
        std::filesystem::exists(root / "sys" / "module" / "amdxdna", ec);

    if (found.empty()) {
        return {DoctorOutcome::Warn,
                "no AMD accel device in /sys/class/accel (expected on LTS kernel)"};
    }
    if (found.size() == 1) {
        const auto& h = found.front();
        if (mod_loaded) {
            return {DoctorOutcome::Ok,
                    fmt::format("{} device={} amdxdna loaded", h.name, h.id)};
        }
        return {DoctorOutcome::Warn,
                fmt::format("{} device={} but amdxdna module not loaded", h.name, h.id)};
    }
    return {DoctorOutcome::Ok,
            fmt::format("{} AMD accel devices found", found.size())};
}

ProbeResult xe2_probe(const std::filesystem::path& root)
{
    const auto pci = root / "sys" / "bus" / "pci" / "devices";
    std::error_code ec;
    if (!std::filesystem::is_directory(pci, ec)) {
        return {DoctorOutcome::Warn, "no /sys/bus/pci/devices — not a PCI host?"};
    }
    static constexpr std::array<std::string_view, 7> kIds = {
        "0xe20b","0xe202","0xe20c","0xe210","0xe212","0xe215","0xe216",
    };
    for (const auto& ent : std::filesystem::directory_iterator(pci, ec)) {
        const auto vendor = read_file_str(ent.path() / "vendor");
        const auto device = read_file_str(ent.path() / "device");
        if (vendor != "0x8086") continue;
        if (std::find(kIds.begin(), kIds.end(), device) == kIds.end()) continue;
        const auto drv = read_link_basename(ent.path() / "driver").string();
        if (drv == "xe") {
            return {DoctorOutcome::Ok, fmt::format("Battlemage {} bound to xe", device)};
        }
        if (drv == "i915") {
            return {DoctorOutcome::Warn,
                    fmt::format("Battlemage {} bound to i915 — force xe.conf", device)};
        }
        if (drv.empty()) {
            return {DoctorOutcome::Warn,
                    fmt::format("Battlemage {} present, no driver bound", device)};
        }
        return {DoctorOutcome::Warn,
                fmt::format("Battlemage {} bound to {}", device, drv)};
    }
    return {DoctorOutcome::Warn, "no Intel Battlemage device on PCI bus (sliger-only)"};
}

ProbeResult gfx1201_probe(const std::filesystem::path& root)
{
    const auto drm = root / "sys" / "class" / "drm";
    std::error_code ec;
    if (!std::filesystem::is_directory(drm, ec)) {
        return {DoctorOutcome::Warn, "no /sys/class/drm — no DRM subsystem"};
    }
    static constexpr std::array<std::string_view, 3> kIds = {"0x7590","0x7591","0x7592"};
    for (const auto& ent : std::filesystem::directory_iterator(drm, ec)) {
        const auto name = ent.path().filename().string();
        if (!name.starts_with("card")) continue;
        if (name.find('-') != std::string::npos) continue;
        const auto dev = ent.path() / "device";
        const auto vendor = read_file_str(dev / "vendor");
        const auto device = read_file_str(dev / "device");
        if (vendor != "0x1002") continue;
        if (std::find(kIds.begin(), kIds.end(), device) == kIds.end()) continue;
        const auto drv = read_link_basename(dev / "driver").string();
        if (drv == "amdgpu") {
            return {DoctorOutcome::Ok,
                    fmt::format("gfx1201 RX 9070 XT ({}) bound to amdgpu", device)};
        }
        return {DoctorOutcome::Warn,
                fmt::format("gfx1201 {} present, driver={}", device,
                            drv.empty() ? "<none>" : drv)};
    }
    return {DoctorOutcome::Warn, "no gfx1201 (Navi 48) on DRM bus (ryzen host only)"};
}

std::pair<std::uint32_t, std::uint32_t> tally_for_oobe()
{
    std::uint32_t warn = 0, fail = 0;
    auto t = [&](DoctorOutcome o) {
        if (o == DoctorOutcome::Warn) ++warn;
        else if (o == DoctorOutcome::Fail) ++fail;
    };

    // gpu — rocminfo + gfx1151
    {
        auto r = run_capture({"/opt/rocm/bin/rocminfo"});
        if (!r || r->exit_code != 0)        t(DoctorOutcome::Fail);
        else if (r->stdout_text.find("gfx1151") == std::string::npos)
                                            t(DoctorOutcome::Fail);
        else                                t(DoctorOutcome::Ok);
    }
    // kernel
    {
        auto r = run_capture({"uname", "-r"});
        if (!r || r->exit_code != 0)        t(DoctorOutcome::Fail);
        else {
            std::string s = r->stdout_text;
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            std::uint32_t major = 0;
            for (char c : s) {
                if (c < '0' || c > '9') break;
                major = major * 10 + static_cast<std::uint32_t>(c - '0');
            }
            t(major >= 7 ? DoctorOutcome::Warn : DoctorOutcome::Ok);
        }
    }
    // accelerator probes (real fs root)
    const std::filesystem::path root("/");
    t(npu_probe(root).outcome);
    t(xe2_probe(root).outcome);
    t(gfx1201_probe(root).outcome);
    // services
    for (const auto& s : services()) {
        const bool active    = systemctl_user_active(s.unit);
        const bool listening = (s.port == 0) || port_listening(s.port);
        if (!active)             t(DoctorOutcome::Fail);
        else if (!listening)     t(DoctorOutcome::Warn);
        else                     t(DoctorOutcome::Ok);
    }
    return {warn, fail};
}

int run_doctor()
{
    auto [warn, fail] = tally_for_oobe();
    std::cout << warn << " warn, " << fail << " fail\n";
    if (fail > 0) return 2;
    if (warn > 0) return 1;
    return 0;
}

}  // namespace onebit::cli
