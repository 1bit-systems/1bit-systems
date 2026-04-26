#include "onebit/cli/preflight.hpp"

#include "onebit/cli/proc.hpp"

#include <fmt/core.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/statvfs.h>

namespace onebit::cli {

namespace {

[[nodiscard]] std::optional<std::uint32_t> first_int(std::string_view s) noexcept
{
    std::size_t i = 0;
    while (i < s.size() && (s[i] < '0' || s[i] > '9')) ++i;
    if (i == s.size()) return std::nullopt;
    std::uint32_t v = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
        v = v * 10 + static_cast<std::uint32_t>(s[i] - '0');
        ++i;
    }
    return v;
}

}  // namespace

// ───── RealProbe ─────────────────────────────────────────────────────────

std::string RealProbe::kernel_release()
{
    auto out = run_capture({"uname", "-r"});
    if (!out || out->exit_code != 0) return "unknown";
    std::string s = out->stdout_text;
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

bool RealProbe::rocminfo_ok()
{
    auto out = run_capture({"rocminfo"});
    return out && out->exit_code == 0;
}

bool RealProbe::systemd_user_ok()
{
    auto out = run_capture({"systemctl", "--user", "is-system-running"});
    if (!out) return false;
    std::string s = out->stdout_text;
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s == "running" || s == "degraded" || s == "starting";
}

std::uint64_t RealProbe::disk_free_gb()
{
    struct statvfs st {};
    if (::statvfs("/", &st) != 0) return 0;
    const std::uint64_t bytes =
        static_cast<std::uint64_t>(st.f_bavail) *
        static_cast<std::uint64_t>(st.f_frsize);
    return bytes / (1024ULL * 1024ULL * 1024ULL);
}

std::uint64_t RealProbe::ram_total_gb()
{
    std::ifstream mem("/proc/meminfo");
    if (!mem) return 0;
    std::string line;
    while (std::getline(mem, line)) {
        if (line.starts_with("MemTotal:")) {
            // "MemTotal:       131072 kB"
            std::istringstream is(line.substr(9));
            std::uint64_t kb = 0;
            is >> kb;
            return kb / (1024ULL * 1024ULL);
        }
    }
    return 0;
}

// ───── gates ─────────────────────────────────────────────────────────────

PreflightOutcome gate_kernel(SystemProbe& probe)
{
    const auto rel = probe.kernel_release();
    const auto major = first_int(rel);
    if (!major) {
        return PreflightSkip{fmt::format("kernel release unparseable: \"{}\"", rel)};
    }
    if (*major == 6) {
        return PreflightPass{fmt::format("kernel {} (LTS OK)", rel)};
    }
    if (*major == 7) {
        return PreflightFail{OobeError::kernel_too_new(rel)};
    }
    return PreflightSkip{
        fmt::format("kernel {} (major {}) is outside the tested range — proceeding anyway",
                    rel, *major)
    };
}

PreflightOutcome gate_rocm(SystemProbe& probe)
{
    if (probe.rocminfo_ok()) {
        return PreflightPass{"rocminfo reachable"};
    }
    return PreflightFail{OobeError::rocm_missing()};
}

PreflightOutcome gate_disk(SystemProbe& probe)
{
    const auto free = probe.disk_free_gb();
    if (free >= 10) {
        return PreflightPass{fmt::format("{} GB free on /", free)};
    }
    return PreflightFail{OobeError::disk_too_small(free)};
}

PreflightOutcome gate_ram(SystemProbe& probe)
{
    const auto ram   = probe.ram_total_gb();
    const auto floor = default_ram_floor_gb();
    if (ram < floor) {
        return PreflightFail{OobeError::ram_too_small(ram, floor)};
    }
    if (ram < 128) {
        return PreflightSkip{
            fmt::format("RAM {} GB < 128 GB recommended — halo-v2 Q4_K_M will be tight",
                        ram)
        };
    }
    return PreflightPass{fmt::format("RAM {} GB", ram)};
}

std::vector<GateResult> run_all(SystemProbe& probe)
{
    std::vector<GateResult> out;
    out.reserve(5);
    out.push_back(GateResult{"kernel",  gate_kernel(probe)});
    out.push_back(GateResult{"rocm",    gate_rocm(probe)});
    out.push_back(GateResult{"disk",    gate_disk(probe)});
    out.push_back(GateResult{"ram",     gate_ram(probe)});
    if (probe.systemd_user_ok()) {
        out.push_back(GateResult{"systemd", PreflightPass{"user bus reachable"}});
    } else {
        out.push_back(GateResult{"systemd",
            PreflightSkip{
                "systemd --user bus not reachable (containers / CI are OK here)"
            }});
    }
    return out;
}

}  // namespace onebit::cli
