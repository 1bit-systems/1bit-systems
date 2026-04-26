#include "onebit/cli/service.hpp"

#include "onebit/cli/proc.hpp"

#include <fmt/core.h>
#include <fmt/format.h>
#include <array>
#include <iostream>
#include <string>

namespace onebit::cli {

namespace {

constexpr std::array<ServiceEntry, 11> kServices = {{
    {"bitnet",   "1bit-halo-bitnet.service",  8080},
    {"strix",    "strix-server.service",      8180},
    {"sd",       "1bit-halo-sd.service",      8081},
    {"whisper",  "1bit-halo-whisper.service", 8082},
    {"kokoro",   "1bit-halo-kokoro.service",  8083},
    {"lemonade", "1bit-halo-lemonade.service",8000},
    {"strix-lm", "strix-lemonade.service",    8200},
    {"landing",  "strix-landing.service",     8190},
    {"burnin",   "strix-burnin.service",      0},
    {"tunnel",   "strix-cloudflared.service", 0},
    {"agent",    "1bit-halo-agent.service",   0},
}};

constexpr std::array<TimerEntry, 4> kTimers = {{
    {"anvil",       "1bit-halo-anvil.timer"},
    {"gh-trio",     "1bit-halo-gh-trio.timer"},
    {"memory-sync", "1bit-halo-memory-sync.timer"},
    {"archive",     "1bit-halo-archive.timer"},
}};

}  // namespace

std::span<const ServiceEntry> services() noexcept
{
    return std::span<const ServiceEntry>(kServices.data(), kServices.size());
}

std::span<const TimerEntry> timers() noexcept
{
    return std::span<const TimerEntry>(kTimers.data(), kTimers.size());
}

std::optional<ServiceEntry> resolve_service(std::string_view short_name) noexcept
{
    for (const auto& s : kServices) {
        if (short_name == s.short_name) return s;
    }
    return std::nullopt;
}

bool systemctl_user_active(std::string_view unit)
{
    auto out = run_capture({
        "systemctl", "--user", "is-active", "--quiet", std::string(unit),
    });
    return out && out->exit_code == 0;
}

bool port_listening(std::uint16_t port)
{
    if (port == 0) return true;
    auto out = run_capture({"ss", "-lnt"});
    if (!out) return false;
    const std::string needle = fmt::format("127.0.0.1:{}", port);
    return out->stdout_text.find(needle) != std::string::npos;
}

int run_status()
{
    std::cout << "─── services (user systemd) ─────────────────\n";
    for (const auto& s : kServices) {
        const bool active    = systemctl_user_active(s.unit);
        const bool listening = port_listening(s.port);
        const char* dot = (active && listening) ? "●" : (active ? "◉" : "○");
        const std::string port_s = (s.port == 0) ? std::string{}
                                                  : fmt::format(":{}", s.port);
        const char* state = active ? (listening ? "active" : "active (no port)") : "inactive";
        std::cout << fmt::format("  {}  {:<10} {:<28} {:<5} {}\n",
                                 dot, s.short_name, s.unit, port_s, state);
    }
    std::cout << "\n─── timers ──────────────────────────────────\n";
    for (const auto& t : kTimers) {
        const bool active = systemctl_user_active(t.unit);
        const char* dot = active ? "●" : "○";
        std::cout << fmt::format("  {}  {:<12} {}\n", dot, t.short_name, t.unit);
    }
    return 0;
}

int run_logs(std::string_view service, bool follow, std::uint32_t lines)
{
    auto svc = resolve_service(service);
    if (!svc) {
        std::cerr << "unknown service '" << service << "'\n";
        return 2;
    }
    std::vector<std::string> argv = {
        "journalctl", "--user", "-u", svc->unit,
        "-n", std::to_string(lines),
    };
    if (follow) argv.emplace_back("-f");
    auto rc = run_inherit(argv);
    if (!rc) {
        std::cerr << rc.error().message << '\n';
        return 1;
    }
    return *rc;
}

int run_restart(std::string_view service)
{
    auto svc = resolve_service(service);
    if (!svc) {
        std::cerr << "unknown service '" << service << "'\n";
        return 2;
    }
    auto rc = run_inherit({"systemctl", "--user", "restart", svc->unit});
    if (!rc) { std::cerr << rc.error().message << '\n'; return 1; }
    if (*rc != 0) {
        std::cerr << "restart failed for " << svc->unit << '\n';
        return *rc;
    }
    std::cout << "✓ " << svc->unit << " restarted\n";
    return 0;
}

}  // namespace onebit::cli
