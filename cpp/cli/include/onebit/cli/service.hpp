#pragma once

// Service registry — the strix-side fixed map of (short name → systemd
// unit, port). Used by `1bit status`, `1bit logs`, `1bit restart`.

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::cli {

struct ServiceEntry {
    const char*    short_name;
    const char*    unit;
    std::uint16_t  port;     ///< 0 = no port
};

[[nodiscard]] std::span<const ServiceEntry> services() noexcept;

struct TimerEntry {
    const char* short_name;
    const char* unit;
};

[[nodiscard]] std::span<const TimerEntry> timers() noexcept;

[[nodiscard]] std::optional<ServiceEntry> resolve_service(std::string_view short_name) noexcept;

// Thin wrappers that shell out via subprocess; return true on success.
[[nodiscard]] bool systemctl_user_active(std::string_view unit);
[[nodiscard]] bool port_listening(std::uint16_t port);

// Subcommand entry points. `journalctl`/`systemctl` are exec'd with
// stdin/stdout inherited — there's no programmatic surface we can mock.
[[nodiscard]] int run_status();
[[nodiscard]] int run_logs(std::string_view service, bool follow, std::uint32_t lines);
[[nodiscard]] int run_restart(std::string_view service);

}  // namespace onebit::cli
