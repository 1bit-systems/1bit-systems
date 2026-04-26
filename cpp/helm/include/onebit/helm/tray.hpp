// 1bit-helm — KDE Plasma StatusNotifierItem tray (MVP).
//
// Mirrors crates/1bit-helm/src/tray.rs (pure logic) +
// src/bin/tray.rs (the binary). Live tray uses Qt6::DBus to
// register with `org.kde.StatusNotifierWatcher`; pure helpers
// (service probe + status-line formatter + menu shape) compile
// without any DBus deps so headless tests can exercise them.

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::helm::tray {

// Services the tray controls. Order is stable (used for the status
// line) and matches the Rust `SERVICES` constant.
inline constexpr std::array<std::string_view, 2> SERVICES = {
    "1bit-halo-bitnet",
    "strix-server",
};

enum class Action : std::uint8_t {
    Status,        // Top dimmed status line; not clickable.
    StartAll,      // systemctl --user start every entry in SERVICES.
    StopAll,       // mirror of StartAll.
    RestartServer, // systemctl --user restart strix-server.
    OpenSite,      // xdg-open https://1bit.systems.
    Quit,          // Exits the tray; does NOT stop services.
};

inline constexpr std::array<Action, 6> ACTIONS_ALL = {
    Action::Status,        Action::StartAll, Action::StopAll,
    Action::RestartServer, Action::OpenSite, Action::Quit,
};

[[nodiscard]] std::string_view action_label(Action a) noexcept;

enum class ServiceState : std::uint8_t { Active, Inactive, Unknown };
[[nodiscard]] std::string_view  service_state_str(ServiceState s) noexcept;
[[nodiscard]] ServiceState      parse_service_state(std::string_view s) noexcept;

struct ServiceStatus {
    std::string  name;
    ServiceState state{ServiceState::Unknown};
};

// Format the top-of-menu status line. Compact — Plasma clips long
// menu labels around ~60 chars.
[[nodiscard]] std::string build_status_line(
    const std::vector<ServiceStatus>& rows);

// Live probe of `SERVICES` via `systemctl --user is-active`. One
// subprocess per service; failure → Unknown for that row.
[[nodiscard]] std::vector<ServiceStatus> probe_services();

// Fire-and-forget systemctl helper.
[[nodiscard]] int systemctl(std::string_view              verb,
                            std::vector<std::string_view> units);

// xdg-open https://1bit.systems. Detached; returns true on spawn.
[[nodiscard]] bool open_site();

// 3 s — fast enough to feel snappy on user-initiated actions, slow
// enough not to hammer the dbus socket.
inline constexpr int REFRESH_INTERVAL_MS = 3000;

// Themed icon name registered with Plasma. The base64 PNG below is
// the strict-spec artifact from the Rust crate; production
// installs prefer the themed name.
inline constexpr std::string_view ICON_THEME_NAME = "applications-system";
inline constexpr std::string_view ICON_PNG_PLACEHOLDER_B64 =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

} // namespace onebit::helm::tray
