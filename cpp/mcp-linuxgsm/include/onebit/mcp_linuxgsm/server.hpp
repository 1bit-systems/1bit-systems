#pragma once

// 1bit-mcp-linuxgsm — MCP stdio bridge to LinuxGSM.
//
// Each LinuxGSM install exposes a `<game>server` bash driver under a
// dedicated unix user. This server forks those drivers (no shell, args
// never expanded) and returns their stdout/exit code as MCP tool
// results. We allow only known-safe subcommands; `console` and
// `sendcommand` are deliberately excluded.
//
// Discovery: trust $HALO_LINUXGSM_ROOT first, fall back to $HOME/linuxgsm.

#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::mcp_linuxgsm {

using json = nlohmann::json;

inline constexpr std::string_view PROTOCOL_VERSION = "2025-06-18";
inline constexpr std::string_view SERVER_NAME      = "1bit-mcp-linuxgsm";
inline constexpr std::string_view SERVER_VERSION   = "0.1.0";

inline constexpr std::array<std::string_view, 7> ALLOWED_SUBCOMMANDS = {
    "details", "status", "start", "stop", "restart", "update", "backup",
};

[[nodiscard]] bool is_allowed_subcommand(std::string_view sub) noexcept;
[[nodiscard]] bool is_safe_server_name(std::string_view name) noexcept;

[[nodiscard]] std::filesystem::path gsm_root();

[[nodiscard]] json tools();

// Boundary used by tests. Returns (stdout/stderr text, success flag).
struct DriverOutput {
    std::string text;
    bool        success = true;
};

using DriverRunner = std::function<DriverOutput(const std::filesystem::path& driver,
                                                std::string_view subcommand)>;

// Real fork/exec runner — production default.
DriverOutput run_driver_process(const std::filesystem::path& driver,
                                std::string_view subcommand);

// Pure helpers. `runner` defaults to run_driver_process.
[[nodiscard]] json text_result(std::string_view text, bool is_error);
[[nodiscard]] json list_servers(const std::filesystem::path& root);
[[nodiscard]] json run_driver(const std::filesystem::path& root,
                              std::string_view server,
                              std::string_view subcommand,
                              const DriverRunner& runner);

// Single-frame dispatcher. Pure — no I/O.
[[nodiscard]] json handle(const json& request,
                          const std::filesystem::path& root,
                          const DriverRunner& runner);

// Drives stdio JSON-RPC loop until EOF.
void run(std::istream& in, std::ostream& out,
         const std::filesystem::path& root,
         const DriverRunner& runner);

} // namespace onebit::mcp_linuxgsm
