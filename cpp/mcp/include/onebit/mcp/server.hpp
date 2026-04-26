#pragma once

#include <iosfwd>
#include <string>
#include <string_view>

namespace onebit::mcp {

inline constexpr std::string_view PROTOCOL_VERSION = "2024-11-05";
inline constexpr std::string_view SERVER_NAME      = "1bit-mcp";
inline constexpr std::string_view SERVER_VERSION   = "0.1.0";

// Placeholder tool registry. Empty until re-pointed at GAIA agent-core
// or a successor surface. Mirrors the post-2026-04-25-cull Rust stub.
class ToolRegistry {
public:
    [[nodiscard]] std::size_t size()   const noexcept { return 0; }
    [[nodiscard]] bool        empty()  const noexcept { return true; }
};

// Stdio JSON-RPC server. Reads `\n`-delimited JSON-RPC objects from
// `in`, writes responses to `out`. Returns when EOF reached on `in`.
class StdioServer {
public:
    StdioServer() = default;
    [[nodiscard]] const ToolRegistry& registry() const noexcept { return registry_; }

    // Process a single line of input; returns the JSON response string
    // (with trailing newline) or empty string for blank input.
    [[nodiscard]] std::string handle_line(std::string_view line) const;

    // Drive the read/write loop until EOF on `in`.
    void run(std::istream& in, std::ostream& out) const;

private:
    ToolRegistry registry_{};
};

} // namespace onebit::mcp
