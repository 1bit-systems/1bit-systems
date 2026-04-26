#pragma once

// JSON-RPC 2.0 wire types for the subset of MCP we speak on the client
// side: initialize, tools/list, tools/call. Mirrors the Rust crate's
// protocol.rs.

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::mcp_clients {

using json = nlohmann::json;

// Matches the Rust crate: 2025-06-18 revision.
inline constexpr std::string_view PROTOCOL_VERSION = "2025-06-18";

struct Tool {
    std::string name;
    std::string description;
    json        input_schema;  // serialized as "inputSchema"
};

void to_json(json& j, const Tool& t);
void from_json(const json& j, Tool& t);

enum class ContentBlockKind {
    Text,
    Other,
};

struct ContentBlock {
    ContentBlockKind kind = ContentBlockKind::Other;
    std::string      text;

    [[nodiscard]] std::optional<std::string_view> as_text() const noexcept
    {
        if (kind == ContentBlockKind::Text) return text;
        return std::nullopt;
    }
};

void to_json(json& j, const ContentBlock& b);
void from_json(const json& j, ContentBlock& b);

struct ToolCallResult {
    std::vector<ContentBlock> content;
    bool                      is_error = false;
};

void to_json(json& j, const ToolCallResult& r);
void from_json(const json& j, ToolCallResult& r);

[[nodiscard]] json build_request(std::uint64_t id, std::string_view method,
                                 const std::optional<json>& params);
[[nodiscard]] json initialize_params(std::string_view client_name,
                                     std::string_view client_version);

} // namespace onebit::mcp_clients
