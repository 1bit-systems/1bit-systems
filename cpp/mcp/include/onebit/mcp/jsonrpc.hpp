#pragma once

// Shared JSON-RPC 2.0 framing helpers used across all 1bit MCP crates
// (server-side and client-side). Header-only, leans on nlohmann/json.
//
// Responsibilities:
//   * build_request(id, method, params) — caller-side request envelope
//   * build_result(id, result)          — server-side success envelope
//   * build_error(id, code, message)    — server-side error envelope
//   * parse_response                    — split result/error from a server reply
//   * encode_line / read_line           — newline-delimited stdio framing
//
// We deliberately avoid pulling in transport code here so this header
// can be used by both the stdio server (cpp/mcp), the stdio client
// (cpp/mcp-clients), and the LinuxGSM server (cpp/mcp-linuxgsm).

#include <nlohmann/json.hpp>

#include <istream>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>

namespace onebit::mcp::jsonrpc {

using json = nlohmann::json;

// Standard JSON-RPC 2.0 error codes.
inline constexpr int kParseError     = -32700;
inline constexpr int kInvalidRequest = -32600;
inline constexpr int kMethodNotFound = -32601;
inline constexpr int kInvalidParams  = -32602;
inline constexpr int kInternalError  = -32603;

[[nodiscard]] inline json build_request(json id, std::string_view method,
                                        std::optional<json> params = std::nullopt)
{
    json out = {
        {"jsonrpc", "2.0"},
        {"id",      std::move(id)},
        {"method",  std::string(method)},
    };
    if (params) {
        out["params"] = std::move(*params);
    }
    return out;
}

[[nodiscard]] inline json build_result(json id, json result)
{
    return {
        {"jsonrpc", "2.0"},
        {"id",      std::move(id)},
        {"result",  std::move(result)},
    };
}

[[nodiscard]] inline json build_error(json id, int code, std::string_view message)
{
    return {
        {"jsonrpc", "2.0"},
        {"id",      std::move(id)},
        {"error",   { {"code", code}, {"message", std::string(message)} }},
    };
}

struct ParsedResponse {
    json id;
    std::optional<json> result;
    std::optional<int>  error_code;
    std::optional<std::string> error_message;
    [[nodiscard]] bool has_error() const noexcept { return error_code.has_value(); }
};

[[nodiscard]] inline ParsedResponse parse_response(const json& v)
{
    ParsedResponse r;
    r.id = v.contains("id") ? v.at("id") : json(nullptr);
    if (v.contains("error") && v.at("error").is_object()) {
        const auto& e = v.at("error");
        r.error_code    = e.value("code", 0);
        r.error_message = e.value("message", std::string{});
    } else if (v.contains("result")) {
        r.result = v.at("result");
    }
    return r;
}

// Newline-delimited frame encoder: dump JSON, append '\n'.
[[nodiscard]] inline std::string encode_line(const json& v)
{
    std::string s = v.dump();
    s.push_back('\n');
    return s;
}

// Read one line of JSON-RPC traffic. Returns nullopt at EOF.
[[nodiscard]] inline std::optional<std::string> read_line(std::istream& in)
{
    std::string line;
    if (!std::getline(in, line)) {
        return std::nullopt;
    }
    return line;
}

inline void write_line(std::ostream& out, const json& v)
{
    out << encode_line(v);
    out.flush();
}

} // namespace onebit::mcp::jsonrpc
