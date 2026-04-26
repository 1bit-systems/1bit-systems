#include "onebit/mcp/server.hpp"

#include <nlohmann/json.hpp>

#include <istream>
#include <ostream>
#include <string>

namespace onebit::mcp {

namespace {

using json = nlohmann::json;

[[nodiscard]] std::string serialize_with_newline(const json& j)
{
    std::string s = j.dump();
    s.push_back('\n');
    return s;
}

[[nodiscard]] json parse_error_response()
{
    return {
        {"jsonrpc", "2.0"},
        {"id",      nullptr},
        {"error",   { {"code", -32700}, {"message", "parse error"} }},
    };
}

[[nodiscard]] json method_not_found(const json& id)
{
    return {
        {"jsonrpc", "2.0"},
        {"id",      id},
        {"error",   { {"code", -32601}, {"message", "method not found"} }},
    };
}

[[nodiscard]] json initialize_response(const json& id)
{
    return {
        {"jsonrpc", "2.0"},
        {"id",      id},
        {"result",  {
            {"protocolVersion", std::string(PROTOCOL_VERSION)},
            {"capabilities",    { {"tools", json::object()} }},
            {"serverInfo",      {
                {"name",    std::string(SERVER_NAME)},
                {"version", std::string(SERVER_VERSION)},
            }},
        }},
    };
}

[[nodiscard]] json tools_list_response(const json& id)
{
    return {
        {"jsonrpc", "2.0"},
        {"id",      id},
        {"result",  { {"tools", json::array()} }},
    };
}

[[nodiscard]] bool is_blank(std::string_view s) noexcept
{
    for (char c : s) {
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
            return false;
        }
    }
    return true;
}

} // namespace

std::string StdioServer::handle_line(std::string_view line) const
{
    if (is_blank(line)) {
        return {};
    }

    json parsed;
    try {
        parsed = json::parse(line);
    } catch (const json::parse_error&) {
        return serialize_with_newline(parse_error_response());
    }

    const json id     = parsed.contains("id") ? parsed["id"] : json(nullptr);
    const std::string method =
        parsed.contains("method") && parsed["method"].is_string()
            ? parsed["method"].get<std::string>()
            : std::string{};

    if (method == "initialize") {
        return serialize_with_newline(initialize_response(id));
    }
    if (method == "tools/list") {
        return serialize_with_newline(tools_list_response(id));
    }
    return serialize_with_newline(method_not_found(id));
}

void StdioServer::run(std::istream& in, std::ostream& out) const
{
    std::string line;
    while (std::getline(in, line)) {
        const std::string resp = handle_line(line);
        if (!resp.empty()) {
            out << resp;
            out.flush();
        }
    }
}

} // namespace onebit::mcp
