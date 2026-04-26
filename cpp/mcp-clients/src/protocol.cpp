#include "onebit/mcp_clients/protocol.hpp"

namespace onebit::mcp_clients {

void to_json(json& j, const Tool& t)
{
    j = {{"name", t.name}, {"description", t.description}};
    if (!t.input_schema.is_null()) {
        j["inputSchema"] = t.input_schema;
    }
}

void from_json(const json& j, Tool& t)
{
    t.name        = j.value("name", std::string{});
    t.description = j.value("description", std::string{});
    if (j.contains("inputSchema")) {
        t.input_schema = j.at("inputSchema");
    } else {
        t.input_schema = nullptr;
    }
}

void to_json(json& j, const ContentBlock& b)
{
    if (b.kind == ContentBlockKind::Text) {
        j = {{"type", "text"}, {"text", b.text}};
    } else {
        j = {{"type", "other"}};
    }
}

void from_json(const json& j, ContentBlock& b)
{
    const std::string type = j.value("type", std::string{});
    if (type == "text") {
        b.kind = ContentBlockKind::Text;
        b.text = j.value("text", std::string{});
    } else {
        b.kind = ContentBlockKind::Other;
        b.text.clear();
    }
}

void to_json(json& j, const ToolCallResult& r)
{
    j = {{"content", r.content}, {"isError", r.is_error}};
}

void from_json(const json& j, ToolCallResult& r)
{
    r.content.clear();
    if (j.contains("content") && j.at("content").is_array()) {
        for (const auto& item : j.at("content")) {
            r.content.push_back(item.get<ContentBlock>());
        }
    }
    r.is_error = j.value("isError", false);
}

json build_request(std::uint64_t id, std::string_view method,
                   const std::optional<json>& params)
{
    json out = {
        {"jsonrpc", "2.0"},
        {"id",      id},
        {"method",  std::string(method)},
        {"params",  params.has_value() ? *params : json(nullptr)},
    };
    return out;
}

json initialize_params(std::string_view client_name, std::string_view client_version)
{
    return {
        {"protocolVersion", std::string(PROTOCOL_VERSION)},
        {"capabilities",    json::object()},
        {"clientInfo",      {
            {"name",    std::string(client_name)},
            {"version", std::string(client_version)},
        }},
    };
}

} // namespace onebit::mcp_clients
