#include <doctest/doctest.h>

#include "onebit/mcp_clients/protocol.hpp"

using namespace onebit::mcp_clients;

TEST_CASE("protocol version is 2025-06-18")
{
    CHECK(PROTOCOL_VERSION == "2025-06-18");
}

TEST_CASE("build_request shape matches jsonrpc 2.0")
{
    json v = build_request(7, "tools/list", std::nullopt);
    CHECK(v["jsonrpc"] == "2.0");
    CHECK(v["id"]      == 7);
    CHECK(v["method"]  == "tools/list");
    CHECK(v["params"].is_null());
}

TEST_CASE("build_request includes params when provided")
{
    json p = {{"name", "x"}};
    json v = build_request(2, "tools/call", p);
    CHECK(v["params"]["name"] == "x");
}

TEST_CASE("Tool round-trips loose schema (description default-empty, schema null)")
{
    json raw = json::parse(R"({"name":"x","description":"d"})");
    Tool t = raw.get<Tool>();
    CHECK(t.name == "x");
    CHECK(t.description == "d");
    CHECK(t.input_schema.is_null());
}

TEST_CASE("Tool serialization round-trips inputSchema")
{
    Tool t;
    t.name         = "n";
    t.description  = "d";
    t.input_schema = json::parse(R"({"type":"object"})");
    json j = t;
    CHECK(j["inputSchema"]["type"] == "object");

    Tool back = j.get<Tool>();
    CHECK(back.name == "n");
    CHECK(back.input_schema["type"] == "object");
}

TEST_CASE("ContentBlock text extracts; unknown is Other")
{
    auto t = json::parse(R"({"type":"text","text":"hello"})").get<ContentBlock>();
    CHECK(t.kind == ContentBlockKind::Text);
    CHECK(t.as_text().has_value());
    CHECK(*t.as_text() == "hello");

    auto img = json::parse(R"({"type":"image","data":"..."})").get<ContentBlock>();
    CHECK(img.kind == ContentBlockKind::Other);
    CHECK_FALSE(img.as_text().has_value());
}

TEST_CASE("initialize_params has clientInfo + protocolVersion")
{
    json p = initialize_params("halo-test", "1.2.3");
    CHECK(p["protocolVersion"]      == "2025-06-18");
    CHECK(p["clientInfo"]["name"]   == "halo-test");
    CHECK(p["clientInfo"]["version"] == "1.2.3");
    CHECK(p["capabilities"].is_object());
}

TEST_CASE("ToolCallResult round-trips content + isError")
{
    json raw = json::parse(R"({"content":[{"type":"text","text":"ok"}],"isError":false})");
    ToolCallResult r = raw.get<ToolCallResult>();
    CHECK(r.content.size() == 1);
    CHECK(r.content[0].as_text() == std::string_view{"ok"});
    CHECK_FALSE(r.is_error);
}
