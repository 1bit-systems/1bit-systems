#include <doctest/doctest.h>

#include "onebit/mcp_clients/http.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace onebit::mcp_clients;

namespace {

struct FakeTransport : HttpTransport {
    std::vector<HttpRequest> seen;
    std::vector<HttpResponse> canned;
    std::size_t idx = 0;

    HttpResponse post(const HttpRequest& req) override
    {
        seen.push_back(req);
        if (idx < canned.size()) return canned[idx++];
        return HttpResponse{200, "{}"};
    }
};

} // namespace

TEST_CASE("header rejects invalid name (CR/LF injection)")
{
    HttpClient c("http://localhost:9/mcp");
    CHECK_THROWS_AS(c.add_header("bad name\n", "v"), McpError);
    CHECK(is_valid_header_name("Authorization"));
    CHECK_FALSE(is_valid_header_name(""));
    CHECK_FALSE(is_valid_header_name("bad name"));
    CHECK_FALSE(is_valid_header_value("foo\r\nX-Inject: bar"));
}

TEST_CASE("header accepts bearer + records value")
{
    HttpClient c("http://localhost:9/mcp");
    c.add_header("Authorization", "Bearer xyz");
    CHECK(c.headers().at("Authorization") == "Bearer xyz");
}

TEST_CASE("endpoint preserved")
{
    HttpClient c("https://api.example/mcp");
    CHECK(c.endpoint() == "https://api.example/mcp");
}

TEST_CASE("round_trip serializes a JSON-RPC 2.0 request and parses result")
{
    auto fake = std::make_shared<FakeTransport>();
    fake->canned.push_back({200,
        R"({"jsonrpc":"2.0","id":1,"result":{"tools":[]}})"});

    HttpClient c("http://localhost/mcp", fake);
    json result = c.round_trip("tools/list", json::object());

    REQUIRE(fake->seen.size() == 1);
    json sent = json::parse(fake->seen[0].body);
    CHECK(sent["jsonrpc"] == "2.0");
    CHECK(sent["method"]  == "tools/list");
    CHECK(sent["params"].is_object());
    CHECK(fake->seen[0].headers.at("Content-Type") == "application/json");
    CHECK(fake->seen[0].headers.at("Accept")
          == "application/json, text/event-stream");
    CHECK(result["tools"].is_array());
}

TEST_CASE("round_trip propagates RPC error code + message")
{
    auto fake = std::make_shared<FakeTransport>();
    fake->canned.push_back({200,
        R"({"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"method not found"}})"});

    HttpClient c("http://localhost/mcp", fake);
    try {
        c.round_trip("frobnicate", std::nullopt);
        FAIL("expected throw");
    } catch (const McpError& e) {
        CHECK(e.kind() == ErrorKind::Rpc);
        CHECK(e.code() == -32601);
        CHECK(std::string(e.message()).find("method not found") != std::string::npos);
    }
}

TEST_CASE("round_trip raises Protocol on non-2xx status")
{
    auto fake = std::make_shared<FakeTransport>();
    fake->canned.push_back({500, "{}"});
    HttpClient c("http://localhost/mcp", fake);
    try {
        c.round_trip("tools/list", json::object());
        FAIL("expected throw");
    } catch (const McpError& e) {
        CHECK(e.kind() == ErrorKind::Protocol);
        CHECK(std::string(e.message()).find("500") != std::string::npos);
    }
}

TEST_CASE("list_tools and call_tool round-trip canned payloads")
{
    auto fake = std::make_shared<FakeTransport>();
    fake->canned.push_back({200,
        R"({"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"t","description":"d"}]}})"});
    fake->canned.push_back({200,
        R"({"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"ok"}],"isError":false}})"});

    HttpClient c("http://localhost/mcp", fake);
    auto tools = c.list_tools();
    REQUIRE(tools.size() == 1);
    CHECK(tools[0].name == "t");

    auto r = c.call_tool("t", json::object());
    CHECK_FALSE(r.is_error);
    REQUIRE(r.content.size() == 1);
    CHECK(r.content[0].as_text() == std::string_view{"ok"});

    // Verify ids increment per call.
    json sent0 = json::parse(fake->seen[0].body);
    json sent1 = json::parse(fake->seen[1].body);
    CHECK(sent0["id"].get<std::uint64_t>() == 1);
    CHECK(sent1["id"].get<std::uint64_t>() == 2);
    CHECK(sent1["method"] == "tools/call");
    CHECK(sent1["params"]["name"] == "t");
}
