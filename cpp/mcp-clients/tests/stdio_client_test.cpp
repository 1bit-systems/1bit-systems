#include <doctest/doctest.h>

#include "onebit/mcp_clients/stdio_client.hpp"

#include <deque>
#include <memory>
#include <string>
#include <vector>

using namespace onebit::mcp_clients;

namespace {

struct FakeStdioTransport : StdioTransport {
    std::deque<std::string> reads;
    std::vector<std::string> writes;
    bool closed = false;

    void write_line(std::string_view line) override
    {
        if (closed) throw McpError::closed();
        writes.emplace_back(line);
    }

    std::string read_line() override
    {
        if (reads.empty()) return {};
        auto s = reads.front();
        reads.pop_front();
        return s;
    }

    void shutdown() override { closed = true; }
};

} // namespace

TEST_CASE("stdio round-trips a canned initialize response")
{
    auto fake = std::make_shared<FakeStdioTransport>();
    fake->reads.push_back(
        R"({"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-06-18","capabilities":{},"serverInfo":{"name":"fake","version":"0"}}})");

    StdioClient c(fake);
    json result = c.initialize("halo-test", "0.0.1");
    CHECK(result["protocolVersion"] == "2025-06-18");

    REQUIRE(fake->writes.size() == 1);
    json sent = json::parse(fake->writes[0]);
    CHECK(sent["method"] == "initialize");
    CHECK(sent["params"]["clientInfo"]["name"] == "halo-test");
    CHECK(fake->writes[0].back() == '\n');
}

TEST_CASE("stdio propagates RPC error")
{
    auto fake = std::make_shared<FakeStdioTransport>();
    fake->reads.push_back(
        R"({"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"method not found"}})");

    StdioClient c(fake);
    try {
        c.initialize("halo-test", "0.0.1");
        FAIL("expected throw");
    } catch (const McpError& e) {
        CHECK(e.kind() == ErrorKind::Rpc);
        CHECK(e.code() == -32601);
    }
}

TEST_CASE("stdio empty read_line surface = Closed")
{
    auto fake = std::make_shared<FakeStdioTransport>();
    // No reads queued -> EOF on first read.
    StdioClient c(fake);
    try {
        c.initialize("halo-test", "0.0.1");
        FAIL("expected throw");
    } catch (const McpError& e) {
        CHECK(e.kind() == ErrorKind::Closed);
    }
}

TEST_CASE("stdio list_tools parses tools array")
{
    auto fake = std::make_shared<FakeStdioTransport>();
    fake->reads.push_back(
        R"({"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"a"},{"name":"b"}]}})");

    StdioClient c(fake);
    auto tools = c.list_tools();
    REQUIRE(tools.size() == 2);
    CHECK(tools[0].name == "a");
    CHECK(tools[1].name == "b");
}

TEST_CASE("stdio call_tool emits tools/call frame")
{
    auto fake = std::make_shared<FakeStdioTransport>();
    fake->reads.push_back(
        R"({"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"ok"}],"isError":false}})");

    StdioClient c(fake);
    auto r = c.call_tool("status", json::object());
    CHECK_FALSE(r.is_error);
    REQUIRE(r.content.size() == 1);
    CHECK(r.content[0].as_text() == std::string_view{"ok"});

    REQUIRE(fake->writes.size() == 1);
    json sent = json::parse(fake->writes[0]);
    CHECK(sent["method"] == "tools/call");
    CHECK(sent["params"]["name"] == "status");
}

TEST_CASE("byte-exact framing for canned tools/list request")
{
    auto fake = std::make_shared<FakeStdioTransport>();
    fake->reads.push_back(
        R"({"jsonrpc":"2.0","id":1,"result":{"tools":[]}})");

    StdioClient c(fake);
    (void)c.list_tools();

    REQUIRE(fake->writes.size() == 1);
    // First call uses id=1; tools/list params = {} object; trailing \n.
    // nlohmann/json by default sorts keys alphabetically when dumping.
    const std::string expected =
        R"({"id":1,"jsonrpc":"2.0","method":"tools/list","params":{}})"
        "\n";
    CHECK(fake->writes[0] == expected);
}
