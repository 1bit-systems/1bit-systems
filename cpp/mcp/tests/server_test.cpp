#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/mcp/server.hpp"

#include <sstream>
#include <string>

using onebit::mcp::StdioServer;
using onebit::mcp::ToolRegistry;

TEST_CASE("registry is empty post-cull")
{
    ToolRegistry r;
    CHECK(r.size() == 0);
    CHECK(r.empty());
}

TEST_CASE("server default constructs with empty registry")
{
    StdioServer s;
    CHECK(s.registry().size() == 0);
}

TEST_CASE("tools/list returns empty array")
{
    StdioServer s;
    auto resp = s.handle_line(
        R"({"jsonrpc":"2.0","id":1,"method":"tools/list"})");
    CHECK(resp.find("\"tools\":[]") != std::string::npos);
}

TEST_CASE("initialize includes protocol version + capabilities")
{
    StdioServer s;
    auto resp = s.handle_line(
        R"({"jsonrpc":"2.0","id":7,"method":"initialize"})");
    CHECK(resp.find("\"protocolVersion\":\"2024-11-05\"") != std::string::npos);
    CHECK(resp.find("\"name\":\"1bit-mcp\"") != std::string::npos);
}

TEST_CASE("unknown method returns -32601")
{
    StdioServer s;
    auto resp = s.handle_line(
        R"({"jsonrpc":"2.0","id":3,"method":"frobnicate"})");
    CHECK(resp.find("-32601") != std::string::npos);
    CHECK(resp.find("method not found") != std::string::npos);
}

TEST_CASE("malformed input returns parse error -32700")
{
    StdioServer s;
    auto resp = s.handle_line("not valid json {{{");
    CHECK(resp.find("-32700") != std::string::npos);
    CHECK(resp.find("parse error") != std::string::npos);
}

TEST_CASE("blank input is silently skipped")
{
    StdioServer s;
    CHECK(s.handle_line("").empty());
    CHECK(s.handle_line("   \t\n").empty());
}

TEST_CASE("run loop drains EOF cleanly")
{
    std::istringstream in(
        R"({"jsonrpc":"2.0","id":1,"method":"tools/list"})" "\n"
        R"({"jsonrpc":"2.0","id":2,"method":"initialize"})" "\n");
    std::ostringstream out;
    StdioServer{}.run(in, out);
    const auto s = out.str();
    CHECK(s.find("\"tools\":[]")           != std::string::npos);
    CHECK(s.find("\"protocolVersion\"")    != std::string::npos);
}
