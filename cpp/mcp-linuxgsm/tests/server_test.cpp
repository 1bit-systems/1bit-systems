#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/mcp_linuxgsm/server.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
using namespace onebit::mcp_linuxgsm;

namespace {

// Builds a temp dir with two LinuxGSM servers: mcserver (driver
// present) and arkserver (driver missing). Returns the root path.
struct TempRoot {
    fs::path path;
    TempRoot()
    {
        path = fs::temp_directory_path() /
               (std::string("onebit_lgsm_test_") + std::to_string(::getpid()) +
                "_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
        fs::create_directories(path);
        fs::create_directories(path / "mcserver");
        std::ofstream(path / "mcserver" / "mcserverbin") << "stub";
        std::ofstream(path / "mcserver" / "mcserverserver") << "stub";
        ::chmod((path / "mcserver" / "mcserverserver").c_str(), 0755);
        // arkserver has no matching driver
        fs::create_directories(path / "arkserver");
        std::ofstream(path / "arkserver" / "arkserverbin") << "no driver";
    }
    ~TempRoot()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

// Mock runner that records calls but never spawns a real binary.
DriverRunner mock_runner_ok(std::string canned_stdout, bool success = true)
{
    return [out = std::move(canned_stdout), success](const fs::path&, std::string_view) {
        return DriverOutput{out, success};
    };
}

} // namespace

TEST_CASE("allowed subcommands exclude interactive ones (console, sendcommand)")
{
    CHECK_FALSE(is_allowed_subcommand("console"));
    CHECK_FALSE(is_allowed_subcommand("sendcommand"));
    CHECK(is_allowed_subcommand("status"));
    CHECK(is_allowed_subcommand("backup"));
}

TEST_CASE("safe_server_name rejects traversal + spaces + uppercase")
{
    CHECK_FALSE(is_safe_server_name(".."));
    CHECK_FALSE(is_safe_server_name("mc/server"));
    CHECK_FALSE(is_safe_server_name("mc server"));
    CHECK_FALSE(is_safe_server_name(""));
    CHECK_FALSE(is_safe_server_name("McServer"));
    CHECK(is_safe_server_name("mcserver"));
    CHECK(is_safe_server_name("cs2-server"));
    CHECK(is_safe_server_name("ark_server"));
}

TEST_CASE("run_driver rejects disallowed subcommand without invoking runner")
{
    bool called = false;
    DriverRunner spy = [&](const fs::path&, std::string_view) {
        called = true;
        return DriverOutput{"", true};
    };
    json v = run_driver("/nonexistent", "mcserver", "console", spy);
    CHECK_FALSE(called);
    CHECK(v["isError"].get<bool>());
    CHECK(v["content"][0]["text"].get<std::string>().find("not allowed")
          != std::string::npos);
}

TEST_CASE("run_driver rejects unsafe server name")
{
    bool called = false;
    DriverRunner spy = [&](const fs::path&, std::string_view) {
        called = true;
        return DriverOutput{"", true};
    };
    json v = run_driver("/nonexistent", "../etc", "status", spy);
    CHECK_FALSE(called);
    CHECK(v["isError"].get<bool>());
    CHECK(v["content"][0]["text"].get<std::string>().find("invalid server name")
          != std::string::npos);
}

TEST_CASE("run_driver reports missing driver path")
{
    TempRoot root;
    json v = run_driver(root.path, "csgoserver", "status",
                        mock_runner_ok("ignored"));
    CHECK(v["isError"].get<bool>());
    CHECK(v["content"][0]["text"].get<std::string>().find("driver missing")
          != std::string::npos);
}

TEST_CASE("run_driver invokes runner when path + subcommand are valid")
{
    TempRoot root;
    int     calls = 0;
    fs::path seen_driver;
    std::string seen_sub;
    DriverRunner spy = [&](const fs::path& d, std::string_view s) {
        ++calls;
        seen_driver = d;
        seen_sub    = s;
        return DriverOutput{"running", true};
    };
    json v = run_driver(root.path, "mcserver", "status", spy);
    CHECK(calls == 1);
    CHECK(seen_driver == root.path / "mcserver" / "mcserverserver");
    CHECK(seen_sub == "status");
    CHECK_FALSE(v["isError"].get<bool>());
    CHECK(v["content"][0]["text"].get<std::string>() == "running");
}

TEST_CASE("list_servers finds entries that have <name>server driver")
{
    TempRoot root;
    json v = list_servers(root.path);
    CHECK_FALSE(v["isError"].get<bool>());
    const std::string text = v["content"][0]["text"].get<std::string>();
    CHECK(text.find("mcserver") != std::string::npos);
    // arkserver has no matching driver, so absent
    CHECK(text.find("arkserver") == std::string::npos);
}

TEST_CASE("handle initialize advertises server name + protocol")
{
    json req = {{"jsonrpc","2.0"}, {"id",1}, {"method","initialize"}};
    json resp = handle(req, "/tmp", mock_runner_ok(""));
    CHECK(resp["result"]["serverInfo"]["name"] == "1bit-mcp-linuxgsm");
    CHECK(resp["result"]["protocolVersion"] == "2025-06-18");
    CHECK(resp["result"]["capabilities"]["tools"]["listChanged"] == false);
}

TEST_CASE("handle tools/list returns both tools with correct schemas")
{
    json req = {{"jsonrpc","2.0"}, {"id",2}, {"method","tools/list"}};
    json resp = handle(req, "/tmp", mock_runner_ok(""));
    auto& t = resp["result"]["tools"];
    REQUIRE(t.size() == 2);
    CHECK(t[0]["name"] == "linuxgsm_list");
    CHECK(t[1]["name"] == "linuxgsm_run");
    auto& enum_arr = t[1]["inputSchema"]["properties"]["subcommand"]["enum"];
    REQUIRE(enum_arr.is_array());
    CHECK(enum_arr.size() == 7);
    CHECK(t[1]["inputSchema"]["required"][0] == "server");
    CHECK(t[1]["inputSchema"]["required"][1] == "subcommand");
}

TEST_CASE("handle tools/call linuxgsm_list returns text + isError=false")
{
    TempRoot root;
    json req = {
        {"jsonrpc","2.0"}, {"id",3}, {"method","tools/call"},
        {"params", { {"name","linuxgsm_list"}, {"arguments", json::object()} }},
    };
    json resp = handle(req, root.path, mock_runner_ok(""));
    CHECK_FALSE(resp["result"]["isError"].get<bool>());
    CHECK(resp["result"]["content"][0]["type"] == "text");
}

TEST_CASE("handle tools/call linuxgsm_run dispatches to runner mock")
{
    TempRoot root;
    int calls = 0;
    DriverRunner spy = [&](const fs::path&, std::string_view sub) {
        ++calls;
        CHECK(sub == "status");
        return DriverOutput{"online", true};
    };
    json req = {
        {"jsonrpc","2.0"}, {"id",4}, {"method","tools/call"},
        {"params", { {"name","linuxgsm_run"},
                     {"arguments", { {"server","mcserver"}, {"subcommand","status"} }} }},
    };
    json resp = handle(req, root.path, spy);
    CHECK(calls == 1);
    CHECK(resp["result"]["content"][0]["text"] == "online");
}

TEST_CASE("handle tools/call rejects unknown tool with isError=true")
{
    json req = {
        {"jsonrpc","2.0"}, {"id",5}, {"method","tools/call"},
        {"params", { {"name","frobnicate"}, {"arguments", json::object()} }},
    };
    json resp = handle(req, "/tmp", mock_runner_ok(""));
    CHECK(resp["result"]["isError"].get<bool>());
    CHECK(resp["result"]["content"][0]["text"].get<std::string>().find("unknown tool")
          != std::string::npos);
}

TEST_CASE("handle unknown method returns -32601")
{
    json req = {{"jsonrpc","2.0"}, {"id",6}, {"method","frobnicate"}};
    json resp = handle(req, "/tmp", mock_runner_ok(""));
    CHECK(resp["error"]["code"] == -32601);
    CHECK(resp["error"]["message"].get<std::string>().find("method not found")
          != std::string::npos);
}

TEST_CASE("run loop handles multiple frames and reports parse errors")
{
    TempRoot root;
    std::istringstream in(
        R"({"jsonrpc":"2.0","id":1,"method":"tools/list"})" "\n"
        "not valid json\n"
        R"({"jsonrpc":"2.0","id":2,"method":"initialize"})" "\n");
    std::ostringstream out;
    run(in, out, root.path, mock_runner_ok(""));
    const std::string s = out.str();
    CHECK(s.find("\"linuxgsm_list\"")    != std::string::npos);
    CHECK(s.find("-32700")               != std::string::npos);
    CHECK(s.find("\"protocolVersion\"")  != std::string::npos);
}
