// (main lives in test_main.cpp)
#include <doctest/doctest.h>

#include "onebit/agent/config.hpp"

#include <cstdlib>

using namespace onebit::agent;

namespace {

constexpr const char* kMinimal = R"TOML(
[agent]
brain_url = "http://127.0.0.1:8180"

[adapter]
kind = "stdin"

[memory]
sqlite_path = ":memory:"

[tools]
enabled = ["repo_search"]
)TOML";

constexpr const char* kFull = R"TOML(
[agent]
name             = "halo-helpdesk"
brain_url        = "http://lemond.local:8180"
system_prompt    = "be terse"
model            = "halo-1.58b"
max_history      = 16
max_tool_iters   = 7
request_timeout_ms = 30000
stream           = false
temperature      = 0.4

[adapter]
kind      = "discord"
token     = "${ENV:CFG_TEST_DISCORD_TOKEN}"
bind_host = "0.0.0.0"
bind_port = 8086

[memory]
sqlite_path   = "/tmp/halo.db"
keep_messages = 5000

[tools]
enabled = ["repo_search", "url_fetch"]
)TOML";

} // namespace

TEST_CASE("parse_config: minimal file populates defaults")
{
    auto cfg = parse_config(kMinimal);
    REQUIRE(cfg.has_value());
    CHECK_EQ(cfg->agent.brain_url, "http://127.0.0.1:8180");
    CHECK_EQ(cfg->agent.max_history, 32);          // default
    CHECK_EQ(cfg->agent.max_tool_iters, 5);        // default
    CHECK_EQ(cfg->adapter.kind, "stdin");
    CHECK_EQ(cfg->memory.sqlite_path.string(), ":memory:");
    CHECK_EQ(cfg->tools.enabled.size(), 1u);
}

TEST_CASE("parse_config: full schema round-trips every field")
{
    auto cfg = parse_config(kFull);
    REQUIRE(cfg.has_value());
    CHECK_EQ(cfg->agent.name, "halo-helpdesk");
    CHECK_EQ(cfg->agent.system_prompt, "be terse");
    CHECK_EQ(cfg->agent.model, "halo-1.58b");
    CHECK_EQ(cfg->agent.max_history, 16);
    CHECK_EQ(cfg->agent.max_tool_iters, 7);
    CHECK_EQ(cfg->agent.request_timeout_ms, 30000);
    CHECK_FALSE(cfg->agent.stream);
    CHECK(cfg->agent.temperature == doctest::Approx(0.4));
    CHECK_EQ(cfg->adapter.kind, "discord");
    CHECK_EQ(cfg->adapter.bind_host, "0.0.0.0");
    CHECK_EQ(cfg->adapter.bind_port, 8086);
    CHECK_EQ(cfg->memory.keep_messages, 5000);
    CHECK_EQ(cfg->tools.enabled.size(), 2u);
}

TEST_CASE("parse_config: ${ENV:VAR} substitution lands in token field")
{
    setenv("CFG_TEST_DISCORD_TOKEN", "secret-shh-1234", 1);
    auto cfg = parse_config(kFull);
    REQUIRE(cfg.has_value());
    CHECK_EQ(cfg->adapter.token, "secret-shh-1234");
    unsetenv("CFG_TEST_DISCORD_TOKEN");
}

TEST_CASE("parse_config: missing brain_url is a config error")
{
    constexpr const char* kBad = R"TOML(
[adapter]
kind = "stdin"
[memory]
sqlite_path = ":memory:"
)TOML";
    auto cfg = parse_config(kBad);
    REQUIRE_FALSE(cfg.has_value());
    CHECK(cfg.error().what().find("brain_url") != std::string::npos);
}

TEST_CASE("parse_config: missing sqlite_path is a config error")
{
    constexpr const char* kBad = R"TOML(
[agent]
brain_url = "http://x"
[adapter]
kind = "stdin"
)TOML";
    auto cfg = parse_config(kBad);
    REQUIRE_FALSE(cfg.has_value());
    CHECK(cfg.error().what().find("sqlite_path") != std::string::npos);
}

TEST_CASE("parse_config: garbled TOML surfaces a config error not a crash")
{
    auto cfg = parse_config("this is = = not toml [[[");
    REQUIRE_FALSE(cfg.has_value());
    CHECK(cfg.error().what().find("toml") != std::string::npos);
}

TEST_CASE("parse_config: bind_port out of range rejected")
{
    constexpr const char* kBad = R"TOML(
[agent]
brain_url = "http://x"
[adapter]
kind = "stdin"
bind_port = 99999
[memory]
sqlite_path = ":memory:"
)TOML";
    auto cfg = parse_config(kBad);
    REQUIRE_FALSE(cfg.has_value());
    CHECK(cfg.error().what().find("bind_port") != std::string::npos);
}
