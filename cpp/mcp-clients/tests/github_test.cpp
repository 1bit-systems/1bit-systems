#include <doctest/doctest.h>

#include "onebit/mcp_clients/github.hpp"

#include <cstdlib>
#include <stdlib.h>  // setenv/unsetenv (POSIX)
#include <string>

using namespace onebit::mcp_clients;

TEST_CASE("default endpoint matches official copilot mcp")
{
    CHECK(GITHUB_DEFAULT_ENDPOINT == "https://api.githubcopilot.com/mcp/");
}

TEST_CASE("with_token attaches bearer header")
{
    GitHub gh = GitHub::with_token("https://example/mcp", "ghp_dummy");
    CHECK(gh.inner().headers().at("Authorization") == "Bearer ghp_dummy");
    CHECK(gh.inner().endpoint() == "https://example/mcp");
}

TEST_CASE("from_env returns nullopt when GITHUB_TOKEN unset")
{
    const char* prev = std::getenv("GITHUB_TOKEN");
    std::string saved = prev ? prev : "";
    ::unsetenv("GITHUB_TOKEN");

    auto gh = GitHub::from_env();
    CHECK_FALSE(gh.has_value());

    if (!saved.empty()) ::setenv("GITHUB_TOKEN", saved.c_str(), 1);
}

TEST_CASE("from_env returns Some when GITHUB_TOKEN set")
{
    const char* prev = std::getenv("GITHUB_TOKEN");
    std::string saved = prev ? prev : "";
    ::setenv("GITHUB_TOKEN", "ghp_test", 1);

    auto gh = GitHub::from_env();
    REQUIRE(gh.has_value());
    CHECK(gh->inner().headers().at("Authorization") == "Bearer ghp_test");

    if (saved.empty()) ::unsetenv("GITHUB_TOKEN");
    else               ::setenv("GITHUB_TOKEN", saved.c_str(), 1);
}

TEST_CASE("with_token rejects invalid token (CR/LF)")
{
    CHECK_THROWS_AS(GitHub::with_token("https://x/mcp", "bad\r\ntoken"), McpError);
}
