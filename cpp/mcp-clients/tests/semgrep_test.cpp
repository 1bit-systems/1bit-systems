#include <doctest/doctest.h>

#include "onebit/mcp_clients/semgrep.hpp"

using namespace onebit::mcp_clients;

TEST_CASE("default endpoint is hosted semgrep")
{
    CHECK(SEMGREP_DEFAULT_ENDPOINT == "https://mcp.semgrep.ai/mcp");
}

TEST_CASE("default-constructed Semgrep points at hosted endpoint")
{
    Semgrep s;
    CHECK(s.inner().endpoint() == "https://mcp.semgrep.ai/mcp");
}

TEST_CASE("custom endpoint overrides default")
{
    Semgrep s("https://self-hosted/mcp");
    CHECK(s.inner().endpoint() == "https://self-hosted/mcp");
}

TEST_CASE("Semgrep does not attach auth header by default")
{
    Semgrep s;
    CHECK(s.inner().headers().find("Authorization") == s.inner().headers().end());
}

TEST_CASE("Two Semgrep instances do not share state")
{
    Semgrep a;
    Semgrep b("https://other/mcp");
    CHECK(a.inner().endpoint() != b.inner().endpoint());
}
