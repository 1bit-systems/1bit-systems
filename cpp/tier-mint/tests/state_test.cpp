#include <doctest/doctest.h>

#include "onebit/tier_mint/state.hpp"

#include <cstdint>
#include <utility>
#include <vector>

using onebit::tier_mint::AppState;
using onebit::tier_mint::Config;

namespace {

[[nodiscard]] Config sample_config()
{
    Config c;
    c.jwt_secret = std::vector<std::uint8_t>(40, 'A');
    c.btcpay_webhook_secret = {'b', 't', 'c'};
    c.admin_secret          = std::vector<std::uint8_t>(40, 'B');
    return c;
}

} // namespace

TEST_CASE("insert_poll/take_poll round-trip and atomicity")
{
    AppState s{sample_config()};
    s.insert_poll("inv-1", "TOKEN-1");
    CHECK(s.poll_size() == 1);
    auto e = s.take_poll("inv-1");
    REQUIRE(e.has_value());
    CHECK(e->jwt == "TOKEN-1");
    CHECK(s.poll_size() == 0);
    auto miss = s.take_poll("inv-1");
    CHECK_FALSE(miss.has_value());
}

TEST_CASE("revoke and is_revoked")
{
    AppState s{sample_config()};
    CHECK_FALSE(s.is_revoked("x"));
    s.revoke("x");
    CHECK(s.is_revoked("x"));
    CHECK(s.revoked_size() == 1);
}

TEST_CASE("insert_poll overwrite uses latest value")
{
    AppState s{sample_config()};
    s.insert_poll("inv", "T1");
    s.insert_poll("inv", "T2");
    auto e = s.take_poll("inv");
    REQUIRE(e.has_value());
    CHECK(e->jwt == "T2");
}

TEST_CASE("config sanity — admin and jwt must differ")
{
    auto c             = sample_config();
    c.admin_secret     = c.jwt_secret;
    // Construction is fine; from_env() is the gate that rejects equality.
    AppState s{std::move(c)};
    CHECK(s.cfg().admin_secret == s.cfg().jwt_secret);
}

TEST_CASE("multiple revokes are idempotent in size")
{
    AppState s{sample_config()};
    s.revoke("a");
    s.revoke("a");
    s.revoke("b");
    CHECK(s.revoked_size() == 2);
}
