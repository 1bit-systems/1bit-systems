#include <doctest/doctest.h>

#include "onebit/tier_mint/state.hpp"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

using onebit::tier_mint::AppState;
using onebit::tier_mint::Config;

namespace {

struct EnvScope {
    // RAII overlay over the four envvars Config::from_env() reads. Restores
    // (unsets) them on destruction so cases don't leak state into each other.
    EnvScope()
    {
        ::unsetenv("HALO_TIER_HMAC_SECRET");
        ::unsetenv("HALO_BTCPAY_WEBHOOK_SECRET");
        ::unsetenv("HALO_TIER_ADMIN_SECRET");
    }
    ~EnvScope()
    {
        ::unsetenv("HALO_TIER_HMAC_SECRET");
        ::unsetenv("HALO_BTCPAY_WEBHOOK_SECRET");
        ::unsetenv("HALO_TIER_ADMIN_SECRET");
    }
    EnvScope(const EnvScope&)            = delete;
    EnvScope& operator=(const EnvScope&) = delete;

    static void set(const char* k, const std::string& v)
    {
        ::setenv(k, v.c_str(), 1);
    }
};

constexpr const char* kHmac  = "HALO_TIER_HMAC_SECRET";
constexpr const char* kBtc   = "HALO_BTCPAY_WEBHOOK_SECRET";
constexpr const char* kAdmin = "HALO_TIER_ADMIN_SECRET";

[[nodiscard]] std::string repeat(char ch, std::size_t n) { return std::string(n, ch); }

[[nodiscard]] Config sample_config()
{
    Config c;
    c.jwt_secret            = std::vector<std::uint8_t>(40, 'A');
    c.btcpay_webhook_secret = std::vector<std::uint8_t>(40, 'C');
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

TEST_CASE("from_env: btcpay secret shorter than 32 bytes is rejected")
{
    EnvScope env;
    EnvScope::set(kHmac,  repeat('A', 40));
    EnvScope::set(kBtc,   repeat('C', 16)); // too short
    EnvScope::set(kAdmin, repeat('B', 40));
    auto r = Config::from_env();
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().find("HALO_BTCPAY_WEBHOOK_SECRET") != std::string::npos);
    CHECK(r.error().find("too short") != std::string::npos);
}

TEST_CASE("from_env: btcpay secret may not equal jwt secret")
{
    EnvScope env;
    const auto shared = repeat('Z', 40);
    EnvScope::set(kHmac,  shared);
    EnvScope::set(kBtc,   shared); // collides with jwt
    EnvScope::set(kAdmin, repeat('B', 40));
    auto r = Config::from_env();
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().find("HALO_BTCPAY_WEBHOOK_SECRET") != std::string::npos);
    CHECK(r.error().find("HALO_TIER_HMAC_SECRET") != std::string::npos);
}

TEST_CASE("from_env: btcpay secret may not equal admin secret")
{
    EnvScope env;
    const auto shared = repeat('Q', 40);
    EnvScope::set(kHmac,  repeat('A', 40));
    EnvScope::set(kBtc,   shared);
    EnvScope::set(kAdmin, shared); // collides with admin
    auto r = Config::from_env();
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().find("HALO_BTCPAY_WEBHOOK_SECRET") != std::string::npos);
    CHECK(r.error().find("HALO_TIER_ADMIN_SECRET") != std::string::npos);
}

TEST_CASE("from_env: admin secret may not equal jwt secret")
{
    EnvScope env;
    const auto shared = repeat('Y', 40);
    EnvScope::set(kHmac,  shared);
    EnvScope::set(kBtc,   repeat('C', 40));
    EnvScope::set(kAdmin, shared); // collides with jwt
    auto r = Config::from_env();
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().find("HALO_TIER_ADMIN_SECRET") != std::string::npos);
}

TEST_CASE("from_env: three distinct, sufficiently long secrets accept")
{
    EnvScope env;
    EnvScope::set(kHmac,  repeat('A', 40));
    EnvScope::set(kBtc,   repeat('C', 40));
    EnvScope::set(kAdmin, repeat('B', 40));
    auto r = Config::from_env();
    REQUIRE(r.has_value());
    CHECK(r->jwt_secret.size() == 40);
    CHECK(r->btcpay_webhook_secret.size() == 40);
    CHECK(r->admin_secret.size() == 40);
}
