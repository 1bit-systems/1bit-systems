#include <doctest/doctest.h>

#include "onebit/stream/auth.hpp"
#include "onebit/stream/jwt.hpp"

#include <httplib.h>

#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using onebit::stream::AuthConfig;
using onebit::stream::check_admin;
using onebit::stream::check_premium;
using onebit::stream::GateOutcome;

namespace {

[[nodiscard]] httplib::Request bearer_req(const std::string& token)
{
    httplib::Request r;
    r.headers.emplace("Authorization", "Bearer " + token);
    return r;
}

[[nodiscard]] std::span<const std::uint8_t> as_bytes(std::string_view s) noexcept
{
    return {reinterpret_cast<const std::uint8_t*>(s.data()), s.size()};
}

[[nodiscard]] AuthConfig with_secret(const std::string& s)
{
    AuthConfig c;
    c.jwt_secret.assign(s.begin(), s.end());
    return c;
}

} // namespace

TEST_CASE("missing secret => server misconfigured")
{
    AuthConfig c{};
    httplib::Request r;
    r.headers.emplace("Authorization", "Bearer foo");
    CHECK(check_premium(c, r) == GateOutcome::ServerMisconfigured);
}

TEST_CASE("missing header => MissingHeader")
{
    auto             c = with_secret("s");
    httplib::Request r;
    CHECK(check_premium(c, r) == GateOutcome::MissingHeader);
}

TEST_CASE("non-Bearer scheme => BadScheme")
{
    auto             c = with_secret("s");
    httplib::Request r;
    r.headers.emplace("Authorization", "Basic Zm9vOmJhcg==");
    CHECK(check_premium(c, r) == GateOutcome::BadScheme);
}

TEST_CASE("bad token => BadToken")
{
    auto c   = with_secret("test-secret-x");
    auto req = bearer_req("garbage.token.value");
    CHECK(check_premium(c, req) == GateOutcome::BadToken);
}

TEST_CASE("wrong tier => WrongTier")
{
    const std::string secret = "test-secret-x";
    const auto        token  = onebit::stream::jwt::mint_hs256(
        as_bytes(secret),
        R"({"sub":"u","tier":"lossy","iss":"1bit.systems"})");
    auto c   = with_secret(secret);
    auto req = bearer_req(token);
    CHECK(check_premium(c, req) == GateOutcome::WrongTier);
}

TEST_CASE("valid premium token => Allow")
{
    const std::string secret = "test-secret-y";
    const auto        token  = onebit::stream::jwt::mint_hs256(
        as_bytes(secret),
        R"({"sub":"u","tier":"premium","iss":"1bit.systems"})");
    auto c   = with_secret(secret);
    auto req = bearer_req(token);
    CHECK(check_premium(c, req) == GateOutcome::Allow);
}

TEST_CASE("admin bearer compare allows match")
{
    AuthConfig c;
    c.admin_bearer = "abc";
    auto good      = bearer_req("abc");
    auto bad       = bearer_req("nope");
    CHECK(check_admin(c, good) == GateOutcome::Allow);
    CHECK(check_admin(c, bad) == GateOutcome::BadToken);
}

TEST_CASE("admin fails closed when bearer not configured")
{
    // Regression: prior behaviour returned Allow when admin_bearer was empty,
    // leaving /internal/reindex world-callable until the operator set
    // HALO_STREAM_ADMIN_BEARER. Must fail closed instead.
    AuthConfig       c;
    httplib::Request r;
    CHECK(check_admin(c, r) == GateOutcome::ServerMisconfigured);

    auto with_token = bearer_req("anything");
    CHECK(check_admin(c, with_token) == GateOutcome::ServerMisconfigured);
}

TEST_CASE("admin status mapping surfaces 503 on misconfigured")
{
    AuthConfig       c;
    httplib::Request r;
    CHECK(onebit::stream::as_status(check_admin(c, r)) == 503);
}
