#include <doctest/doctest.h>

#include "onebit/tier_mint/jwt.hpp"

#include "onebit/stream/jwt.hpp"

#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

using onebit::tier_mint::jwt::base_claims;
using onebit::tier_mint::jwt::mint_btcpay;

namespace {

[[nodiscard]] std::span<const std::uint8_t> as_bytes(std::string_view s) noexcept
{
    return {reinterpret_cast<const std::uint8_t*>(s.data()), s.size()};
}

} // namespace

TEST_CASE("mint_btcpay produces a verifiable HS256 token")
{
    const std::string secret = "test-secret-at-least-32-bytes-long-x";
    const auto        token  = mint_btcpay(as_bytes(secret), "inv-abc",
                                           std::chrono::seconds{3600},
                                           "1bit.systems");
    auto v = onebit::stream::jwt::verify_hs256(as_bytes(secret), token, /*now=*/0);
    REQUIRE(v.has_value());
    CHECK(v->tier == "premium");
    CHECK(v->iss == "1bit.systems");
    CHECK(v->jti.has_value());
    CHECK(*v->jti == "inv-abc");
    REQUIRE(v->btcpay_invoice.has_value());
    CHECK(*v->btcpay_invoice == "inv-abc");
}

TEST_CASE("bad secret fails verification")
{
    const auto token = mint_btcpay(as_bytes("right-secret-32-bytes-long-aaa-x"),
                                    "x", std::chrono::seconds{60},
                                    "1bit.systems");
    auto v = onebit::stream::jwt::verify_hs256(
        as_bytes("different-secret-32-bytes-long-y"), token, 0);
    CHECK_FALSE(v.has_value());
}

TEST_CASE("base_claims sets sub == jti == invoice id")
{
    const auto c = base_claims("inv-7", std::chrono::seconds{100}, "issuer-x");
    CHECK(c.sub == "inv-7");
    CHECK(c.jti == "inv-7");
    CHECK(c.btcpay_invoice == "inv-7");
    CHECK(c.iss == "issuer-x");
    CHECK(c.tier == "premium");
    CHECK(c.exp == c.iat + 100);
}

TEST_CASE("expired token is rejected by stream's verify")
{
    const std::string secret = "stub-secret-32-bytes-xxxxxxxxxxxx";
    const auto        token  = mint_btcpay(as_bytes(secret), "x",
                                           std::chrono::seconds{-1},
                                           "1bit.systems");
    auto v = onebit::stream::jwt::verify_hs256(as_bytes(secret), token,
                                               /*now=*/100'000'000'000LL);
    CHECK_FALSE(v.has_value());
}

TEST_CASE("token claims include btcpay_invoice provenance field")
{
    const std::string secret = "p-secret-32-bytes-xxxxxxxxxxxxxxx";
    const auto        token  = mint_btcpay(as_bytes(secret), "inv-prov",
                                           std::chrono::seconds{60}, "i");
    auto v = onebit::stream::jwt::verify_hs256(as_bytes(secret), token, 0);
    REQUIRE(v.has_value());
    REQUIRE(v->btcpay_invoice.has_value());
    CHECK(*v->btcpay_invoice == "inv-prov");
}
