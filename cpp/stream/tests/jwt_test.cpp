#include <doctest/doctest.h>

#include "onebit/stream/jwt.hpp"

#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using onebit::stream::jwt::base64url;
using onebit::stream::jwt::base64url_decode;
using onebit::stream::jwt::hmac_sha256;
using onebit::stream::jwt::mint_hs256;
using onebit::stream::jwt::verify_hs256;

namespace {

[[nodiscard]] std::span<const std::uint8_t> as_bytes(std::string_view s) noexcept
{
    return std::span<const std::uint8_t>{
        reinterpret_cast<const std::uint8_t*>(s.data()), s.size()};
}

} // namespace

TEST_CASE("hmac-sha256 RFC 4231 case 1 (key 0x0b*20, data='Hi There')")
{
    std::vector<std::uint8_t>      key(20, 0x0b);
    const std::string_view         msg = "Hi There";
    auto                           h   = hmac_sha256(key, as_bytes(msg));
    static const std::uint8_t expected[] = {
        0xb0, 0x34, 0x4c, 0x61, 0xd8, 0xdb, 0x38, 0x53, 0x5c, 0xa8, 0xaf,
        0xce, 0xaf, 0x0b, 0xf1, 0x2b, 0x88, 0x1d, 0xc2, 0x00, 0xc9, 0x83,
        0x3d, 0xa7, 0x26, 0xe9, 0x37, 0x6c, 0x2e, 0x32, 0xcf, 0xf7,
    };
    CHECK(std::memcmp(h.data(), expected, 32) == 0);
}

TEST_CASE("base64url round-trip handles non-padded lengths")
{
    for (std::string_view s : {"", "f", "fo", "foo", "foob", "fooba", "foobar"}) {
        const auto encoded = base64url(as_bytes(s));
        auto       decoded = base64url_decode(encoded);
        REQUIRE(decoded.has_value());
        const std::string back(decoded->begin(), decoded->end());
        CHECK(back == s);
    }
}

TEST_CASE("mint + verify round-trip with no exp")
{
    const std::string secret_str = "test-secret-string";
    const auto        secret     = as_bytes(secret_str);
    const std::string claims_json =
        R"({"sub":"u-1","tier":"premium","iss":"1bit.systems"})";
    const auto token = mint_hs256(secret, claims_json);

    auto v = verify_hs256(secret, token, /*now=*/0);
    REQUIRE(v.has_value());
    CHECK(v->tier == "premium");
    CHECK(v->sub == "u-1");
    CHECK(v->iss == "1bit.systems");
}

TEST_CASE("verify rejects wrong secret")
{
    const auto token =
        mint_hs256(as_bytes("right-secret"),
                   R"({"sub":"u","tier":"premium","iss":"x"})");
    auto v = verify_hs256(as_bytes("wrong-secret-x"), token, 0);
    CHECK_FALSE(v.has_value());
}

TEST_CASE("verify rejects expired token")
{
    const auto token = mint_hs256(
        as_bytes("s"),
        R"({"sub":"u","tier":"premium","iss":"x","exp":100})");
    auto v = verify_hs256(as_bytes("s"), token, 200);
    CHECK_FALSE(v.has_value());
}

TEST_CASE("verify rejects malformed token")
{
    auto v = verify_hs256(as_bytes("s"), "not.a.jwt-bad", 0);
    CHECK_FALSE(v.has_value());
}
