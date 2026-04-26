#include <doctest/doctest.h>

#include "onebit/tier_mint/btcpay.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

using onebit::tier_mint::btcpay::parse_event;
using onebit::tier_mint::btcpay::sign_for_test;
using onebit::tier_mint::btcpay::verify_signature;

namespace {

[[nodiscard]] std::span<const std::uint8_t> as_bytes(std::string_view s) noexcept
{
    return {reinterpret_cast<const std::uint8_t*>(s.data()), s.size()};
}

} // namespace

TEST_CASE("round-trip signature with sha256= prefix")
{
    const std::string_view secret = "webhook-secret";
    const std::string_view body =
        R"({"type":"InvoiceSettled","invoiceId":"abc"})";
    const auto header = sign_for_test(as_bytes(secret), as_bytes(body));
    CHECK(verify_signature(as_bytes(secret), as_bytes(body), header));
}

TEST_CASE("tampered body fails")
{
    const std::string_view secret = "webhook-secret";
    const std::string_view body =
        R"({"type":"InvoiceSettled","invoiceId":"abc"})";
    const auto header = sign_for_test(as_bytes(secret), as_bytes(body));
    const std::string_view evil =
        R"({"type":"InvoiceSettled","invoiceId":"xxx"})";
    CHECK_FALSE(verify_signature(as_bytes(secret), as_bytes(evil), header));
}

TEST_CASE("accepts bare hex header")
{
    const std::string_view secret = "s";
    const std::string_view body   = "hello";
    const auto             header = sign_for_test(as_bytes(secret), as_bytes(body));
    REQUIRE(header.starts_with("sha256="));
    const std::string bare = header.substr(7);
    CHECK(verify_signature(as_bytes(secret), as_bytes(body), bare));
}

TEST_CASE("invalid hex length rejected")
{
    CHECK_FALSE(verify_signature(as_bytes("s"), as_bytes("body"), "sha256=cafe"));
}

TEST_CASE("parse_event extracts type and invoiceId")
{
    const auto e = parse_event(R"({"type":"InvoiceSettled","invoiceId":"abc"})");
    CHECK(e.event_type == "InvoiceSettled");
    CHECK(e.invoice_id == "abc");
}

TEST_CASE("parse_event empty on garbage")
{
    const auto e = parse_event("not json");
    CHECK(e.event_type.empty());
    CHECK(e.invoice_id.empty());
}
