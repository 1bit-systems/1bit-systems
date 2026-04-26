#pragma once

// Minimal HS256 JWT helpers — verify-only on the read side, plus a
// `mint` helper used by tests. The matching service that actually mints
// premium tokens lives in tier-mint and ships its own copy.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::stream::jwt {

struct VerifyError {
    std::string message;
};

struct Claims {
    std::string                  sub;
    std::string                  tier;
    std::string                  iss;
    std::optional<std::int64_t>  exp;
    std::optional<std::int64_t>  iat;
    std::optional<std::string>   jti;
    std::optional<std::string>   btcpay_invoice;
};

// HMAC-SHA256 — `key` is the raw secret. Public for tests.
[[nodiscard]] std::array<std::uint8_t, 32>
                          hmac_sha256(std::span<const std::uint8_t> key,
                                      std::span<const std::uint8_t> msg) noexcept;

// Base64url (no-padding) encode/decode. Public for tests.
[[nodiscard]] std::string base64url(std::span<const std::uint8_t> bytes);
[[nodiscard]] std::expected<std::vector<std::uint8_t>, VerifyError>
                          base64url_decode(std::string_view s);

// Mint an HS256 token from a JSON claims string. Used by tests.
[[nodiscard]] std::string mint_hs256(std::span<const std::uint8_t> secret,
                                     std::string_view              claims_json);

// Verify token signature + decode payload. Returns the parsed claims if
// the signature checks; the caller is responsible for tier/exp policy.
// `now_unix` is injectable for testing (set to current time in production).
[[nodiscard]] std::expected<Claims, VerifyError>
verify_hs256(std::span<const std::uint8_t> secret,
             std::string_view              token,
             std::int64_t                  now_unix);

} // namespace onebit::stream::jwt
