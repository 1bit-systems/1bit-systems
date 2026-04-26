#pragma once

// JWT minting for BTCPay-paid invoices. HS256, claim shape locked by
// docs/wiki/tier-jwt-flow.md.

#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace onebit::tier_mint::jwt {

struct Claims {
    std::string                 sub;
    std::string                 tier{"premium"};
    std::string                 iss{"1bit.systems"};
    std::int64_t                exp{0};
    std::int64_t                iat{0};
    std::string                 jti;
    std::string                 btcpay_invoice;
};

// Mint a Premium JWT for a BTCPay invoice.
[[nodiscard]] std::string
                          mint_btcpay(std::span<const std::uint8_t> secret,
                                      std::string_view              invoice_id,
                                      std::chrono::seconds          ttl,
                                      std::string_view              issuer);

// Build claims with the same defaults as `mint_btcpay`. Useful in tests.
[[nodiscard]] Claims base_claims(std::string_view id, std::chrono::seconds ttl,
                                  std::string_view issuer);

} // namespace onebit::tier_mint::jwt
