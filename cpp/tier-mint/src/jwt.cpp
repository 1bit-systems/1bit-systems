#include "onebit/tier_mint/jwt.hpp"

#include "onebit/stream/jwt.hpp"

#include <nlohmann/json.hpp>

namespace onebit::tier_mint::jwt {

namespace {

[[nodiscard]] std::int64_t unix_now() noexcept
{
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

} // namespace

Claims base_claims(std::string_view id, std::chrono::seconds ttl, std::string_view issuer)
{
    Claims c;
    c.sub             = id;
    c.tier            = "premium";
    c.iss             = issuer;
    c.iat             = unix_now();
    c.exp             = c.iat + ttl.count();
    c.jti             = std::string{id};
    c.btcpay_invoice  = std::string{id};
    return c;
}

std::string mint_btcpay(std::span<const std::uint8_t> secret,
                        std::string_view              invoice_id,
                        std::chrono::seconds          ttl,
                        std::string_view              issuer)
{
    const auto c = base_claims(invoice_id, ttl, issuer);

    nlohmann::json j{
        {"sub", c.sub},
        {"tier", c.tier},
        {"iss", c.iss},
        {"exp", c.exp},
        {"iat", c.iat},
        {"jti", c.jti},
        {"btcpay_invoice", c.btcpay_invoice},
    };
    return onebit::stream::jwt::mint_hs256(secret, j.dump());
}

} // namespace onebit::tier_mint::jwt
