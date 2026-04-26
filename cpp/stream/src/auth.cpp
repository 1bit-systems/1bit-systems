#include "onebit/stream/auth.hpp"

#include "onebit/stream/jwt.hpp"

#include <chrono>
#include <cstdlib>
#include <string_view>

namespace onebit::stream {

namespace {

[[nodiscard]] std::int64_t now_unix() noexcept
{
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

[[nodiscard]] std::string env_or(const char* name, std::string_view fallback)
{
    const auto* v = std::getenv(name);
    if (v == nullptr) {
        return std::string{fallback};
    }
    return std::string{v};
}

[[nodiscard]] std::string_view bearer_token(const httplib::Request& req)
{
    auto it = req.headers.find("Authorization");
    if (it == req.headers.end()) {
        return {};
    }
    std::string_view raw = it->second;
    if (raw.starts_with("Bearer ")) {
        return raw.substr(7);
    }
    if (raw.starts_with("bearer ")) {
        return raw.substr(7);
    }
    return {};
}

[[nodiscard]] bool constant_time_eq(std::string_view a, std::string_view b) noexcept
{
    if (a.size() != b.size()) {
        return false;
    }
    std::uint8_t diff = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff |= static_cast<std::uint8_t>(a[i]) ^ static_cast<std::uint8_t>(b[i]);
    }
    return diff == 0;
}

} // namespace

AuthConfig AuthConfig::from_env()
{
    AuthConfig c{};
    auto       secret = env_or("HALO_STREAM_JWT_SECRET", "");
    if (!secret.empty()) {
        c.jwt_secret.assign(secret.begin(), secret.end());
    }
    c.admin_bearer = env_or("HALO_STREAM_ADMIN_BEARER", "");
    return c;
}

AuthConfig AuthConfig::make(std::vector<std::uint8_t> secret, std::string bearer)
{
    AuthConfig c;
    c.jwt_secret   = std::move(secret);
    c.admin_bearer = std::move(bearer);
    return c;
}

int as_status(GateOutcome g) noexcept
{
    switch (g) {
    case GateOutcome::Allow:
        return 200;
    case GateOutcome::MissingHeader:
    case GateOutcome::BadScheme:
    case GateOutcome::BadToken:
        return 401;
    case GateOutcome::WrongTier:
        return 403;
    case GateOutcome::ServerMisconfigured:
        return 503;
    }
    return 500;
}

GateOutcome check_premium(const AuthConfig& cfg, const httplib::Request& req)
{
    if (cfg.jwt_secret.empty()) {
        return GateOutcome::ServerMisconfigured;
    }
    auto it = req.headers.find("Authorization");
    if (it == req.headers.end()) {
        return GateOutcome::MissingHeader;
    }
    const auto token = bearer_token(req);
    if (token.empty()) {
        return GateOutcome::BadScheme;
    }
    auto verified = jwt::verify_hs256(
        std::span<const std::uint8_t>{cfg.jwt_secret.data(), cfg.jwt_secret.size()},
        token, now_unix());
    if (!verified) {
        return GateOutcome::BadToken;
    }
    if (verified->tier != "premium") {
        return GateOutcome::WrongTier;
    }
    return GateOutcome::Allow;
}

GateOutcome check_admin(const AuthConfig& cfg, const httplib::Request& req)
{
    // Fail closed: an unset admin bearer means the operator has not opted into
    // exposing /internal/* routes. Refuse rather than wave the request through.
    if (cfg.admin_bearer.empty()) {
        return GateOutcome::ServerMisconfigured;
    }
    auto it = req.headers.find("Authorization");
    if (it == req.headers.end()) {
        return GateOutcome::MissingHeader;
    }
    const auto token = bearer_token(req);
    if (token.empty()) {
        return GateOutcome::BadScheme;
    }
    return constant_time_eq(token, cfg.admin_bearer) ? GateOutcome::Allow
                                                     : GateOutcome::BadToken;
}

} // namespace onebit::stream
