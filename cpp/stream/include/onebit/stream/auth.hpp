#pragma once

// HS256 JWT verification for the lossless gate + bearer compare for the
// admin /internal endpoints.

#include <httplib.h>

#include <optional>
#include <string>
#include <vector>

namespace onebit::stream {

struct AuthConfig {
    // Raw HS256 secret. Empty disables the lossless gate (server is then
    // explicit about being misconfigured rather than silently passing).
    std::vector<std::uint8_t> jwt_secret;
    // Shared bearer for /internal/* admin routes. Empty means the admin
    // surface is open (the deployment is expected to bind 127.0.0.1).
    std::string admin_bearer;

    [[nodiscard]] static AuthConfig from_env();
    [[nodiscard]] static AuthConfig make(std::vector<std::uint8_t> secret,
                                          std::string               bearer);
};

enum class GateOutcome : std::uint8_t {
    Allow,
    MissingHeader,
    BadScheme,
    BadToken,
    WrongTier,
    ServerMisconfigured,
};

[[nodiscard]] int as_status(GateOutcome g) noexcept;

[[nodiscard]] GateOutcome check_premium(const AuthConfig& cfg, const httplib::Request& req);
[[nodiscard]] GateOutcome check_admin(const AuthConfig& cfg, const httplib::Request& req);

} // namespace onebit::stream
