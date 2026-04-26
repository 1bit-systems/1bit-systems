#pragma once

// Service config + shared state.

#include <chrono>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace onebit::tier_mint {

struct Config {
    std::vector<std::uint8_t> jwt_secret;            // HALO_TIER_HMAC_SECRET
    std::vector<std::uint8_t> btcpay_webhook_secret; // HALO_BTCPAY_WEBHOOK_SECRET
    std::vector<std::uint8_t> admin_secret;          // HALO_TIER_ADMIN_SECRET
    std::string               issuer{"1bit.systems"};
    std::chrono::seconds      jwt_ttl{60 * 60 * 24 * 30}; // 30 days

    [[nodiscard]] static std::expected<Config, std::string> from_env();
};

struct PollEntry {
    std::string  jwt;
    std::int64_t minted_at_unix{0};
};

class AppState {
public:
    explicit AppState(Config cfg);
    AppState(const AppState&)            = delete;
    AppState& operator=(const AppState&) = delete;
    AppState(AppState&&) noexcept;
    AppState& operator=(AppState&&) noexcept;
    ~AppState();

    [[nodiscard]] const Config& cfg() const noexcept;

    void                              insert_poll(std::string invoice_id, std::string jwt);
    [[nodiscard]] std::optional<PollEntry> take_poll(const std::string& invoice_id);
    [[nodiscard]] bool                is_revoked(const std::string& id) const;
    void                              revoke(std::string id);

    // Test-only accessors.
    [[nodiscard]] std::size_t poll_size() const;
    [[nodiscard]] std::size_t revoked_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

inline constexpr std::int64_t POLL_CACHE_TTL_SECS = 600;

} // namespace onebit::tier_mint
