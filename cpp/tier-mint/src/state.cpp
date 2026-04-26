#include "onebit/tier_mint/state.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace onebit::tier_mint {

namespace {

[[nodiscard]] std::int64_t unix_now() noexcept
{
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

[[nodiscard]] std::vector<std::uint8_t> env_bytes(const char* name)
{
    const auto* v = std::getenv(name);
    if (v == nullptr) {
        return {};
    }
    return {v, v + std::strlen(v)};
}

[[nodiscard]] bool eq_bytes(const std::vector<std::uint8_t>& a,
                            const std::vector<std::uint8_t>& b)
{
    return a == b;
}

} // namespace

std::expected<Config, std::string> Config::from_env()
{
    Config c;
    c.jwt_secret = env_bytes("HALO_TIER_HMAC_SECRET");
    if (c.jwt_secret.empty()) {
        return std::unexpected(
            std::string{"HALO_TIER_HMAC_SECRET must be set"});
    }
    if (c.jwt_secret.size() < 32) {
        return std::unexpected(
            std::string{"HALO_TIER_HMAC_SECRET too short; need >= 32 bytes"});
    }
    c.btcpay_webhook_secret = env_bytes("HALO_BTCPAY_WEBHOOK_SECRET");
    if (c.btcpay_webhook_secret.empty()) {
        return std::unexpected(
            std::string{"HALO_BTCPAY_WEBHOOK_SECRET must be set"});
    }
    if (c.btcpay_webhook_secret.size() < 32) {
        return std::unexpected(
            std::string{"HALO_BTCPAY_WEBHOOK_SECRET too short; need >= 32 bytes"});
    }
    c.admin_secret = env_bytes("HALO_TIER_ADMIN_SECRET");
    if (c.admin_secret.empty()) {
        return std::unexpected(
            std::string{"HALO_TIER_ADMIN_SECRET must be set"});
    }
    if (c.admin_secret.size() < 32) {
        return std::unexpected(
            std::string{"HALO_TIER_ADMIN_SECRET too short; need >= 32 bytes"});
    }
    // Pairwise inequality across all three secrets. Reusing one byte-string in
    // two roles would let a leak in one path forge tokens in another. In
    // particular, btcpay_webhook_secret == jwt_secret would let BTCPay (or
    // anyone who learns the webhook secret) mint premium JWTs.
    if (eq_bytes(c.admin_secret, c.jwt_secret)) {
        return std::unexpected(
            std::string{"HALO_TIER_ADMIN_SECRET must NOT equal HALO_TIER_HMAC_SECRET"});
    }
    if (eq_bytes(c.btcpay_webhook_secret, c.jwt_secret)) {
        return std::unexpected(
            std::string{"HALO_BTCPAY_WEBHOOK_SECRET must NOT equal HALO_TIER_HMAC_SECRET"});
    }
    if (eq_bytes(c.btcpay_webhook_secret, c.admin_secret)) {
        return std::unexpected(
            std::string{"HALO_BTCPAY_WEBHOOK_SECRET must NOT equal HALO_TIER_ADMIN_SECRET"});
    }
    return c;
}

struct AppState::Impl {
    Config                                       cfg;
    mutable std::mutex                           m;
    std::unordered_map<std::string, PollEntry>   poll;
    std::unordered_set<std::string>              revoked;
};

AppState::AppState(Config cfg) : impl_{std::make_unique<Impl>()}
{
    impl_->cfg = std::move(cfg);
}
AppState::AppState(AppState&&) noexcept            = default;
AppState& AppState::operator=(AppState&&) noexcept = default;
AppState::~AppState()                              = default;

const Config& AppState::cfg() const noexcept { return impl_->cfg; }

void AppState::insert_poll(std::string invoice_id, std::string jwt)
{
    std::lock_guard lk(impl_->m);
    impl_->poll[std::move(invoice_id)] = PollEntry{std::move(jwt), unix_now()};
}

std::optional<PollEntry> AppState::take_poll(const std::string& invoice_id)
{
    std::lock_guard lk(impl_->m);
    auto            it = impl_->poll.find(invoice_id);
    if (it == impl_->poll.end()) {
        return std::nullopt;
    }
    PollEntry e = std::move(it->second);
    impl_->poll.erase(it);
    return e;
}

bool AppState::is_revoked(const std::string& id) const
{
    std::lock_guard lk(impl_->m);
    return impl_->revoked.contains(id);
}

void AppState::revoke(std::string id)
{
    std::lock_guard lk(impl_->m);
    impl_->revoked.insert(std::move(id));
}

std::size_t AppState::poll_size() const
{
    std::lock_guard lk(impl_->m);
    return impl_->poll.size();
}

std::size_t AppState::revoked_size() const
{
    std::lock_guard lk(impl_->m);
    return impl_->revoked.size();
}

} // namespace onebit::tier_mint
