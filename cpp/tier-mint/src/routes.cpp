#include "onebit/tier_mint/routes.hpp"

#include "onebit/tier_mint/btcpay.hpp"
#include "onebit/tier_mint/jwt.hpp"

#include <nlohmann/json.hpp>

#include <chrono>

namespace onebit::tier_mint {

namespace {

[[nodiscard]] std::int64_t unix_now() noexcept
{
    using namespace std::chrono;
    return duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
}

[[nodiscard]] std::string_view header_or(const httplib::Request& req,
                                          std::string_view        a,
                                          std::string_view        b,
                                          std::string_view        c)
{
    auto it = req.headers.find(std::string{a});
    if (it == req.headers.end()) {
        it = req.headers.find(std::string{b});
    }
    if (it == req.headers.end()) {
        it = req.headers.find(std::string{c});
    }
    if (it == req.headers.end()) {
        return {};
    }
    return it->second;
}

[[nodiscard]] bool constant_time_eq(std::span<const std::uint8_t> a,
                                     std::span<const std::uint8_t> b) noexcept
{
    if (a.size() != b.size()) {
        return false;
    }
    std::uint8_t diff = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        diff |= a[i] ^ b[i];
    }
    return diff == 0;
}

} // namespace

void build_router(httplib::Server& server, AppState& state)
{
    server.Get("/v1/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("ok", "text/plain");
    });

    server.Post(
        "/btcpay/webhook",
        [&state](const httplib::Request& req, httplib::Response& res) {
            const auto sig = header_or(req, "btcpay-sig", "BTCPay-Sig", "BTCPAY-SIG");

            const auto& secret_v = state.cfg().btcpay_webhook_secret;
            std::span<const std::uint8_t> secret{secret_v.data(), secret_v.size()};
            std::span<const std::uint8_t> body{
                reinterpret_cast<const std::uint8_t*>(req.body.data()),
                req.body.size()};
            if (!btcpay::verify_signature(secret, body, sig)) {
                res.status = 401;
                res.set_content("bad signature", "text/plain");
                return;
            }

            const auto event = btcpay::parse_event(req.body);
            if (event.invoice_id.empty()) {
                res.status = 400;
                res.set_content("malformed body", "text/plain");
                return;
            }
            if (event.event_type != "InvoiceSettled") {
                nlohmann::json ok{{"ok", true}};
                res.status = 200;
                res.set_content(ok.dump(), "application/json");
                return;
            }
            if (state.is_revoked(event.invoice_id)) {
                res.status = 409;
                res.set_content("invoice revoked", "text/plain");
                return;
            }

            const auto& jwt_secret_v = state.cfg().jwt_secret;
            const auto  token        = jwt::mint_btcpay(
                std::span<const std::uint8_t>{jwt_secret_v.data(), jwt_secret_v.size()},
                event.invoice_id, state.cfg().jwt_ttl, state.cfg().issuer);
            state.insert_poll(event.invoice_id, token);

            nlohmann::json ok{{"ok", true}, {"jwt", token}};
            res.status = 200;
            res.set_content(ok.dump(), "application/json");
        });

    server.Get(R"(/tier/poll/([^/]+))",
               [&state](const httplib::Request& req, httplib::Response& res) {
                   const std::string invoice_id = req.matches[1];
                   auto              entry      = state.take_poll(invoice_id);
                   if (!entry) {
                       res.status = 202;
                       res.set_content("pending", "text/plain");
                       return;
                   }
                   if (unix_now() - entry->minted_at_unix > POLL_CACHE_TTL_SECS) {
                       res.status = 410;
                       res.set_content("expired", "text/plain");
                       return;
                   }
                   nlohmann::json body{{"jwt", entry->jwt}};
                   res.status = 200;
                   res.set_content(body.dump(), "application/json");
               });

    server.Post(
        "/tier/revoke",
        [&state](const httplib::Request& req, httplib::Response& res) {
            std::string  id;
            std::string  admin_token;
            try {
                const auto j = nlohmann::json::parse(req.body);
                if (!j.is_object()) {
                    res.status = 400;
                    res.set_content("malformed body", "text/plain");
                    return;
                }
                if (j.contains("id") && j["id"].is_string()) {
                    id = j["id"].get<std::string>();
                }
                if (j.contains("admin_token") && j["admin_token"].is_string()) {
                    admin_token = j["admin_token"].get<std::string>();
                }
            } catch (...) {
                res.status = 400;
                res.set_content("malformed body", "text/plain");
                return;
            }
            const auto& expected_v = state.cfg().admin_secret;
            std::span<const std::uint8_t> provided{
                reinterpret_cast<const std::uint8_t*>(admin_token.data()),
                admin_token.size()};
            std::span<const std::uint8_t> expected{
                expected_v.data(), expected_v.size()};
            if (!constant_time_eq(provided, expected)) {
                res.status = 401;
                res.set_content("bad admin token", "text/plain");
                return;
            }
            if (id.empty()) {
                res.status = 400;
                res.set_content("missing id", "text/plain");
                return;
            }
            state.revoke(std::move(id));
            res.status = 200;
            res.set_content("revoked", "text/plain");
        });
}

} // namespace onebit::tier_mint
