#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/tier_mint/btcpay.hpp"
#include "onebit/tier_mint/routes.hpp"
#include "onebit/tier_mint/state.hpp"

#include "onebit/stream/jwt.hpp"

#include <httplib.h>

#include <chrono>
#include <random>
#include <thread>

using onebit::tier_mint::AppState;
using onebit::tier_mint::Config;

namespace {

[[nodiscard]] std::span<const std::uint8_t> as_bytes(std::string_view s) noexcept
{
    return {reinterpret_cast<const std::uint8_t*>(s.data()), s.size()};
}

[[nodiscard]] Config test_config()
{
    Config c;
    c.jwt_secret            = std::vector<std::uint8_t>(40, 'A');
    c.btcpay_webhook_secret = std::vector<std::uint8_t>{'w', 'h'};
    c.admin_secret          = std::vector<std::uint8_t>(40, 'Z');
    c.issuer                = "1bit.systems";
    c.jwt_ttl               = std::chrono::seconds{600};
    return c;
}

[[nodiscard]] int random_port() noexcept
{
    std::random_device              rd;
    std::uniform_int_distribution<> d(20000, 30000);
    return d(rd);
}

class Server {
public:
    Server(AppState& state, int port) : port_{port}
    {
        onebit::tier_mint::build_router(http_, state);
        thread_ = std::thread([this]() { http_.listen("127.0.0.1", port_); });
        for (int i = 0; i < 40; ++i) {
            if (http_.is_running()) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
    }
    ~Server()
    {
        http_.stop();
        if (thread_.joinable()) {
            thread_.join();
        }
    }
    [[nodiscard]] int port() const noexcept { return port_; }

private:
    httplib::Server http_;
    std::thread     thread_;
    int             port_;
};

} // namespace

TEST_CASE("health endpoint")
{
    AppState        s{test_config()};
    Server          srv(s, random_port());
    httplib::Client c("127.0.0.1", srv.port());
    auto            r = c.Get("/v1/health");
    REQUIRE(r);
    CHECK(r->status == 200);
    CHECK(r->body == "ok");
}

TEST_CASE("webhook rejects bad signature")
{
    AppState        s{test_config()};
    Server          srv(s, random_port());
    httplib::Client c("127.0.0.1", srv.port());
    auto            r = c.Post("/btcpay/webhook",
                               R"({"type":"InvoiceSettled","invoiceId":"x"})",
                               "application/json");
    REQUIRE(r);
    CHECK(r->status == 401);
}

TEST_CASE("webhook accepts valid signature, mints token, poll returns it")
{
    auto            cfg = test_config();
    AppState        s{cfg};
    Server          srv(s, random_port());
    httplib::Client c("127.0.0.1", srv.port());

    const std::string body =
        R"({"type":"InvoiceSettled","invoiceId":"inv-pay-1"})";
    const auto sig = onebit::tier_mint::btcpay::sign_for_test(
        std::span<const std::uint8_t>{cfg.btcpay_webhook_secret.data(),
                                      cfg.btcpay_webhook_secret.size()},
        as_bytes(body));

    httplib::Headers h{{"BTCPay-Sig", sig}};
    auto             r = c.Post("/btcpay/webhook", h, body, "application/json");
    REQUIRE(r);
    CHECK(r->status == 200);
    CHECK(r->body.find("\"jwt\":") != std::string::npos);

    // First poll succeeds.
    auto p = c.Get("/tier/poll/inv-pay-1");
    REQUIRE(p);
    CHECK(p->status == 200);
    CHECK(p->body.find("\"jwt\":") != std::string::npos);

    // Second poll => pending (entry was taken).
    auto p2 = c.Get("/tier/poll/inv-pay-1");
    REQUIRE(p2);
    CHECK(p2->status == 202);
}

TEST_CASE("non-Settled events are accepted as 200/no-mint")
{
    auto            cfg = test_config();
    AppState        s{cfg};
    Server          srv(s, random_port());
    httplib::Client c("127.0.0.1", srv.port());

    const std::string body =
        R"({"type":"InvoiceProcessing","invoiceId":"inv-proc"})";
    const auto sig = onebit::tier_mint::btcpay::sign_for_test(
        std::span<const std::uint8_t>{cfg.btcpay_webhook_secret.data(),
                                      cfg.btcpay_webhook_secret.size()},
        as_bytes(body));

    httplib::Headers h{{"BTCPay-Sig", sig}};
    auto             r = c.Post("/btcpay/webhook", h, body, "application/json");
    REQUIRE(r);
    CHECK(r->status == 200);
    // No JWT minted.
    CHECK(r->body.find("\"jwt\":") == std::string::npos);
}

TEST_CASE("revoke requires admin token; webhook then refuses to mint")
{
    auto            cfg = test_config();
    AppState        s{cfg};
    Server          srv(s, random_port());
    httplib::Client c("127.0.0.1", srv.port());

    // Without admin token => 401.
    {
        auto r = c.Post("/tier/revoke", R"({"id":"inv-9"})", "application/json");
        REQUIRE(r);
        CHECK(r->status == 401);
    }

    // With admin token => 200.
    {
        std::string admin(cfg.admin_secret.begin(), cfg.admin_secret.end());
        const auto  body = std::string{"{\"id\":\"inv-9\",\"admin_token\":\""} +
                          admin + "\"}";
        auto r = c.Post("/tier/revoke", body, "application/json");
        REQUIRE(r);
        CHECK(r->status == 200);
    }

    // Webhook for that id now refuses with 409.
    {
        const std::string body =
            R"({"type":"InvoiceSettled","invoiceId":"inv-9"})";
        const auto sig = onebit::tier_mint::btcpay::sign_for_test(
            std::span<const std::uint8_t>{cfg.btcpay_webhook_secret.data(),
                                          cfg.btcpay_webhook_secret.size()},
            as_bytes(body));
        httplib::Headers h{{"BTCPay-Sig", sig}};
        auto             r = c.Post("/btcpay/webhook", h, body, "application/json");
        REQUIRE(r);
        CHECK(r->status == 409);
    }
}
