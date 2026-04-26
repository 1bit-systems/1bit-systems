#include "onebit/tier_mint/btcpay.hpp"

#include "onebit/stream/jwt.hpp"

#include <nlohmann/json.hpp>

#include <cctype>

namespace onebit::tier_mint::btcpay {

namespace {

[[nodiscard]] std::string lc(std::string_view s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

[[nodiscard]] std::string trim(std::string_view s)
{
    std::size_t a = 0;
    std::size_t b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) {
        ++a;
    }
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) {
        --b;
    }
    return std::string{s.substr(a, b - a)};
}

} // namespace

bool verify_signature(std::span<const std::uint8_t> secret,
                       std::span<const std::uint8_t> body,
                       std::string_view              header) noexcept
{
    auto h = trim(header);
    if (h.starts_with("sha256=")) {
        h.erase(0, 7);
    }
    if (h.size() != 64) {
        return false;
    }
    const auto       mac     = onebit::stream::jwt::hmac_sha256(secret, body);
    static const char hex[]  = "0123456789abcdef";
    std::string      expected;
    expected.reserve(64);
    for (auto b : mac) {
        expected.push_back(hex[(b >> 4) & 0xF]);
        expected.push_back(hex[b & 0xF]);
    }
    const auto observed = lc(h);
    if (observed.size() != expected.size()) {
        return false;
    }
    std::uint8_t diff = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        diff |= static_cast<std::uint8_t>(observed[i]) ^
                static_cast<std::uint8_t>(expected[i]);
    }
    return diff == 0;
}

std::string sign_for_test(std::span<const std::uint8_t> secret,
                          std::span<const std::uint8_t> body)
{
    const auto       mac    = onebit::stream::jwt::hmac_sha256(secret, body);
    static const char hex[] = "0123456789abcdef";
    std::string      out    = "sha256=";
    for (auto b : mac) {
        out.push_back(hex[(b >> 4) & 0xF]);
        out.push_back(hex[b & 0xF]);
    }
    return out;
}

Event parse_event(std::string_view body)
{
    Event e;
    try {
        const auto j = nlohmann::json::parse(body);
        if (!j.is_object()) {
            return e;
        }
        if (j.contains("type") && j["type"].is_string()) {
            e.event_type = j["type"].get<std::string>();
        }
        if (j.contains("invoiceId") && j["invoiceId"].is_string()) {
            e.invoice_id = j["invoiceId"].get<std::string>();
        }
    } catch (...) {
    }
    return e;
}

} // namespace onebit::tier_mint::btcpay
