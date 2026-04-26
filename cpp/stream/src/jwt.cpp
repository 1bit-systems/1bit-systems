#include "onebit/stream/jwt.hpp"

#include "onebit/ingest/sha256.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cstring>

namespace onebit::stream::jwt {

namespace {

constexpr std::size_t SHA256_BLOCK = 64;

[[nodiscard]] std::array<std::uint8_t, SHA256_BLOCK>
build_padded_key(std::span<const std::uint8_t> key)
{
    std::array<std::uint8_t, SHA256_BLOCK> k{};
    if (key.size() > SHA256_BLOCK) {
        const auto h = onebit::ingest::detail::sha256(key);
        std::memcpy(k.data(), h.data(), 32);
    } else {
        std::memcpy(k.data(), key.data(), key.size());
    }
    return k;
}

constexpr std::string_view URL_ALPHA =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

} // namespace

std::array<std::uint8_t, 32> hmac_sha256(std::span<const std::uint8_t> key,
                                          std::span<const std::uint8_t> msg) noexcept
{
    auto                                  pk = build_padded_key(key);
    std::array<std::uint8_t, SHA256_BLOCK> ipad{};
    std::array<std::uint8_t, SHA256_BLOCK> opad{};
    for (std::size_t i = 0; i < SHA256_BLOCK; ++i) {
        ipad[i] = pk[i] ^ 0x36;
        opad[i] = pk[i] ^ 0x5C;
    }
    onebit::ingest::detail::Sha256 inner;
    inner.update(std::span<const std::uint8_t>{ipad.data(), ipad.size()});
    inner.update(msg);
    const auto inner_hash = inner.finalize();

    onebit::ingest::detail::Sha256 outer;
    outer.update(std::span<const std::uint8_t>{opad.data(), opad.size()});
    outer.update(std::span<const std::uint8_t>{inner_hash.data(), inner_hash.size()});
    return outer.finalize();
}

std::string base64url(std::span<const std::uint8_t> bytes)
{
    std::string out;
    out.reserve(((bytes.size() + 2) / 3) * 4);
    std::size_t i = 0;
    while (i + 3 <= bytes.size()) {
        std::uint32_t v = (static_cast<std::uint32_t>(bytes[i]) << 16) |
                          (static_cast<std::uint32_t>(bytes[i + 1]) << 8) |
                          static_cast<std::uint32_t>(bytes[i + 2]);
        out.push_back(URL_ALPHA[(v >> 18) & 0x3F]);
        out.push_back(URL_ALPHA[(v >> 12) & 0x3F]);
        out.push_back(URL_ALPHA[(v >> 6) & 0x3F]);
        out.push_back(URL_ALPHA[v & 0x3F]);
        i += 3;
    }
    if (i < bytes.size()) {
        const std::size_t rem = bytes.size() - i;
        std::uint32_t     v   = static_cast<std::uint32_t>(bytes[i]) << 16;
        if (rem == 2) {
            v |= static_cast<std::uint32_t>(bytes[i + 1]) << 8;
        }
        out.push_back(URL_ALPHA[(v >> 18) & 0x3F]);
        out.push_back(URL_ALPHA[(v >> 12) & 0x3F]);
        if (rem == 2) {
            out.push_back(URL_ALPHA[(v >> 6) & 0x3F]);
        }
    }
    return out;
}

std::expected<std::vector<std::uint8_t>, VerifyError>
base64url_decode(std::string_view s)
{
    auto idx = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') {
            return c - 'A';
        }
        if (c >= 'a' && c <= 'z') {
            return c - 'a' + 26;
        }
        if (c >= '0' && c <= '9') {
            return c - '0' + 52;
        }
        if (c == '-') {
            return 62;
        }
        if (c == '_') {
            return 63;
        }
        return -1;
    };

    // Skip optional padding for tolerance.
    while (!s.empty() && s.back() == '=') {
        s.remove_suffix(1);
    }
    if (s.size() % 4 == 1) {
        return std::unexpected(VerifyError{"base64url: invalid length"});
    }

    std::vector<std::uint8_t> out;
    out.reserve((s.size() * 3) / 4);
    std::uint32_t acc      = 0;
    int           bits     = 0;
    for (char c : s) {
        const int v = idx(c);
        if (v < 0) {
            return std::unexpected(VerifyError{"base64url: invalid char"});
        }
        acc = (acc << 6) | static_cast<std::uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>((acc >> bits) & 0xFF));
        }
    }
    return out;
}

std::string mint_hs256(std::span<const std::uint8_t> secret,
                       std::string_view              claims_json)
{
    static constexpr std::string_view header =
        R"({"alg":"HS256","typ":"JWT"})";
    const auto h_b64 =
        base64url(std::span<const std::uint8_t>{
            reinterpret_cast<const std::uint8_t*>(header.data()), header.size()});
    const auto p_b64 =
        base64url(std::span<const std::uint8_t>{
            reinterpret_cast<const std::uint8_t*>(claims_json.data()),
            claims_json.size()});

    std::string signing_input = h_b64;
    signing_input.push_back('.');
    signing_input.append(p_b64);

    const auto sig = hmac_sha256(
        secret, std::span<const std::uint8_t>{
                    reinterpret_cast<const std::uint8_t*>(signing_input.data()),
                    signing_input.size()});
    const auto s_b64 =
        base64url(std::span<const std::uint8_t>{sig.data(), sig.size()});

    std::string token = signing_input;
    token.push_back('.');
    token.append(s_b64);
    return token;
}

std::expected<Claims, VerifyError> verify_hs256(std::span<const std::uint8_t> secret,
                                                std::string_view              token,
                                                std::int64_t                  now_unix)
{
    const auto first = token.find('.');
    if (first == std::string_view::npos) {
        return std::unexpected(VerifyError{"jwt: missing first dot"});
    }
    const auto second = token.find('.', first + 1);
    if (second == std::string_view::npos) {
        return std::unexpected(VerifyError{"jwt: missing second dot"});
    }
    const auto h_b64    = token.substr(0, first);
    const auto p_b64    = token.substr(first + 1, second - first - 1);
    const auto sig_b64  = token.substr(second + 1);
    const auto signing  = std::string{token.substr(0, second)};

    const auto expected = hmac_sha256(
        secret, std::span<const std::uint8_t>{
                    reinterpret_cast<const std::uint8_t*>(signing.data()),
                    signing.size()});
    auto provided = base64url_decode(sig_b64);
    if (!provided) {
        return std::unexpected(provided.error());
    }
    if (provided->size() != 32) {
        return std::unexpected(VerifyError{"jwt: signature length"});
    }
    std::uint8_t diff = 0;
    for (std::size_t i = 0; i < 32; ++i) {
        diff |= expected[i] ^ (*provided)[i];
    }
    if (diff != 0) {
        return std::unexpected(VerifyError{"jwt: signature mismatch"});
    }

    auto payload_bytes = base64url_decode(p_b64);
    if (!payload_bytes) {
        return std::unexpected(payload_bytes.error());
    }
    const std::string payload(payload_bytes->begin(), payload_bytes->end());

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(payload);
    } catch (const nlohmann::json::parse_error& e) {
        return std::unexpected(VerifyError{std::string{"jwt: payload not json: "} + e.what()});
    }
    if (!j.is_object()) {
        return std::unexpected(VerifyError{"jwt: payload not an object"});
    }

    Claims c;
    if (j.contains("sub") && j["sub"].is_string()) {
        c.sub = j["sub"].get<std::string>();
    }
    if (j.contains("tier") && j["tier"].is_string()) {
        c.tier = j["tier"].get<std::string>();
    }
    if (j.contains("iss") && j["iss"].is_string()) {
        c.iss = j["iss"].get<std::string>();
    }
    if (j.contains("exp") && j["exp"].is_number_integer()) {
        c.exp = j["exp"].get<std::int64_t>();
    }
    if (j.contains("iat") && j["iat"].is_number_integer()) {
        c.iat = j["iat"].get<std::int64_t>();
    }
    if (j.contains("jti") && j["jti"].is_string()) {
        c.jti = j["jti"].get<std::string>();
    }
    if (j.contains("btcpay_invoice") && j["btcpay_invoice"].is_string()) {
        c.btcpay_invoice = j["btcpay_invoice"].get<std::string>();
    }

    if (c.exp.has_value() && now_unix > *c.exp) {
        return std::unexpected(VerifyError{"jwt: token expired"});
    }
    return c;
}

} // namespace onebit::stream::jwt
