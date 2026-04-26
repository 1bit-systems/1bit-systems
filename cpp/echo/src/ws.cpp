#include "onebit/echo/ws.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>

namespace onebit::echo {

namespace {

// ---------------------------------------------------------------------------
// SHA-1 (compact, public-domain-style impl) + Base64 — needed for the
// accept-key. Pulling in OpenSSL for one constant-input hash isn't worth
// the build dep, and our deps.cmake list bans non-header-only adds.
// ---------------------------------------------------------------------------

struct Sha1 {
    std::uint32_t s[5]{0x67452301U, 0xEFCDAB89U, 0x98BADCFEU,
                       0x10325476U, 0xC3D2E1F0U};
    std::uint64_t total = 0;
    std::uint8_t  block[64]{};
    std::size_t   used = 0;

    static std::uint32_t rol(std::uint32_t v, int n) {
        return (v << n) | (v >> (32 - n));
    }

    void process_block(const std::uint8_t* p) {
        std::uint32_t w[80];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(p[i * 4]) << 24) |
                   (static_cast<std::uint32_t>(p[i * 4 + 1]) << 16) |
                   (static_cast<std::uint32_t>(p[i * 4 + 2]) << 8) |
                   (static_cast<std::uint32_t>(p[i * 4 + 3]));
        }
        for (int i = 16; i < 80; ++i) {
            w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        }
        std::uint32_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4];
        for (int i = 0; i < 80; ++i) {
            std::uint32_t f, k;
            if      (i < 20) { f = (b & c) | ((~b) & d);  k = 0x5A827999U; }
            else if (i < 40) { f = b ^ c ^ d;             k = 0x6ED9EBA1U; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCU; }
            else             { f = b ^ c ^ d;             k = 0xCA62C1D6U; }
            const std::uint32_t t = rol(a, 5) + f + e + k + w[i];
            e = d; d = c; c = rol(b, 30); b = a; a = t;
        }
        s[0] += a; s[1] += b; s[2] += c; s[3] += d; s[4] += e;
    }

    void update(const std::uint8_t* p, std::size_t n) {
        total += n;
        while (n > 0) {
            const std::size_t take = std::min<std::size_t>(64 - used, n);
            std::memcpy(block + used, p, take);
            used += take; p += take; n -= take;
            if (used == 64) { process_block(block); used = 0; }
        }
    }
    std::array<std::uint8_t, 20> finish() {
        const std::uint64_t bits = total * 8;
        std::uint8_t        pad  = 0x80;
        update(&pad, 1);
        const std::uint8_t zero = 0;
        while (used != 56) update(&zero, 1);
        std::uint8_t lenb[8];
        for (int i = 0; i < 8; ++i) {
            lenb[i] = static_cast<std::uint8_t>(bits >> ((7 - i) * 8));
        }
        update(lenb, 8);
        std::array<std::uint8_t, 20> out{};
        for (int i = 0; i < 5; ++i) {
            out[i * 4]     = static_cast<std::uint8_t>(s[i] >> 24);
            out[i * 4 + 1] = static_cast<std::uint8_t>(s[i] >> 16);
            out[i * 4 + 2] = static_cast<std::uint8_t>(s[i] >> 8);
            out[i * 4 + 3] = static_cast<std::uint8_t>(s[i]);
        }
        return out;
    }
};

constexpr char kB64Tab[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

[[nodiscard]] std::string base64_encode(const std::uint8_t* data, std::size_t len)
{
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    std::size_t i = 0;
    while (i + 3 <= len) {
        const std::uint32_t v =
            (static_cast<std::uint32_t>(data[i]) << 16) |
            (static_cast<std::uint32_t>(data[i + 1]) << 8) |
            (static_cast<std::uint32_t>(data[i + 2]));
        out.push_back(kB64Tab[(v >> 18) & 0x3F]);
        out.push_back(kB64Tab[(v >> 12) & 0x3F]);
        out.push_back(kB64Tab[(v >> 6) & 0x3F]);
        out.push_back(kB64Tab[v & 0x3F]);
        i += 3;
    }
    if (i < len) {
        std::uint32_t v = static_cast<std::uint32_t>(data[i]) << 16;
        if (i + 1 < len) v |= static_cast<std::uint32_t>(data[i + 1]) << 8;
        out.push_back(kB64Tab[(v >> 18) & 0x3F]);
        out.push_back(kB64Tab[(v >> 12) & 0x3F]);
        if (i + 1 < len) {
            out.push_back(kB64Tab[(v >> 6) & 0x3F]);
            out.push_back('=');
        } else {
            out.push_back('=');
            out.push_back('=');
        }
    }
    return out;
}

bool send_all(int fd, const std::uint8_t* p, std::size_t n)
{
    while (n > 0) {
        const ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
        if (w <= 0) return false;
        p += static_cast<std::size_t>(w);
        n -= static_cast<std::size_t>(w);
    }
    return true;
}

[[nodiscard]] std::expected<void, WsError>
recv_exact(int fd, std::uint8_t* p, std::size_t n)
{
    while (n > 0) {
        const ssize_t r = ::recv(fd, p, n, 0);
        if (r == 0) {
            return std::unexpected(WsError{WsError::Kind::Closed, "eof", 0});
        }
        if (r < 0) {
            return std::unexpected(WsError{WsError::Kind::Recv,
                                           std::strerror(errno), 0});
        }
        p += static_cast<std::size_t>(r);
        n -= static_cast<std::size_t>(r);
    }
    return {};
}

void str_lower(std::string& s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
}

[[nodiscard]] bool header_contains_token(std::string_view header,
                                         std::string_view token_lower)
{
    // RFC 7230 §3.2.6: comma-separated list, optional whitespace.
    std::string buf;
    buf.reserve(header.size());
    for (char c : header) {
        buf.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    }
    std::size_t i = 0;
    while (i < buf.size()) {
        while (i < buf.size() && (buf[i] == ' ' || buf[i] == '\t')) ++i;
        std::size_t start = i;
        while (i < buf.size() && buf[i] != ',') ++i;
        std::size_t end = i;
        while (end > start && (buf[end - 1] == ' ' || buf[end - 1] == '\t')) {
            --end;
        }
        if (std::string_view{buf}.substr(start, end - start) == token_lower) {
            return true;
        }
        if (i < buf.size()) ++i; // skip ','
    }
    return false;
}

// Send an HTTP/1.1 400 response then close. Used when the upgrade
// handshake is malformed.
void send_http_400(int fd, std::string_view reason)
{
    std::string resp;
    resp.reserve(128 + reason.size());
    resp += "HTTP/1.1 400 Bad Request\r\n";
    resp += "Content-Type: text/plain; charset=utf-8\r\n";
    resp += "Content-Length: ";
    resp += std::to_string(reason.size());
    resp += "\r\nConnection: close\r\n\r\n";
    resp.append(reason.data(), reason.size());
    (void)send_all(fd, reinterpret_cast<const std::uint8_t*>(resp.data()),
                   resp.size());
}

// Build a Close frame with an optional 16-bit status code (RFC 6455
// §5.5.1: payload starts with a 2-byte big-endian status code).
[[nodiscard]] std::vector<std::uint8_t>
build_close_frame(std::uint16_t code)
{
    std::vector<std::uint8_t> frame;
    if (code == 0) {
        frame = {0x88U, 0x00U};   // FIN | Close, len=0
        return frame;
    }
    frame = {0x88U, 0x02U};       // FIN | Close, len=2
    frame.push_back(static_cast<std::uint8_t>((code >> 8) & 0xFFU));
    frame.push_back(static_cast<std::uint8_t>(code & 0xFFU));
    return frame;
}

void send_close_best_effort(int fd, std::uint16_t code)
{
    const auto frame = build_close_frame(code);
    (void)send_all(fd, frame.data(), frame.size());
}

} // namespace

// ---------------------------------------------------------------------------
// detail:: helpers (declared in the public header for test access)
// ---------------------------------------------------------------------------

namespace detail {

std::expected<std::vector<std::uint8_t>, std::string>
base64_decode(std::string_view in)
{
    // Strict: length must be a multiple of 4, only the standard alphabet,
    // padding only at the end.
    if ((in.size() % 4) != 0 || in.empty()) {
        return std::unexpected(std::string{"base64 length not multiple of 4"});
    }
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::vector<std::uint8_t> out;
    out.reserve((in.size() / 4) * 3);
    std::size_t pad = 0;
    for (std::size_t i = 0; i < in.size(); i += 4) {
        int v[4];
        for (int k = 0; k < 4; ++k) {
            const char c = in[i + k];
            if (c == '=') {
                // Padding only legal in the last quartet.
                if (i + 4 != in.size()) {
                    return std::unexpected(
                        std::string{"base64 stray padding"});
                }
                if (k < 2) {
                    return std::unexpected(
                        std::string{"base64 too much padding"});
                }
                v[k] = 0;
                ++pad;
            } else {
                v[k] = val(c);
                if (v[k] < 0) {
                    return std::unexpected(
                        std::string{"base64 illegal char"});
                }
                if (pad != 0) {
                    return std::unexpected(
                        std::string{"base64 char after padding"});
                }
            }
        }
        const std::uint32_t triple =
            (static_cast<std::uint32_t>(v[0]) << 18) |
            (static_cast<std::uint32_t>(v[1]) << 12) |
            (static_cast<std::uint32_t>(v[2]) << 6) |
            (static_cast<std::uint32_t>(v[3]));
        out.push_back(static_cast<std::uint8_t>((triple >> 16) & 0xFFU));
        if (pad < 2) {
            out.push_back(static_cast<std::uint8_t>((triple >> 8) & 0xFFU));
        }
        if (pad < 1) {
            out.push_back(static_cast<std::uint8_t>(triple & 0xFFU));
        }
    }
    return out;
}

std::expected<void, WsError>
validate_frame_header(std::uint8_t  b0,
                      std::uint8_t  b1,
                      std::uint64_t payload_len,
                      bool          is_server_side)
{
    // RFC 6455 §5.2: RSV1/RSV2/RSV3 MUST be 0 unless an extension was
    // negotiated. We negotiate none, so any non-zero RSV is a violation.
    if ((b0 & 0x70) != 0) {
        return std::unexpected(WsError{WsError::Kind::Protocol,
                                       "RSV bits set without extension",
                                       kWsCloseProtocolErr});
    }
    // Opcode must be one of {0,1,2,8,9,A}. 3-7 and B-F are reserved.
    const std::uint8_t opcode = b0 & 0x0F;
    switch (opcode) {
        case 0x0: case 0x1: case 0x2:
        case 0x8: case 0x9: case 0xA:
            break;
        default:
            return std::unexpected(WsError{WsError::Kind::Protocol,
                                           "reserved opcode",
                                           kWsCloseProtocolErr});
    }
    // Control frames (0x8/0x9/0xA) MUST NOT be fragmented and MUST be
    // ≤125 bytes (RFC 6455 §5.5).
    const bool is_control = (opcode & 0x08) != 0;
    if (is_control) {
        const bool fin = (b0 & 0x80) != 0;
        if (!fin) {
            return std::unexpected(WsError{WsError::Kind::Protocol,
                                           "fragmented control frame",
                                           kWsCloseProtocolErr});
        }
        if (payload_len > 125) {
            return std::unexpected(WsError{WsError::Kind::Protocol,
                                           "control frame > 125 bytes",
                                           kWsCloseProtocolErr});
        }
    }
    // RFC 6455 §5.1: client→server frames MUST be masked. As a server we
    // see MASK=0 from a client → fail closed with 1002.
    const bool masked = (b1 & 0x80) != 0;
    if (is_server_side && !masked) {
        return std::unexpected(WsError{WsError::Kind::Protocol,
                                       "client frame not masked",
                                       kWsCloseProtocolErr});
    }
    // RFC 6455 §5.2: when length is encoded as 64-bit, the high bit MUST
    // be 0. We additionally cap at our DoS bound BEFORE allocating.
    if (payload_len & (1ULL << 63)) {
        return std::unexpected(WsError{WsError::Kind::Protocol,
                                       "64-bit length MSB set",
                                       kWsCloseProtocolErr});
    }
    if (payload_len > kMaxFramePayload) {
        return std::unexpected(WsError{WsError::Kind::Protocol,
                                       "frame too big",
                                       kWsCloseProtocolErr});
    }
    return {};
}

std::expected<WsMessage, WsError>
parse_frame(const ByteReader& read, bool is_server_side)
{
    std::uint8_t hdr[2];
    if (!read(hdr, 2)) {
        return std::unexpected(
            WsError{WsError::Kind::Closed, "eof in frame header", 0});
    }
    // Decode the length field (it can be in three places). Validate the
    // RSV/opcode/control bits BEFORE we look at length, but we need the
    // resolved length to apply the DoS cap and the 64-bit-MSB rule, so
    // do that resolution first.
    std::uint64_t len = hdr[1] & 0x7FU;
    if (len == 126) {
        std::uint8_t ext[2];
        if (!read(ext, 2)) {
            return std::unexpected(
                WsError{WsError::Kind::Closed, "eof in 16-bit length", 0});
        }
        len = (static_cast<std::uint64_t>(ext[0]) << 8) | ext[1];
        // RFC 6455 §5.2: 16-bit length MUST be > 125 (use minimal form).
        if (len <= 125) {
            return std::unexpected(WsError{WsError::Kind::Protocol,
                                           "non-minimal 16-bit length",
                                           kWsCloseProtocolErr});
        }
    } else if (len == 127) {
        std::uint8_t ext[8];
        if (!read(ext, 8)) {
            return std::unexpected(
                WsError{WsError::Kind::Closed, "eof in 64-bit length", 0});
        }
        len = 0;
        for (int k = 0; k < 8; ++k) {
            len = (len << 8) | ext[k];
        }
        // RFC 6455 §5.2: 64-bit length MUST be > 0xFFFF (minimal form).
        // The MSB-must-be-0 check happens in validate_frame_header.
        if (len <= 0xFFFFULL) {
            return std::unexpected(WsError{WsError::Kind::Protocol,
                                           "non-minimal 64-bit length",
                                           kWsCloseProtocolErr});
        }
    }
    if (auto v = validate_frame_header(hdr[0], hdr[1], len, is_server_side);
        !v) {
        return std::unexpected(v.error());
    }

    std::uint8_t mask[4]{};
    const bool   masked = (hdr[1] & 0x80U) != 0;
    if (masked) {
        if (!read(mask, 4)) {
            return std::unexpected(
                WsError{WsError::Kind::Closed, "eof in mask key", 0});
        }
    }

    WsMessage msg;
    msg.opcode = static_cast<WsOpcode>(hdr[0] & 0x0FU);
    // Length cap was checked in validate_frame_header; allocate now.
    msg.payload.resize(static_cast<std::size_t>(len));
    if (len > 0) {
        if (!read(msg.payload.data(), msg.payload.size())) {
            return std::unexpected(
                WsError{WsError::Kind::Closed, "eof in payload", 0});
        }
        if (masked) {
            for (std::size_t k = 0; k < msg.payload.size(); ++k) {
                msg.payload[k] ^= mask[k & 3];
            }
        }
    }

    const bool fin = (hdr[0] & 0x80U) != 0;
    if (!fin) {
        // Fragmented payloads aren't supported by this minimal codec.
        // Return Protocol so the caller sends a 1002 (we've already
        // bounded the allocation via the cap above, so no DoS).
        return std::unexpected(WsError{WsError::Kind::Protocol,
                                       "fragmented frames not supported",
                                       kWsCloseProtocolErr});
    }
    return msg;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::string ws_accept_key(std::string_view key)
{
    static constexpr std::string_view kMagic =
        "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    Sha1 h;
    h.update(reinterpret_cast<const std::uint8_t*>(key.data()), key.size());
    h.update(reinterpret_cast<const std::uint8_t*>(kMagic.data()),
             kMagic.size());
    const auto digest = h.finish();
    return base64_encode(digest.data(), digest.size());
}

std::expected<std::string, WsError> ws_server_handshake(int fd)
{
    std::string buf;
    buf.reserve(1024);
    std::uint8_t tmp[1024];
    while (buf.find("\r\n\r\n") == std::string::npos) {
        const ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
        if (n <= 0) {
            return std::unexpected(
                WsError{WsError::Kind::Handshake, "premature close", 0});
        }
        buf.append(reinterpret_cast<const char*>(tmp),
                   static_cast<std::size_t>(n));
        if (buf.size() > 16 * 1024) {
            send_http_400(fd, "request too large");
            return std::unexpected(
                WsError{WsError::Kind::Handshake, "request too large", 0});
        }
    }

    // Parse request line.
    const auto first_nl = buf.find("\r\n");
    if (first_nl == std::string::npos) {
        send_http_400(fd, "no request line");
        return std::unexpected(
            WsError{WsError::Kind::Handshake, "no request line", 0});
    }
    const std::string req_line = buf.substr(0, first_nl);
    const auto sp1 = req_line.find(' ');
    const auto sp2 = req_line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        send_http_400(fd, "bad request line");
        return std::unexpected(
            WsError{WsError::Kind::Handshake, "bad request line", 0});
    }
    std::string       method = req_line.substr(0, sp1);
    const std::string path   = req_line.substr(sp1 + 1, sp2 - sp1 - 1);
    if (method != "GET") {
        send_http_400(fd, "method not GET");
        return std::unexpected(
            WsError{WsError::Kind::Handshake, "method not GET", 0});
    }

    // Parse headers (case-insensitive name; preserve value).
    std::string upgrade_h;
    std::string connection_h;
    std::string version_h;
    std::string key_h;
    std::size_t i = first_nl + 2;
    while (i < buf.size()) {
        const auto nl = buf.find("\r\n", i);
        if (nl == i || nl == std::string::npos) break;
        std::string line = buf.substr(i, nl - i);
        i = nl + 2;
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string name = line.substr(0, colon);
        std::string val  = line.substr(colon + 1);
        while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) {
            val.erase(0, 1);
        }
        while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) {
            val.pop_back();
        }
        str_lower(name);
        if      (name == "sec-websocket-key")     key_h        = std::move(val);
        else if (name == "upgrade")               upgrade_h    = std::move(val);
        else if (name == "connection")            connection_h = std::move(val);
        else if (name == "sec-websocket-version") version_h    = std::move(val);
    }

    // RFC 6455 §4.2.1.3: Upgrade: websocket (case-insensitive).
    {
        std::string up = upgrade_h;
        str_lower(up);
        if (up != "websocket") {
            send_http_400(fd, "missing or wrong Upgrade header");
            return std::unexpected(WsError{
                WsError::Kind::Handshake, "missing/wrong Upgrade", 0});
        }
    }
    // §4.2.1.4: Connection field MUST contain "Upgrade" token.
    if (!header_contains_token(connection_h, "upgrade")) {
        send_http_400(fd, "Connection header missing Upgrade token");
        return std::unexpected(WsError{
            WsError::Kind::Handshake, "Connection lacks Upgrade", 0});
    }
    // §4.2.1.6: Sec-WebSocket-Version MUST be 13.
    if (version_h != "13") {
        send_http_400(fd, "Sec-WebSocket-Version must be 13");
        return std::unexpected(WsError{
            WsError::Kind::Handshake, "version != 13", 0});
    }
    // §4.2.1.5: Sec-WebSocket-Key MUST decode to 16 bytes.
    if (key_h.empty()) {
        send_http_400(fd, "missing Sec-WebSocket-Key");
        return std::unexpected(WsError{
            WsError::Kind::Handshake, "missing Sec-WebSocket-Key", 0});
    }
    {
        auto decoded = detail::base64_decode(key_h);
        if (!decoded || decoded->size() != 16) {
            send_http_400(fd, "Sec-WebSocket-Key not 16 bytes base64");
            return std::unexpected(WsError{
                WsError::Kind::Handshake,
                "Sec-WebSocket-Key invalid", 0});
        }
    }

    const std::string accept = ws_accept_key(key_h);
    std::string       resp;
    resp.reserve(200);
    resp += "HTTP/1.1 101 Switching Protocols\r\n";
    resp += "Upgrade: websocket\r\n";
    resp += "Connection: Upgrade\r\n";
    resp += "Sec-WebSocket-Accept: ";
    resp += accept;
    resp += "\r\n\r\n";
    if (!send_all(fd, reinterpret_cast<const std::uint8_t*>(resp.data()),
                  resp.size())) {
        return std::unexpected(
            WsError{WsError::Kind::Send, "handshake send", 0});
    }
    return path;
}

namespace {

[[nodiscard]] std::expected<WsMessage, WsError> ws_recv_one(int fd)
{
    detail::ByteReader reader =
        [fd](std::uint8_t* dst, std::size_t n) -> bool {
            return recv_exact(fd, dst, n).has_value();
        };
    auto m = detail::parse_frame(reader, /*is_server_side=*/true);
    if (!m && m.error().kind == WsError::Kind::Protocol) {
        // RFC 6455 §7.4.1: send Close 1002 before bailing so the peer
        // logs why we hung up.
        const std::uint16_t code = m.error().close_code != 0
                                       ? m.error().close_code
                                       : kWsCloseProtocolErr;
        send_close_best_effort(fd, code);
    }
    return m;
}

} // namespace

std::expected<WsMessage, WsError> ws_recv(int fd)
{
    while (true) {
        auto m = ws_recv_one(fd);
        if (!m) return std::unexpected(m.error());
        switch (m->opcode) {
            case WsOpcode::Ping: {
                // Echo with Pong.
                std::vector<std::uint8_t> frame;
                frame.reserve(m->payload.size() + 2);
                frame.push_back(0x80U | 0x0AU);   // FIN|Pong
                if (m->payload.size() < 126) {
                    frame.push_back(static_cast<std::uint8_t>(m->payload.size()));
                } else if (m->payload.size() <= 0xFFFF) {
                    frame.push_back(126U);
                    frame.push_back(static_cast<std::uint8_t>(m->payload.size() >> 8));
                    frame.push_back(static_cast<std::uint8_t>(m->payload.size() & 0xFFU));
                } else {
                    frame.push_back(127U);
                    for (int k = 7; k >= 0; --k) {
                        frame.push_back(
                            static_cast<std::uint8_t>(m->payload.size() >> (k * 8)));
                    }
                }
                frame.insert(frame.end(), m->payload.begin(), m->payload.end());
                if (!send_all(fd, frame.data(), frame.size())) {
                    return std::unexpected(
                        WsError{WsError::Kind::Send, "pong", 0});
                }
                continue;
            }
            case WsOpcode::Pong: {
                continue;
            }
            default:
                return *std::move(m);
        }
    }
}

namespace {

[[nodiscard]] std::expected<void, WsError>
send_frame(int fd, WsOpcode opcode, const std::uint8_t* data, std::size_t len)
{
    std::vector<std::uint8_t> frame;
    frame.reserve(len + 10);
    frame.push_back(static_cast<std::uint8_t>(0x80U | static_cast<std::uint8_t>(opcode)));
    if (len < 126) {
        frame.push_back(static_cast<std::uint8_t>(len));
    } else if (len <= 0xFFFF) {
        frame.push_back(126U);
        frame.push_back(static_cast<std::uint8_t>(len >> 8));
        frame.push_back(static_cast<std::uint8_t>(len & 0xFFU));
    } else {
        frame.push_back(127U);
        for (int k = 7; k >= 0; --k) {
            frame.push_back(static_cast<std::uint8_t>(len >> (k * 8)));
        }
    }
    if (len > 0) frame.insert(frame.end(), data, data + len);
    if (!send_all(fd, frame.data(), frame.size())) {
        return std::unexpected(WsError{WsError::Kind::Send, "ws send", 0});
    }
    return {};
}

} // namespace

std::expected<void, WsError>
ws_send_text(int fd, std::string_view text)
{
    return send_frame(fd, WsOpcode::Text,
                      reinterpret_cast<const std::uint8_t*>(text.data()),
                      text.size());
}

std::expected<void, WsError>
ws_send_binary(int fd, const std::uint8_t* data, std::size_t len)
{
    return send_frame(fd, WsOpcode::Binary, data, len);
}

std::expected<void, WsError> ws_send_close(int fd)
{
    const auto frame = build_close_frame(0);
    if (!send_all(fd, frame.data(), frame.size())) {
        return std::unexpected(WsError{WsError::Kind::Send, "ws close", 0});
    }
    return {};
}

std::expected<void, WsError> ws_send_close_code(int fd, std::uint16_t code)
{
    const auto frame = build_close_frame(code);
    if (!send_all(fd, frame.data(), frame.size())) {
        return std::unexpected(WsError{WsError::Kind::Send, "ws close", 0});
    }
    return {};
}

} // namespace onebit::echo
