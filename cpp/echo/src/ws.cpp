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
            return std::unexpected(WsError{WsError::Kind::Closed, "eof"});
        }
        if (r < 0) {
            return std::unexpected(WsError{WsError::Kind::Recv,
                                           std::strerror(errno)});
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

} // namespace

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
                WsError{WsError::Kind::Handshake, "premature close"});
        }
        buf.append(reinterpret_cast<const char*>(tmp),
                   static_cast<std::size_t>(n));
        if (buf.size() > 16 * 1024) {
            return std::unexpected(
                WsError{WsError::Kind::Handshake, "request too large"});
        }
    }

    // Parse request line + headers (case-insensitive).
    const auto first_nl = buf.find("\r\n");
    if (first_nl == std::string::npos) {
        return std::unexpected(
            WsError{WsError::Kind::Handshake, "no request line"});
    }
    const std::string req_line = buf.substr(0, first_nl);
    // "GET /ws HTTP/1.1"
    auto sp1 = req_line.find(' ');
    auto sp2 = req_line.find(' ', sp1 + 1);
    if (sp1 == std::string::npos || sp2 == std::string::npos) {
        return std::unexpected(
            WsError{WsError::Kind::Handshake, "bad request line"});
    }
    std::string path = req_line.substr(sp1 + 1, sp2 - sp1 - 1);

    std::string key;
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
        // trim
        while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) {
            val.erase(0, 1);
        }
        while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) {
            val.pop_back();
        }
        str_lower(name);
        if (name == "sec-websocket-key") key = val;
    }
    if (key.empty()) {
        return std::unexpected(
            WsError{WsError::Kind::Handshake, "missing Sec-WebSocket-Key"});
    }
    const std::string accept = ws_accept_key(key);
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
        return std::unexpected(WsError{WsError::Kind::Send, "handshake send"});
    }
    return path;
}

namespace {

[[nodiscard]] std::expected<WsMessage, WsError> ws_recv_one(int fd)
{
    std::uint8_t hdr[2];
    if (auto r = recv_exact(fd, hdr, 2); !r) {
        return std::unexpected(r.error());
    }
    const bool          fin     = (hdr[0] & 0x80) != 0;
    const std::uint8_t  opcode  = hdr[0] & 0x0F;
    const bool          masked  = (hdr[1] & 0x80) != 0;
    std::uint64_t       len     = hdr[1] & 0x7F;
    if (len == 126) {
        std::uint8_t ext[2];
        if (auto r = recv_exact(fd, ext, 2); !r) {
            return std::unexpected(r.error());
        }
        len = (static_cast<std::uint64_t>(ext[0]) << 8) | ext[1];
    } else if (len == 127) {
        std::uint8_t ext[8];
        if (auto r = recv_exact(fd, ext, 8); !r) {
            return std::unexpected(r.error());
        }
        len = 0;
        for (int k = 0; k < 8; ++k) {
            len = (len << 8) | ext[k];
        }
    }
    std::uint8_t mask[4]{};
    if (masked) {
        if (auto r = recv_exact(fd, mask, 4); !r) {
            return std::unexpected(r.error());
        }
    }
    if (len > 64ULL * 1024 * 1024) {
        return std::unexpected(WsError{WsError::Kind::Protocol, "frame too big"});
    }
    WsMessage msg;
    msg.opcode = static_cast<WsOpcode>(opcode);
    msg.payload.resize(static_cast<std::size_t>(len));
    if (len > 0) {
        if (auto r = recv_exact(fd, msg.payload.data(),
                                msg.payload.size());
            !r) {
            return std::unexpected(r.error());
        }
        if (masked) {
            for (std::size_t k = 0; k < msg.payload.size(); ++k) {
                msg.payload[k] ^= mask[k & 3];
            }
        }
    }
    if (!fin) {
        // Fragments not supported beyond this minimal impl.
        return std::unexpected(
            WsError{WsError::Kind::Protocol, "fragmented frames not supported"});
    }
    return msg;
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
                    return std::unexpected(WsError{WsError::Kind::Send, "pong"});
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
        return std::unexpected(WsError{WsError::Kind::Send, "ws send"});
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
    return send_frame(fd, WsOpcode::Close, nullptr, 0);
}

} // namespace onebit::echo
