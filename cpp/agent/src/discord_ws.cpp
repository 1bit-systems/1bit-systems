// SPDX-License-Identifier: Apache-2.0
//
// 1bit-agent — Discord gateway client (raw OpenSSL + RFC 6455).
//
// Layered like cpp/echo/src/ws.cpp, mirrored for the client direction:
//   - parse_frame_header / parse_gateway_frame are pure and fully
//     unit-tested.
//   - DiscordGateway::Impl owns the SSL_CTX + SSL + raw fd, the
//     heartbeat jthread, and the resume bookkeeping.
//
// TLS: OpenSSL 3.x.  We use SSL_CTX_set_default_verify_paths() so the
// system root store handles gateway.discord.gg.  SNI is mandatory
// (CDNs return 400 without it).
//
// Frame masking: RFC 6455 §5.1 — every client → server frame MUST be
// masked. We draw 4 bytes from /dev/urandom per frame; on syscall
// failure we fall back to a counter+std::random_device hash so we
// never send an unmasked frame.

#include "onebit/agent/discord_ws.hpp"
#include "onebit/log.hpp"

#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/ssl.h>
#include <openssl/rand.h>

#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <random>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace onebit::agent::discord {

namespace {

// Cap on a single inbound payload. Discord frames are always small
// (READY < 128 KiB even with hundreds of guilds).
constexpr std::size_t kMaxPayload = 4ULL * 1024ULL * 1024ULL;

// Gateway HELLO comes within ~5 s of TCP connect; we wait at most
// 30 s as a sanity bound before declaring the upgrade dead.
constexpr std::chrono::seconds kHelloDeadline{30};

// One-shot HTTP GET line + headers for the WebSocket upgrade. Sec-
// WebSocket-Key is per RFC 6455 §4.1: 16 random bytes, base64.
[[nodiscard]] std::string make_upgrade_request(
    std::string_view host, std::string_view path,
    std::string_view sec_key)
{
    std::string r;
    r.reserve(256);
    r += "GET ";
    r.append(path.data(), path.size());
    r += " HTTP/1.1\r\n";
    r += "Host: ";
    r.append(host.data(), host.size());
    r += "\r\n";
    r += "Upgrade: websocket\r\n";
    r += "Connection: Upgrade\r\n";
    r += "Sec-WebSocket-Key: ";
    r.append(sec_key.data(), sec_key.size());
    r += "\r\n";
    r += "Sec-WebSocket-Version: 13\r\n";
    // Cloudflare bot-management on gateway.discord.gg accepts a wide
    // range of UAs but will RST connections from anything that smells
    // like a probe. Use a Discord-API-Library-style UA so we land in
    // the bot fast-path. Reference: same shape as discord.py / serenity.
    r += "User-Agent: DiscordBot (https://1bit.systems, 1.0)\r\n";
    r += "\r\n";
    return r;
}

// Compact base64 encoder (same alphabet as echo's). Used for the
// Sec-WebSocket-Key header and nothing else; we don't decode here.
constexpr char kB64[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

[[nodiscard]] std::string b64_encode(const std::uint8_t* data, std::size_t n)
{
    std::string out;
    out.reserve(((n + 2) / 3) * 4);
    std::size_t i = 0;
    while (i + 3 <= n) {
        const std::uint32_t v =
            (static_cast<std::uint32_t>(data[i])     << 16) |
            (static_cast<std::uint32_t>(data[i + 1]) << 8)  |
            (static_cast<std::uint32_t>(data[i + 2]));
        out.push_back(kB64[(v >> 18) & 0x3F]);
        out.push_back(kB64[(v >> 12) & 0x3F]);
        out.push_back(kB64[(v >> 6)  & 0x3F]);
        out.push_back(kB64[ v        & 0x3F]);
        i += 3;
    }
    if (i < n) {
        std::uint32_t v = static_cast<std::uint32_t>(data[i]) << 16;
        if (i + 1 < n) v |= static_cast<std::uint32_t>(data[i + 1]) << 8;
        out.push_back(kB64[(v >> 18) & 0x3F]);
        out.push_back(kB64[(v >> 12) & 0x3F]);
        if (i + 1 < n) {
            out.push_back(kB64[(v >> 6) & 0x3F]);
            out.push_back('=');
        } else {
            out.push_back('=');
            out.push_back('=');
        }
    }
    return out;
}

// 16 random bytes → base64 = 24-char Sec-WebSocket-Key.
[[nodiscard]] std::string make_sec_key()
{
    std::uint8_t buf[16];
    if (RAND_bytes(buf, sizeof(buf)) != 1) {
        // Fallback: random_device. Unlikely path; OpenSSL is normally
        // able to seed itself from /dev/urandom or getrandom().
        std::random_device rd;
        for (std::size_t i = 0; i < sizeof(buf); ++i) {
            buf[i] = static_cast<std::uint8_t>(rd());
        }
    }
    return b64_encode(buf, sizeof(buf));
}

[[nodiscard]] GatewayError tls_err(std::string what)
{
    // Pull the first error out of OpenSSL's queue if any. Otherwise
    // report the supplied context. OpenSSL strings are not localized
    // so we pass them straight through.
    if (const unsigned long e = ERR_peek_last_error(); e != 0) {
        char buf[256]{};
        ERR_error_string_n(e, buf, sizeof(buf));
        what += ": ";
        what += buf;
        ERR_clear_error();
    }
    return GatewayError{GatewayError::Kind::Tls, std::move(what), 0};
}

// Resolve host:port via getaddrinfo, connect TCP. Returns a file
// descriptor or -1; error message goes into `err`.
[[nodiscard]] int tcp_connect(const std::string& host, std::uint16_t port,
                              std::string& err)
{
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    const std::string port_s = std::to_string(port);
    if (const int rc = ::getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res);
        rc != 0) {
        err = std::string{"getaddrinfo: "} + ::gai_strerror(rc);
        return -1;
    }
    int fd = -1;
    for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
        fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
        if (fd < 0) continue;
        if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0) break;
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);
    if (fd < 0) {
        err = std::string{"connect: "} + std::strerror(errno);
        return -1;
    }
    // Disable Nagle — gateway frames are small + latency-sensitive.
    int yes = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));
    return fd;
}

// Read one byte at a time from SSL until we see CRLF CRLF — the entire
// upgrade response is small (< 1 KB); this keeps us from reading into
// the first WS frame's payload by mistake.
[[nodiscard]] std::expected<std::string, GatewayError>
read_http_response(SSL* ssl)
{
    std::string buf;
    buf.reserve(1024);
    while (true) {
        char c = 0;
        const int n = SSL_read(ssl, &c, 1);
        if (n <= 0) {
            return std::unexpected(tls_err("read upgrade response"));
        }
        buf.push_back(c);
        if (buf.size() > 16 * 1024) {
            return std::unexpected(GatewayError{
                GatewayError::Kind::Handshake,
                "upgrade response too large", 0});
        }
        const std::size_t s = buf.size();
        if (s >= 4 && buf[s - 4] == '\r' && buf[s - 3] == '\n' &&
            buf[s - 2] == '\r' && buf[s - 1] == '\n') {
            return buf;
        }
    }
}

[[nodiscard]] bool ssl_write_all(SSL* ssl, const std::uint8_t* data,
                                 std::size_t n)
{
    while (n > 0) {
        const int w = SSL_write(ssl, data,
                                static_cast<int>(std::min<std::size_t>(
                                    n, std::numeric_limits<int>::max())));
        if (w <= 0) return false;
        data += w;
        n    -= static_cast<std::size_t>(w);
    }
    return true;
}

[[nodiscard]] bool ssl_read_all(SSL* ssl, std::uint8_t* dst, std::size_t n)
{
    while (n > 0) {
        const int r = SSL_read(ssl, dst,
                               static_cast<int>(std::min<std::size_t>(
                                   n, std::numeric_limits<int>::max())));
        if (r <= 0) {
            const int err = SSL_get_error(ssl, r);
            // Retry transient WANT_READ/WANT_WRITE — blocking sockets can
            // still surface these during renegotiation. Hard-fail anything
            // else so the caller reconnects.
            if (err == SSL_ERROR_WANT_READ || err == SSL_ERROR_WANT_WRITE) {
                continue;
            }
            char ebuf[160]{};
            unsigned long ec = ERR_peek_last_error();
            if (ec) ERR_error_string_n(ec, ebuf, sizeof(ebuf));
            std::fprintf(stderr,
                "[discord_ws] SSL_read returned %d, ssl_err=%d, errno=%d (%s), openssl=%s\n",
                r, err, errno, std::strerror(errno),
                ebuf[0] ? ebuf : "(none)");
            return false;
        }
        dst += r;
        n   -= static_cast<std::size_t>(r);
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Pure framer + JSON serializers (tested in isolation)
// ---------------------------------------------------------------------------

void fill_mask_key(std::uint8_t out[4])
{
    if (RAND_bytes(out, 4) != 1) {
        // Fallback so we never emit an unmasked frame. RFC 6455 §10.3
        // requires *unpredictability* of the mask, not cryptographic
        // strength; std::random_device is good enough as a last resort.
        std::random_device rd;
        for (int i = 0; i < 4; ++i) {
            out[i] = static_cast<std::uint8_t>(rd());
        }
    }
}

std::vector<std::uint8_t>
build_client_frame(WsOpcode             opcode,
                   const std::uint8_t*  payload,
                   std::size_t          len,
                   const std::uint8_t   mask_key[4])
{
    std::vector<std::uint8_t> frame;
    // Worst case header: 2 + 8 + 4 = 14 bytes.
    frame.reserve(len + 14);
    frame.push_back(static_cast<std::uint8_t>(0x80U |
                    static_cast<std::uint8_t>(opcode)));
    if (len < 126) {
        frame.push_back(static_cast<std::uint8_t>(0x80U | len));
    } else if (len <= 0xFFFFU) {
        frame.push_back(static_cast<std::uint8_t>(0x80U | 126U));
        frame.push_back(static_cast<std::uint8_t>((len >> 8) & 0xFFU));
        frame.push_back(static_cast<std::uint8_t>( len       & 0xFFU));
    } else {
        frame.push_back(static_cast<std::uint8_t>(0x80U | 127U));
        for (int k = 7; k >= 0; --k) {
            frame.push_back(static_cast<std::uint8_t>(
                (static_cast<std::uint64_t>(len) >> (k * 8)) & 0xFFU));
        }
    }
    frame.push_back(mask_key[0]);
    frame.push_back(mask_key[1]);
    frame.push_back(mask_key[2]);
    frame.push_back(mask_key[3]);
    const std::size_t off = frame.size();
    frame.resize(off + len);
    for (std::size_t k = 0; k < len; ++k) {
        frame[off + k] = static_cast<std::uint8_t>(
            payload[k] ^ mask_key[k & 3]);
    }
    return frame;
}

std::expected<FrameHeader, GatewayError>
parse_frame_header(const std::uint8_t* buf, std::size_t available)
{
    if (available < 2) {
        return std::unexpected(GatewayError{
            GatewayError::Kind::Protocol, "header truncated", 0});
    }
    const std::uint8_t b0 = buf[0];
    const std::uint8_t b1 = buf[1];
    if ((b0 & 0x70U) != 0) {
        return std::unexpected(GatewayError{
            GatewayError::Kind::Protocol, "RSV bits set", 1002});
    }
    // Server → client frames MUST NOT be masked (RFC 6455 §5.1).
    if ((b1 & 0x80U) != 0) {
        return std::unexpected(GatewayError{
            GatewayError::Kind::Protocol, "server frame masked", 1002});
    }
    const std::uint8_t opc = b0 & 0x0FU;
    switch (opc) {
        case 0x0: case 0x1: case 0x2:
        case 0x8: case 0x9: case 0xA: break;
        default:
            return std::unexpected(GatewayError{
                GatewayError::Kind::Protocol, "reserved opcode", 1002});
    }
    std::uint64_t len = b1 & 0x7FU;
    std::size_t   hdr = 2;
    if (len == 126) {
        if (available < 4) {
            return std::unexpected(GatewayError{
                GatewayError::Kind::Protocol, "16-bit len truncated", 0});
        }
        len = (static_cast<std::uint64_t>(buf[2]) << 8) | buf[3];
        if (len <= 125) {
            return std::unexpected(GatewayError{
                GatewayError::Kind::Protocol, "non-minimal 16-bit length",
                1002});
        }
        hdr = 4;
    } else if (len == 127) {
        if (available < 10) {
            return std::unexpected(GatewayError{
                GatewayError::Kind::Protocol, "64-bit len truncated", 0});
        }
        len = 0;
        for (int k = 0; k < 8; ++k) {
            len = (len << 8) | buf[2 + k];
        }
        if (len <= 0xFFFFULL) {
            return std::unexpected(GatewayError{
                GatewayError::Kind::Protocol, "non-minimal 64-bit length",
                1002});
        }
        if (len & (1ULL << 63)) {
            return std::unexpected(GatewayError{
                GatewayError::Kind::Protocol, "64-bit length MSB set",
                1002});
        }
        hdr = 10;
    }
    if (len > kMaxPayload) {
        return std::unexpected(GatewayError{
            GatewayError::Kind::Protocol, "payload exceeds cap", 1009});
    }
    FrameHeader h;
    h.opcode      = static_cast<WsOpcode>(opc);
    h.fin         = (b0 & 0x80U) != 0;
    h.payload_len = len;
    h.header_size = hdr;
    return h;
}

std::string serialize_identify(const GatewayConfig& cfg)
{
    // Build by hand so the on-the-wire field order matches what every
    // discord library (discord.py, serenity, …) sends. nlohmann's
    // implicit alphabetical ordering put `d` before `op` and Discord's
    // CF frontend was severing those connections post-IDENTIFY without
    // ever surfacing an Op 9 InvalidSession; using string concatenation
    // here forces the canonical shape and dodges that fingerprint.
    auto esc = [](const std::string& s) {
        std::string out;
        out.reserve(s.size() + 2);
        for (char c : s) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b";  break;
                case '\f': out += "\\f";  break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out += c;
                    }
            }
        }
        return out;
    };
    std::string out;
    out.reserve(256);
    out += "{\"op\":";
    out += std::to_string(static_cast<int>(GatewayOp::Identify));
    out += ",\"d\":{\"token\":\"";
    out += esc(cfg.token);
    out += "\",\"intents\":";
    out += std::to_string(cfg.intents);
    out += ",\"properties\":{\"$os\":\"";
    out += esc(cfg.os);
    out += "\",\"$browser\":\"";
    out += esc(cfg.browser);
    out += "\",\"$device\":\"";
    out += esc(cfg.device);
    out += "\"}}}";
    return out;
}

std::string serialize_heartbeat(std::int64_t last_seq)
{
    nlohmann::json j;
    j["op"] = static_cast<int>(GatewayOp::Heartbeat);
    if (last_seq < 0) {
        j["d"] = nullptr;
    } else {
        j["d"] = last_seq;
    }
    return j.dump();
}

std::string serialize_resume(const std::string& token,
                             const std::string& session_id,
                             std::int64_t       last_seq)
{
    nlohmann::json j;
    j["op"] = static_cast<int>(GatewayOp::Resume);
    j["d"]  = {
        {"token",      token},
        {"session_id", session_id},
        {"seq",        last_seq < 0 ? 0 : last_seq},
    };
    return j.dump();
}

std::expected<ParsedFrame, GatewayError>
parse_gateway_frame(std::string_view json_text)
{
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(json_text);
    } catch (const nlohmann::json::parse_error& e) {
        return std::unexpected(GatewayError{
            GatewayError::Kind::Json,
            std::string{"json parse: "} + e.what(), 0});
    }
    if (!j.contains("op") || !j["op"].is_number_integer()) {
        return std::unexpected(GatewayError{
            GatewayError::Kind::Json, "missing op", 0});
    }
    ParsedFrame f;
    f.op = static_cast<GatewayOp>(j["op"].get<int>());
    if (j.contains("s") && j["s"].is_number_integer()) {
        f.sequence = j["s"].get<std::int64_t>();
    }
    if (j.contains("t") && j["t"].is_string()) {
        f.type = j["t"].get<std::string>();
    }
    if (j.contains("d")) {
        f.data = j["d"];
    }
    return f;
}

// ---------------------------------------------------------------------------
// DiscordGateway::Impl
// ---------------------------------------------------------------------------

struct DiscordGateway::Impl {
    GatewayConfig cfg;

    // OpenSSL state.
    SSL_CTX* ctx = nullptr;
    SSL*     ssl = nullptr;
    int      fd  = -1;

    // RFC 6455 reassembly. We fully accept text frames into a buffer
    // before parsing JSON; Discord rarely fragments at the WS layer
    // but the spec permits it.
    std::vector<std::uint8_t> reassembly;

    // Resume bookkeeping.
    ResumeState                 resume;
    std::atomic<bool>           resuming{false};
    std::atomic<bool>           connected{false};
    std::atomic<bool>           stop_flag{false};
    std::atomic<std::int64_t>   heartbeat_interval_ms{0};
    std::atomic<std::int64_t>   last_seq{-1};
    std::atomic<std::int64_t>   pending_acks{0}; // unacked heartbeats
    std::jthread                heartbeat_thread;
    std::mutex                  send_mu;        // serializes SSL_write

    explicit Impl(GatewayConfig c) : cfg(std::move(c)) {}
    ~Impl() { teardown(); }

    void teardown() noexcept
    {
        stop_flag.store(true, std::memory_order_release);
        // Joining the heartbeat thread before tearing the SSL down
        // avoids a UAF on shutdown.
        if (heartbeat_thread.joinable()) {
            heartbeat_thread.request_stop();
            heartbeat_thread.join();
        }
        if (ssl != nullptr) {
            // SSL_shutdown may block — best-effort, single attempt.
            (void)SSL_shutdown(ssl);
            SSL_free(ssl);
            ssl = nullptr;
        }
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        if (ctx != nullptr) {
            SSL_CTX_free(ctx);
            ctx = nullptr;
        }
        connected.store(false, std::memory_order_release);
    }

    [[nodiscard]] std::expected<void, GatewayError> tls_connect()
    {
        // Lazily init OpenSSL once. OpenSSL 3.x auto-inits on first
        // call to TLS_client_method(); we just create a context.
        ctx = SSL_CTX_new(TLS_client_method());
        if (ctx == nullptr) {
            return std::unexpected(tls_err("SSL_CTX_new"));
        }
        SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);
        SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);
        if (SSL_CTX_set_default_verify_paths(ctx) != 1) {
            return std::unexpected(tls_err("set_default_verify_paths"));
        }
        // Cloudflare's bot-management on gateway.discord.gg keys partly
        // on TLS ALPN. WebSockets ride HTTP/1.1, so advertise that
        // explicitly — without ALPN CF was severing post-IDENTIFY.
        constexpr unsigned char kAlpn[] = "\x08http/1.1";
        SSL_CTX_set_alpn_protos(ctx, kAlpn, sizeof(kAlpn) - 1);

        std::string err_msg;
        fd = tcp_connect(cfg.gateway_host, cfg.gateway_port, err_msg);
        if (fd < 0) {
            return std::unexpected(GatewayError{
                GatewayError::Kind::Tls, std::move(err_msg), 0});
        }

        ssl = SSL_new(ctx);
        if (ssl == nullptr) {
            return std::unexpected(tls_err("SSL_new"));
        }
        if (SSL_set_fd(ssl, fd) != 1) {
            return std::unexpected(tls_err("SSL_set_fd"));
        }
        // SNI is mandatory: gateway.discord.gg shares a TLS frontend
        // with other Discord domains.
        if (SSL_set_tlsext_host_name(ssl, cfg.gateway_host.c_str()) != 1) {
            return std::unexpected(tls_err("SNI"));
        }
        // Hostname verification (X509_V_OK only on match).
        if (SSL_set1_host(ssl, cfg.gateway_host.c_str()) != 1) {
            return std::unexpected(tls_err("SSL_set1_host"));
        }
        if (SSL_connect(ssl) != 1) {
            return std::unexpected(tls_err("SSL_connect"));
        }
        return {};
    }

    [[nodiscard]] std::expected<void, GatewayError> ws_upgrade()
    {
        const std::string sec_key = make_sec_key();
        const std::string req     = make_upgrade_request(
            cfg.gateway_host, cfg.gateway_path, sec_key);
        if (!ssl_write_all(ssl,
                           reinterpret_cast<const std::uint8_t*>(req.data()),
                           req.size())) {
            return std::unexpected(tls_err("write upgrade"));
        }
        auto resp = read_http_response(ssl);
        if (!resp) return std::unexpected(resp.error());
        // Status line: "HTTP/1.1 101 ..."
        if (resp->size() < 12 ||
            resp->compare(0, 9, "HTTP/1.1 ") != 0 ||
            resp->compare(9, 3, "101") != 0) {
            return std::unexpected(GatewayError{
                GatewayError::Kind::Handshake,
                "upgrade rejected: " + resp->substr(0, 64), 0});
        }
        return {};
    }

    // Send a Text frame containing the supplied UTF-8 payload, masked
    // per spec. Caller holds send_mu.
    [[nodiscard]] std::expected<void, GatewayError>
    send_text_locked(std::string_view text)
    {
        std::uint8_t mask[4];
        fill_mask_key(mask);
        const auto frame = build_client_frame(
            WsOpcode::Text,
            reinterpret_cast<const std::uint8_t*>(text.data()),
            text.size(), mask);
        std::fprintf(stderr,
                     "[discord_ws] send_text frame=%zu bytes header=",
                     frame.size());
        for (std::size_t k = 0; k < std::min<std::size_t>(8, frame.size()); ++k) {
            std::fprintf(stderr, "%02x", frame[k]);
        }
        std::fprintf(stderr, "\n");
        if (!ssl_write_all(ssl, frame.data(), frame.size())) {
            std::fprintf(stderr, "[discord_ws] send_text SSL_write failed\n");
            return std::unexpected(tls_err("send_text"));
        }
        std::fprintf(stderr, "[discord_ws] send_text ok\n");
        return {};
    }

    [[nodiscard]] std::expected<void, GatewayError>
    send_text(std::string_view text)
    {
        std::scoped_lock lk(send_mu);
        return send_text_locked(text);
    }

    // Pull a single complete WS message. Handles ping → pong and
    // close opcodes inline; only complete Text/Binary messages surface
    // to caller. Iterative (no recursion) so a flood of pings can't
    // blow the stack.
    [[nodiscard]] std::expected<std::string, GatewayError>
    recv_text(std::chrono::milliseconds timeout)
    {
        while (true) {
            // Wait for readability with poll(); SSL may have already
            // buffered data from a prior read.
            if (SSL_pending(ssl) == 0) {
                pollfd pfd{fd, POLLIN, 0};
                const int rc =
                    ::poll(&pfd, 1, static_cast<int>(timeout.count()));
                if (rc == 0) {
                    return std::unexpected(GatewayError{
                        GatewayError::Kind::Timeout, "poll timeout", 0});
                }
                if (rc < 0) {
                    return std::unexpected(GatewayError{
                        GatewayError::Kind::Tls,
                        std::string{"poll: "} + std::strerror(errno), 0});
                }
            }

            std::uint8_t hdr[14];
            if (!ssl_read_all(ssl, hdr, 2)) {
                return std::unexpected(GatewayError{
                    GatewayError::Kind::Closed, "read header", 0});
            }
            std::size_t need_more = 0;
            const std::uint8_t len_marker = hdr[1] & 0x7FU;
            if (len_marker == 126) need_more = 2;
            else if (len_marker == 127) need_more = 8;
            if (need_more > 0 &&
                !ssl_read_all(ssl, hdr + 2, need_more)) {
                return std::unexpected(GatewayError{
                    GatewayError::Kind::Closed, "read ext-len", 0});
            }
            auto h = parse_frame_header(hdr, 2 + need_more);
            if (!h) return std::unexpected(h.error());

            std::vector<std::uint8_t> payload(
                static_cast<std::size_t>(h->payload_len));
            if (h->payload_len > 0 &&
                !ssl_read_all(ssl, payload.data(), payload.size())) {
                return std::unexpected(GatewayError{
                    GatewayError::Kind::Closed, "read payload", 0});
            }

            switch (h->opcode) {
                case WsOpcode::Text:
                case WsOpcode::Binary:
                case WsOpcode::Continuation: {
                    reassembly.insert(reassembly.end(),
                                      payload.begin(), payload.end());
                    if (h->fin) {
                        std::string out(
                            reinterpret_cast<const char*>(reassembly.data()),
                            reassembly.size());
                        reassembly.clear();
                        return out;
                    }
                    continue; // read next fragment
                }
                case WsOpcode::Ping: {
                    std::uint8_t mask[4];
                    fill_mask_key(mask);
                    const auto pong = build_client_frame(
                        WsOpcode::Pong, payload.data(),
                        payload.size(), mask);
                    std::scoped_lock lk(send_mu);
                    if (!ssl_write_all(ssl, pong.data(), pong.size())) {
                        return std::unexpected(tls_err("pong"));
                    }
                    continue;
                }
                case WsOpcode::Pong:
                    continue;
                case WsOpcode::Close: {
                    std::uint16_t code = 0;
                    if (payload.size() >= 2) {
                        code = static_cast<std::uint16_t>(
                            (payload[0] << 8) | payload[1]);
                    }
                    return std::unexpected(GatewayError{
                        GatewayError::Kind::Closed, "peer close", code});
                }
            }
            return std::unexpected(GatewayError{
                GatewayError::Kind::Protocol, "unreachable opcode", 1002});
        }
    }

    void heartbeat_loop(std::stop_token tok)
    {
        // Wait until HELLO sets the interval.
        while (!tok.stop_requested() &&
               heartbeat_interval_ms.load(std::memory_order_acquire) == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        // Discord recommends a jittered first heartbeat in [0,
        // interval). For simplicity we beat at the full interval.
        while (!tok.stop_requested()) {
            const auto interval = std::chrono::milliseconds(
                heartbeat_interval_ms.load(std::memory_order_acquire));
            std::this_thread::sleep_for(interval);
            if (tok.stop_requested()) break;
            if (pending_acks.load(std::memory_order_acquire) >= 2) {
                // Two missed acks → trigger reconnect via socket close.
                onebit::log::eprintln(
                    "[discord] heartbeat ack missed twice, reconnecting");
                stop_flag.store(true, std::memory_order_release);
                if (fd >= 0) ::shutdown(fd, SHUT_RDWR);
                return;
            }
            const std::string hb = serialize_heartbeat(
                last_seq.load(std::memory_order_acquire));
            if (auto r = send_text(hb); !r) {
                onebit::log::eprintln(
                    "[discord] heartbeat send failed: {}", r.error().message);
                return;
            }
            pending_acks.fetch_add(1, std::memory_order_acq_rel);
        }
    }
};

DiscordGateway::DiscordGateway(GatewayConfig cfg)
    : p_(std::make_unique<Impl>(std::move(cfg)))
{
}

DiscordGateway::~DiscordGateway() = default;
DiscordGateway::DiscordGateway(DiscordGateway&&) noexcept            = default;
DiscordGateway& DiscordGateway::operator=(DiscordGateway&&) noexcept = default;

std::expected<void, GatewayError> DiscordGateway::connect()
{
    // Tear down any prior connection before reconnecting.
    p_->teardown();
    p_->stop_flag.store(false, std::memory_order_release);

    std::fprintf(stderr, "[discord_ws] tls_connect…\n");
    if (auto r = p_->tls_connect(); !r) return std::unexpected(r.error());
    std::fprintf(stderr, "[discord_ws] ws_upgrade…\n");
    if (auto r = p_->ws_upgrade();  !r) return std::unexpected(r.error());

    // Read HELLO.
    std::fprintf(stderr, "[discord_ws] recv HELLO…\n");
    auto hello = p_->recv_text(kHelloDeadline);
    if (!hello) {
        std::fprintf(stderr, "[discord_ws] recv HELLO failed: %s\n",
                     hello.error().message.c_str());
        return std::unexpected(hello.error());
    }
    std::fprintf(stderr, "[discord_ws] HELLO ok (%zu bytes)\n", hello->size());
    auto parsed = parse_gateway_frame(*hello);
    if (!parsed) return std::unexpected(parsed.error());
    if (parsed->op != GatewayOp::Hello) {
        return std::unexpected(GatewayError{
            GatewayError::Kind::Handshake, "no HELLO after upgrade", 0});
    }
    if (!parsed->data.contains("heartbeat_interval")) {
        return std::unexpected(GatewayError{
            GatewayError::Kind::Json, "HELLO missing heartbeat_interval", 0});
    }
    p_->heartbeat_interval_ms.store(
        parsed->data["heartbeat_interval"].get<std::int64_t>(),
        std::memory_order_release);

    // Send IDENTIFY or RESUME.
    if (!p_->resume.session_id.empty()) {
        const std::string r = serialize_resume(
            p_->cfg.token, p_->resume.session_id, p_->resume.last_sequence);
        p_->resuming.store(true, std::memory_order_release);
        if (auto sr = p_->send_text(r); !sr) {
            return std::unexpected(sr.error());
        }
    } else {
        const std::string ident = serialize_identify(p_->cfg);
        // Redact token for the trace; keep length as fingerprint.
        std::string redacted = ident;
        if (auto p = redacted.find("\"token\":\""); p != std::string::npos) {
            auto e = redacted.find('"', p + 9);
            if (e != std::string::npos) {
                redacted.replace(p + 9, e - p - 9,
                    "<redacted len=" + std::to_string(e - p - 9) + ">");
            }
        }
        std::fprintf(stderr, "[discord_ws] IDENTIFY: %s\n", redacted.c_str());
        if (auto sr = p_->send_text(ident); !sr) {
            return std::unexpected(sr.error());
        }
    }
    p_->connected.store(true, std::memory_order_release);
    p_->pending_acks.store(0, std::memory_order_release);

    // Spin up the heartbeat thread now that interval is known.
    p_->heartbeat_thread = std::jthread(
        [impl = p_.get()](std::stop_token tok) {
            impl->heartbeat_loop(tok);
        });
    return {};
}

std::expected<GatewayEvent, GatewayError>
DiscordGateway::recv_event(std::chrono::milliseconds timeout)
{
    while (!p_->stop_flag.load(std::memory_order_acquire)) {
        auto raw = p_->recv_text(timeout);
        if (!raw) return std::unexpected(raw.error());
        auto parsed = parse_gateway_frame(*raw);
        if (!parsed) return std::unexpected(parsed.error());

        if (parsed->sequence >= 0) {
            p_->last_seq.store(parsed->sequence, std::memory_order_release);
            p_->resume.last_sequence = parsed->sequence;
        }

        switch (parsed->op) {
            case GatewayOp::HeartbeatAck:
                if (p_->pending_acks.load(std::memory_order_acquire) > 0) {
                    p_->pending_acks.fetch_sub(1,
                                               std::memory_order_acq_rel);
                }
                continue;
            case GatewayOp::Heartbeat: {
                // Server-requested heartbeat — answer immediately.
                const std::string hb = serialize_heartbeat(
                    p_->last_seq.load(std::memory_order_acquire));
                if (auto r = p_->send_text(hb); !r) {
                    return std::unexpected(r.error());
                }
                continue;
            }
            case GatewayOp::Reconnect:
                return std::unexpected(GatewayError{
                    GatewayError::Kind::Closed, "server requested reconnect",
                    1001});
            case GatewayOp::InvalidSession: {
                const bool resumable = parsed->data.is_boolean() &&
                                       parsed->data.get<bool>();
                if (resumable) {
                    return std::unexpected(GatewayError{
                        GatewayError::Kind::ResumeFailed,
                        "invalid session (resumable)", 0});
                }
                p_->resume = ResumeState{}; // clear; force re-IDENTIFY
                return std::unexpected(GatewayError{
                    GatewayError::Kind::SessionInvalid,
                    "invalid session", 0});
            }
            case GatewayOp::Hello:
                // Should not happen mid-session; ignore defensively.
                continue;
            case GatewayOp::Dispatch: {
                // READY captures session_id + resume_gateway_url for
                // future reconnects.
                if (parsed->type == "READY") {
                    if (parsed->data.contains("session_id")) {
                        p_->resume.session_id =
                            parsed->data["session_id"].get<std::string>();
                    }
                    if (parsed->data.contains("resume_gateway_url")) {
                        const auto url =
                            parsed->data["resume_gateway_url"].get<std::string>();
                        // strip "wss://" prefix if present
                        constexpr std::string_view kPrefix = "wss://";
                        if (url.starts_with(kPrefix)) {
                            p_->resume.resume_url = url.substr(kPrefix.size());
                        } else {
                            p_->resume.resume_url = url;
                        }
                    }
                }
                if (parsed->type == "RESUMED") {
                    p_->resuming.store(false, std::memory_order_release);
                }
                GatewayEvent ev;
                ev.type     = std::move(parsed->type);
                ev.data     = std::move(parsed->data);
                ev.sequence = parsed->sequence;
                return ev;
            }
            default:
                continue;
        }
    }
    return std::unexpected(GatewayError{
        GatewayError::Kind::Closed, "stop requested", 0});
}

void DiscordGateway::stop() noexcept
{
    if (p_) {
        p_->stop_flag.store(true, std::memory_order_release);
        if (p_->fd >= 0) ::shutdown(p_->fd, SHUT_RDWR);
    }
}

const ResumeState& DiscordGateway::resume_state() const noexcept
{
    return p_->resume;
}

} // namespace onebit::agent::discord
