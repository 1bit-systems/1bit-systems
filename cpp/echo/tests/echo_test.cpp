#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/echo/codec.hpp"
#include "onebit/echo/server.hpp"
#include "onebit/echo/ws.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <nlohmann/json.hpp>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using onebit::echo::Codec;
using onebit::echo::codec_from_str;
using onebit::echo::codec_name;
using onebit::echo::EchoConfig;
using onebit::echo::EchoServer;
using onebit::echo::kFrameSamples;
using onebit::echo::kTargetSr;
using onebit::echo::parse_wav;
using onebit::echo::preamble_json;
using onebit::echo::wav_to_pcm_frames;
using onebit::echo::ws_accept_key;
using onebit::echo::WsMessage;
using onebit::echo::WsOpcode;
using nlohmann::json;

// =====================================================================
// Codec
// =====================================================================

TEST_CASE("codec round-trip + opus alias")
{
    CHECK(codec_name(Codec::Wav) == "wav");
    CHECK(codec_name(Codec::Pcm) == "pcm");
    auto a = codec_from_str("wav");
    REQUIRE(a.has_value());
    CHECK(*a == Codec::Wav);
    auto b = codec_from_str("pcm");
    REQUIRE(b.has_value());
    CHECK(*b == Codec::Pcm);
    auto c = codec_from_str("opus");
    REQUIRE(c.has_value());
    CHECK(*c == Codec::Pcm);
    CHECK(!codec_from_str("mp3").has_value());
}

TEST_CASE("preamble json shape")
{
    auto v_pcm = json::parse(preamble_json(Codec::Pcm));
    CHECK(v_pcm["codec"]       == "pcm");
    CHECK(v_pcm["sample_rate"] == 48000);
    CHECK(v_pcm["channels"]    == 1);
    CHECK(v_pcm["frame_ms"]    == 20);
    auto v_wav = json::parse(preamble_json(Codec::Wav));
    CHECK(v_wav["codec"]       == "wav");
    CHECK(v_wav["sample_rate"] == 24000);
    CHECK(v_wav["channels"]    == 1);
    CHECK(v_wav["frame_ms"]    == 0);
}

TEST_CASE("wav build → parse round-trip")
{
    std::vector<std::int16_t> pcm = {0, 123, -123, 32000, -32000};
    auto wav = onebit::echo::build_wav(24'000, 1, pcm.data(), pcm.size());
    auto info = parse_wav(wav.data(), wav.size());
    REQUIRE(info.has_value());
    CHECK(info->sample_rate     == 24'000);
    CHECK(info->channels        == 1);
    CHECK(info->bits_per_sample == 16);
    CHECK(info->data_offset     == 44);
    CHECK(info->data_len        == pcm.size() * 2);
}

TEST_CASE("wav_to_pcm_frames yields 20ms frames at 48k")
{
    // 250 ms of silence at 24 kHz → 6000 samples → after 2x → 12000 samples
    // → 12000 / 960 = 12 full frames + 1 padded.
    std::vector<std::int16_t> pcm(24'000U / 4U, 0);
    auto wav = onebit::echo::build_wav(24'000, 1, pcm.data(), pcm.size());
    auto frames = wav_to_pcm_frames(wav.data(), wav.size());
    REQUIRE(frames.has_value());
    CHECK(frames->size() >= 12);
    for (const auto& f : *frames) {
        CHECK(f.size() == kFrameSamples * 2);
    }
}

TEST_CASE("wav_to_pcm_frames rejects stereo")
{
    std::vector<std::int16_t> pcm = {0, 0, 0, 0};
    auto wav = onebit::echo::build_wav(24'000, 2, pcm.data(), pcm.size());
    auto r = wav_to_pcm_frames(wav.data(), wav.size());
    CHECK(!r.has_value());
}

// =====================================================================
// WS handshake math (RFC 6455 §1.3 example)
// =====================================================================

TEST_CASE("ws_accept_key matches RFC 6455 example")
{
    // RFC 6455 §1.3: key "dGhlIHNhbXBsZSBub25jZQ==" → accept
    // "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=".
    CHECK(ws_accept_key("dGhlIHNhbXBsZSBub25jZQ==") ==
          "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
}

// =====================================================================
// RFC 6455 frame-codec validation (no TCP — drives parse_frame on
// crafted byte buffers).
// =====================================================================

namespace {

// In-memory byte source; vends a `detail::ByteReader` over a buffer.
struct BufReader {
    std::vector<std::uint8_t> bytes;
    std::size_t               pos = 0;

    onebit::echo::detail::ByteReader reader()
    {
        return [this](std::uint8_t* dst, std::size_t n) -> bool {
            if (pos + n > bytes.size()) return false;
            std::memcpy(dst, bytes.data() + pos, n);
            pos += n;
            return true;
        };
    }
};

} // namespace

TEST_CASE("RFC §5.1: server rejects unmasked client frame with 1002")
{
    // FIN | Text, no MASK, len=5, payload "hello"
    BufReader r;
    r.bytes = {0x81U, 0x05U, 'h','e','l','l','o'};
    auto m = onebit::echo::detail::parse_frame(r.reader(),
                                               /*is_server_side=*/true);
    REQUIRE(!m.has_value());
    CHECK(m.error().kind       == onebit::echo::WsError::Kind::Protocol);
    CHECK(m.error().close_code == onebit::echo::kWsCloseProtocolErr);
}

TEST_CASE("RFC §5.2: 64-bit length with MSB=1 rejected with 1002")
{
    // FIN | Binary, MASK=1, len marker=127, ext-len = 0xFFFF...FFFF.
    BufReader r;
    r.bytes = {0x82U, 0xFFU,
               0xFFU, 0xFFU, 0xFFU, 0xFFU,
               0xFFU, 0xFFU, 0xFFU, 0xFFU,
               0x00U, 0x00U, 0x00U, 0x00U /* mask */};
    auto m = onebit::echo::detail::parse_frame(r.reader(),
                                               /*is_server_side=*/true);
    REQUIRE(!m.has_value());
    CHECK(m.error().kind       == onebit::echo::WsError::Kind::Protocol);
    CHECK(m.error().close_code == onebit::echo::kWsCloseProtocolErr);
    // Crucial: we never grew an allocation. The reader was only advanced
    // through the header bytes (no payload bytes consumed).
    CHECK(r.pos <= 10);
}

TEST_CASE("Frame > 64 MiB rejected BEFORE payload alloc")
{
    // FIN | Binary, MASK=1, len marker=127, ext-len = 64 MiB + 1.
    const std::uint64_t big = 64ULL * 1024ULL * 1024ULL + 1ULL;
    BufReader r;
    r.bytes = {0x82U, 0xFFU};
    for (int k = 7; k >= 0; --k) {
        r.bytes.push_back(static_cast<std::uint8_t>(big >> (k * 8)));
    }
    // We deliberately do NOT supply mask or payload bytes. If
    // validate_frame_header doesn't reject up front, parse_frame would
    // try to read the mask, then allocate 64 MiB+1 and EOF.
    auto m = onebit::echo::detail::parse_frame(r.reader(),
                                               /*is_server_side=*/true);
    REQUIRE(!m.has_value());
    CHECK(m.error().kind       == onebit::echo::WsError::Kind::Protocol);
    CHECK(m.error().close_code == onebit::echo::kWsCloseProtocolErr);
    // Reader stopped after the 2-byte header + 8-byte extended length.
    // No mask read, no payload alloc — that's the whole point of the
    // pre-alloc rejection.
    CHECK(r.pos == 10);
}

TEST_CASE("Reserved opcode 0x3 rejected with 1002")
{
    // FIN | opcode=0x3 (reserved data), MASK=1, len=0.
    BufReader r;
    r.bytes = {0x83U, 0x80U, 0x00U, 0x00U, 0x00U, 0x00U};
    auto m = onebit::echo::detail::parse_frame(r.reader(),
                                               /*is_server_side=*/true);
    REQUIRE(!m.has_value());
    CHECK(m.error().kind       == onebit::echo::WsError::Kind::Protocol);
    CHECK(m.error().close_code == onebit::echo::kWsCloseProtocolErr);
}

TEST_CASE("Reserved opcode 0xB rejected with 1002")
{
    // FIN | opcode=0xB (reserved control), MASK=1, len=0.
    BufReader r;
    r.bytes = {0x8BU, 0x80U, 0x00U, 0x00U, 0x00U, 0x00U};
    auto m = onebit::echo::detail::parse_frame(r.reader(),
                                               /*is_server_side=*/true);
    REQUIRE(!m.has_value());
    CHECK(m.error().kind       == onebit::echo::WsError::Kind::Protocol);
    CHECK(m.error().close_code == onebit::echo::kWsCloseProtocolErr);
}

TEST_CASE("Non-zero RSV1 rejected with 1002")
{
    // FIN | RSV1=1 | Text, MASK=1, len=0.
    BufReader r;
    r.bytes = {0xC1U, 0x80U, 0x00U, 0x00U, 0x00U, 0x00U};
    auto m = onebit::echo::detail::parse_frame(r.reader(),
                                               /*is_server_side=*/true);
    REQUIRE(!m.has_value());
    CHECK(m.error().kind       == onebit::echo::WsError::Kind::Protocol);
    CHECK(m.error().close_code == onebit::echo::kWsCloseProtocolErr);
}

TEST_CASE("Fragmented control frame (FIN=0 on Ping) rejected with 1002")
{
    // FIN=0 | Ping, MASK=1, len=0. RFC §5.5.
    BufReader r;
    r.bytes = {0x09U, 0x80U, 0x00U, 0x00U, 0x00U, 0x00U};
    auto m = onebit::echo::detail::parse_frame(r.reader(),
                                               /*is_server_side=*/true);
    REQUIRE(!m.has_value());
    CHECK(m.error().kind       == onebit::echo::WsError::Kind::Protocol);
    CHECK(m.error().close_code == onebit::echo::kWsCloseProtocolErr);
}

TEST_CASE("Masked single-frame Text round-trips through parse_frame")
{
    // FIN | Text, MASK=1, len=5, mask=AA 55 CC 33, payload "hello"^mask.
    BufReader r;
    const std::uint8_t mask[4] = {0xAA, 0x55, 0xCC, 0x33};
    const char         t[]     = "hello";
    r.bytes = {0x81U, 0x85U, mask[0], mask[1], mask[2], mask[3]};
    for (int i = 0; i < 5; ++i) {
        r.bytes.push_back(
            static_cast<std::uint8_t>(t[i]) ^ mask[i & 3]);
    }
    auto m = onebit::echo::detail::parse_frame(r.reader(),
                                               /*is_server_side=*/true);
    REQUIRE(m.has_value());
    CHECK(m->opcode == WsOpcode::Text);
    CHECK(m->text() == "hello");
}

TEST_CASE("base64_decode: RFC 4648 round-trips and 16-byte SWK")
{
    // The RFC-6455 example key decodes to 16 bytes.
    auto v = onebit::echo::detail::base64_decode("dGhlIHNhbXBsZSBub25jZQ==");
    REQUIRE(v.has_value());
    CHECK(v->size() == 16);
    // Garbage cases
    CHECK(!onebit::echo::detail::base64_decode("not_base64!").has_value());
    CHECK(!onebit::echo::detail::base64_decode("AAA").has_value()); // bad len
    CHECK(!onebit::echo::detail::base64_decode("====").has_value()); // all pad
}

// =====================================================================
// Handshake header validation — drives ws_server_handshake over a
// socketpair() so we never bind a real TCP port.
// =====================================================================

namespace {

// Spawn ws_server_handshake on one end of a socketpair, write `req` into
// the other end, then read the server's HTTP response. Returns the first
// status line bytes (e.g. "HTTP/1.1 400 Bad Request").
std::string run_handshake(const std::string& req)
{
    int sv[2];
    REQUIRE(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    std::string resp_status;
    std::thread server_thr([&]() {
        auto r = onebit::echo::ws_server_handshake(sv[0]);
        // We don't care about the result — just that the response
        // reached the peer. Close the server side after responding.
        ::shutdown(sv[0], SHUT_RDWR);
        ::close(sv[0]);
        (void)r;
    });
    ::send(sv[1], req.data(), req.size(), MSG_NOSIGNAL);
    std::string buf;
    char        tmp[1024];
    while (true) {
        const ssize_t n = ::recv(sv[1], tmp, sizeof(tmp), 0);
        if (n <= 0) break;
        buf.append(tmp, tmp + n);
    }
    ::close(sv[1]);
    server_thr.join();
    const auto eol = buf.find("\r\n");
    if (eol != std::string::npos) resp_status = buf.substr(0, eol);
    else                          resp_status = buf;
    return resp_status;
}

} // namespace

TEST_CASE("Handshake without Sec-WebSocket-Version: 13 → 400")
{
    std::string req;
    req += "GET /ws HTTP/1.1\r\n";
    req += "Host: localhost\r\n";
    req += "Upgrade: websocket\r\n";
    req += "Connection: Upgrade\r\n";
    req += "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n";
    // No Sec-WebSocket-Version header.
    req += "\r\n";
    const auto status = run_handshake(req);
    CHECK(status.find("400") != std::string::npos);
    CHECK(status.find("101") == std::string::npos);
}

TEST_CASE("Handshake with wrong Upgrade header → 400")
{
    std::string req;
    req += "GET /ws HTTP/1.1\r\n";
    req += "Host: localhost\r\n";
    req += "Upgrade: h2c\r\n";              // wrong protocol
    req += "Connection: Upgrade\r\n";
    req += "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n";
    req += "Sec-WebSocket-Version: 13\r\n\r\n";
    const auto status = run_handshake(req);
    CHECK(status.find("400") != std::string::npos);
}

TEST_CASE("Handshake with Connection lacking Upgrade token → 400")
{
    std::string req;
    req += "GET /ws HTTP/1.1\r\n";
    req += "Host: localhost\r\n";
    req += "Upgrade: websocket\r\n";
    req += "Connection: keep-alive\r\n";    // missing Upgrade
    req += "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n";
    req += "Sec-WebSocket-Version: 13\r\n\r\n";
    const auto status = run_handshake(req);
    CHECK(status.find("400") != std::string::npos);
}

TEST_CASE("Handshake with non-16-byte Sec-WebSocket-Key → 400")
{
    std::string req;
    req += "GET /ws HTTP/1.1\r\n";
    req += "Host: localhost\r\n";
    req += "Upgrade: websocket\r\n";
    req += "Connection: Upgrade\r\n";
    // "AAAA" decodes to 3 bytes, not 16.
    req += "Sec-WebSocket-Key: AAAA\r\n";
    req += "Sec-WebSocket-Version: 13\r\n\r\n";
    const auto status = run_handshake(req);
    CHECK(status.find("400") != std::string::npos);
}

TEST_CASE("Handshake with all required headers + valid SWK → 101")
{
    std::string req;
    req += "GET /ws HTTP/1.1\r\n";
    req += "Host: localhost\r\n";
    req += "Upgrade: websocket\r\n";
    req += "Connection: keep-alive, Upgrade\r\n";   // multi-token OK
    req += "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n";
    req += "Sec-WebSocket-Version: 13\r\n\r\n";
    const auto status = run_handshake(req);
    CHECK(status.find("101") != std::string::npos);
}

// =====================================================================
// End-to-end WS protocol smoke test against a real EchoServer.
// We inject a fake SSE body + fake TTS into a VoicePipeline so the
// preamble and chunks actually fire without network or kokoro running.
//
// The trick is: the server constructs its own VoicePipeline per
// connection from the EchoConfig. To still exercise the chunk path with
// fake data, we point voice_cfg at unreachable URLs AND we hijack the
// pipeline by building it inline. That requires test plumbing the C++
// port adds: VoicePipeline supports `inject_test_sse`. The server
// constructs a fresh pipeline per connection, so to make the e2e flow
// observable we set voice_cfg.llm_url to "http://127.0.0.1:1/..." and
// rely on the preamble being emitted BEFORE the LLM request fires —
// that part the C++ server preserves from the Rust crate.
// =====================================================================

namespace {

// Trivial WS client: opens TCP, runs a handshake, sends one Text frame,
// drains replies until it sees the first Text frame (the preamble).
struct WsClient {
    int fd = -1;

    static int connect_to(const std::string& host, std::uint16_t port)
    {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return -1;
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port   = ::htons(port);
        ::inet_pton(AF_INET, host.c_str(), &a.sin_addr);
        if (::connect(fd, reinterpret_cast<sockaddr*>(&a), sizeof(a)) < 0) {
            ::close(fd);
            return -1;
        }
        return fd;
    }

    bool handshake(const std::string& host, std::uint16_t port)
    {
        std::string req;
        req += "GET /ws HTTP/1.1\r\n";
        req += "Host: " + host + ":" + std::to_string(port) + "\r\n";
        req += "Upgrade: websocket\r\n";
        req += "Connection: Upgrade\r\n";
        req += "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n";
        req += "Sec-WebSocket-Version: 13\r\n\r\n";
        ::send(fd, req.data(), req.size(), MSG_NOSIGNAL);
        std::string buf;
        char        tmp[1024];
        while (buf.find("\r\n\r\n") == std::string::npos) {
            ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
            if (n <= 0) return false;
            buf.append(tmp, tmp + n);
        }
        return buf.find(" 101 ") != std::string::npos &&
               buf.find("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=") !=
                   std::string::npos;
    }

    void send_text(std::string_view t)
    {
        std::vector<std::uint8_t> frame;
        frame.push_back(0x80U | 0x01U);   // FIN | Text
        // Client must mask.
        const std::size_t len = t.size();
        if (len < 126) {
            frame.push_back(static_cast<std::uint8_t>(0x80U | len));
        } else {
            frame.push_back(0x80U | 126U);
            frame.push_back(static_cast<std::uint8_t>(len >> 8));
            frame.push_back(static_cast<std::uint8_t>(len & 0xFFU));
        }
        std::uint8_t mask[4] = {0xAA, 0x55, 0xCC, 0x33};
        frame.insert(frame.end(), mask, mask + 4);
        for (std::size_t i = 0; i < len; ++i) {
            frame.push_back(static_cast<std::uint8_t>(t[i]) ^ mask[i & 3]);
        }
        ::send(fd, frame.data(), frame.size(), MSG_NOSIGNAL);
    }

    bool recv_frame(std::vector<std::uint8_t>& out, std::uint8_t& opcode)
    {
        std::uint8_t hdr[2];
        if (::recv(fd, hdr, 2, MSG_WAITALL) != 2) return false;
        opcode = hdr[0] & 0x0F;
        std::uint64_t len = hdr[1] & 0x7F;
        if (len == 126) {
            std::uint8_t e[2];
            if (::recv(fd, e, 2, MSG_WAITALL) != 2) return false;
            len = (static_cast<std::uint64_t>(e[0]) << 8) | e[1];
        } else if (len == 127) {
            std::uint8_t e[8];
            if (::recv(fd, e, 8, MSG_WAITALL) != 8) return false;
            len = 0;
            for (int k = 0; k < 8; ++k) {
                len = (len << 8) | e[k];
            }
        }
        out.resize(static_cast<std::size_t>(len));
        if (len > 0) {
            const ssize_t r =
                ::recv(fd, out.data(), out.size(), MSG_WAITALL);
            if (r != static_cast<ssize_t>(len)) return false;
        }
        return true;
    }

    ~WsClient() { if (fd >= 0) ::close(fd); }
};

} // namespace

TEST_CASE("e2e: server emits text preamble before pipeline runs")
{
    EchoConfig cfg{};
    cfg.bind_host          = "127.0.0.1";
    cfg.bind_port          = 0;
    cfg.codec              = Codec::Pcm;
    // Point at unreachable backends; preamble fires BEFORE the pipeline.
    cfg.voice_cfg.llm_url     = "http://127.0.0.1:1/v1/chat/completions";
    cfg.voice_cfg.tts_url     = "http://127.0.0.1:1/tts";
    cfg.voice_cfg.timeout_secs = 1;

    EchoServer server(std::move(cfg));
    auto port = server.listen();
    REQUIRE(port.has_value());

    std::thread srv_t([&]() { server.handle_one_for_tests(); });

    WsClient cli;
    cli.fd = WsClient::connect_to("127.0.0.1", *port);
    REQUIRE(cli.fd >= 0);
    REQUIRE(cli.handshake("127.0.0.1", *port));
    cli.send_text("hello");

    std::vector<std::uint8_t> body;
    std::uint8_t              opcode = 0;
    REQUIRE(cli.recv_frame(body, opcode));
    CHECK(opcode == 0x1);          // Text
    auto v = json::parse(std::string(body.begin(), body.end()));
    CHECK(v["codec"]       == "pcm");
    CHECK(v["sample_rate"] == 48000);
    CHECK(v["channels"]    == 1);
    CHECK(v["frame_ms"]    == 20);

    // Cleanup: drop client; server sees EOF on the watcher recv loop.
    ::close(cli.fd);
    cli.fd = -1;
    server.stop();
    srv_t.join();
}
