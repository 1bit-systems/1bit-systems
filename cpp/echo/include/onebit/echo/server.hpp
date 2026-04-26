#pragma once

// 1bit-echo — browser-side WebSocket gateway over 1bit-voice.
//
// Wire protocol (mirrors crates/1bit-echo/src/server.rs verbatim):
//
//   1. Client opens WS to /ws.
//   2. Client sends ONE Text frame with the LLM prompt.
//   3. Server replies with ONE Text frame: a JSON preamble
//        { "sample_rate": 24000|48000, "channels": 1,
//          "frame_ms": 0|20, "codec": "wav"|"pcm" }
//      Same key names as the Rust crate; "pcm" replaces "opus" because
//      this port emits raw s16le frames instead of Opus packets (see
//      cpp/echo/include/onebit/echo/codec.hpp for the rationale).
//   4. For each VoiceChunk from 1bit-voice, the server emits Binary
//      frames: in Wav mode one frame = the RIFF blob; in Pcm mode
//      one frame = one 20 ms s16le mono PCM chunk at 48 kHz.
//   5. Either party may send Close to terminate.
//
// JSON cancel sentinel (also from Rust): a Text frame containing
// `"cancel"` aborts the in-flight pipeline.
//
// Threading: each accepted socket runs on its own std::thread. Cleanup
// happens when the thread exits. No connection limit (server is bound
// to loopback by default).

#include "onebit/voice/pipeline.hpp"

#include <atomic>
#include <cstdint>
#include <expected>
#include <memory>
#include <string>
#include <string_view>

namespace onebit::echo {

enum class Codec : std::uint8_t {
    Wav,
    Pcm,
};

[[nodiscard]] std::string_view codec_name(Codec c) noexcept;
[[nodiscard]] std::expected<Codec, std::string>
codec_from_str(std::string_view s) noexcept;

[[nodiscard]] std::string preamble_json(Codec codec);

struct EchoConfig {
    std::string         bind_host = "127.0.0.1";
    std::uint16_t       bind_port = 8085;
    onebit::voice::VoiceConfig voice_cfg{};
    Codec               codec     = Codec::Wav;
};

// Forward; see server.cpp for impl. pImpl per Core Guidelines I.27.
class EchoServer {
public:
    explicit EchoServer(EchoConfig cfg);
    ~EchoServer();
    EchoServer(const EchoServer&)            = delete;
    EchoServer& operator=(const EchoServer&) = delete;
    EchoServer(EchoServer&&) noexcept;
    EchoServer& operator=(EchoServer&&) noexcept;

    [[nodiscard]] const EchoConfig& config() const noexcept;

    // Bind + listen. Returns the actual port (useful when bind_port == 0).
    [[nodiscard]] std::expected<std::uint16_t, std::string> listen();

    // Block in the accept loop until stop() is called or the listening
    // socket is closed. Spawns a worker std::thread per connection.
    void serve_forever();

    // Asynchronous stop hook: closes the listening socket so accept()
    // unblocks. Safe to call from any thread.
    void stop();

    // Convenience: listen() + serve_forever() in one call. Returns
    // unexpected if listen() failed; otherwise returns when stop() runs.
    [[nodiscard]] std::expected<void, std::string> run();

    // For tests: handle exactly one accepted connection synchronously
    // then return. Caller is responsible for connecting a client.
    void handle_one_for_tests();

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace onebit::echo
