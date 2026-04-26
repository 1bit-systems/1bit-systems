// SPDX-License-Identifier: Apache-2.0
//
// 1bit-agent — minimal Discord gateway WebSocket client.
//
// We have a hand-rolled RFC 6455 SERVER in cpp/echo/src/ws.cpp. This is
// the CLIENT-side mirror, narrowed to exactly what Discord's gateway
// needs: TLS to wss://gateway.discord.gg/?v=10&encoding=json, IDENTIFY
// (op 2), HEARTBEAT (op 1) at the HELLO interval, RESUME (op 6) on
// disconnect, sequence-number tracking, MESSAGE_CREATE event parsing.
//
// Why hand-rolled: cpp-httplib does not ship WebSocket support and the
// official C++ Discord SDK (discordpp) is too heavy for a token-of-
// trust standpoint and bloats the static binary. RFC 6455 framing fits
// in ~300 LOC; we already proved it on the server side.
//
// TLS: raw OpenSSL (libssl 3.x). cpp-httplib's SSLClient wraps the
// same primitives but closes the connection per request, which
// defeats a long-lived gateway socket. We open SSL_CTX once,
// upgrade-handshake over the same TLS BIO, then read/write framed
// payloads on it forever.
//
// Threading: a single DiscordGateway owns one connection; recv_event()
// blocks the calling thread. Heartbeat lives on a sibling std::jthread
// inside DiscordGateway::Impl. All sends serialize on a mutex.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace onebit::agent::discord {

// RFC 6455 §5.2 — opcodes we care about. Discord only uses Text frames
// at the WS layer; the application-level "op" field is in the JSON
// payload (see GatewayOp below).
enum class WsOpcode : std::uint8_t {
    Continuation = 0x0,
    Text         = 0x1,
    Binary       = 0x2,
    Close        = 0x8,
    Ping         = 0x9,
    Pong         = 0xA,
};

// Discord gateway opcodes (https://discord.com/developers/docs/topics/
// opcodes-and-status-codes).
enum class GatewayOp : int {
    Dispatch            = 0,
    Heartbeat           = 1,
    Identify            = 2,
    PresenceUpdate      = 3,
    VoiceStateUpdate    = 4,
    Resume              = 6,
    Reconnect           = 7,
    RequestGuildMembers = 8,
    InvalidSession      = 9,
    Hello               = 10,
    HeartbeatAck        = 11,
};

// Discord gateway intents bits we use. The brief calls for
// GUILDS | GUILD_MESSAGES | DIRECT_MESSAGES | MESSAGE_CONTENT.
inline constexpr std::uint32_t kIntentGuilds          = 1U << 0;
inline constexpr std::uint32_t kIntentGuildMessages   = 1U << 9;
inline constexpr std::uint32_t kIntentDirectMessages  = 1U << 12;
inline constexpr std::uint32_t kIntentMessageContent  = 1U << 15;
inline constexpr std::uint32_t kIntentDefault =
    kIntentGuilds | kIntentGuildMessages |
    kIntentDirectMessages | kIntentMessageContent;

// Strongly-typed transport error. Mirrors echo::WsError but with
// gateway-specific kinds folded in so the adapter can branch on
// "session invalidated → must re-IDENTIFY" without string-matching.
struct GatewayError {
    enum class Kind {
        Tls,            // OpenSSL handshake / read / write failure
        Handshake,      // HTTP-101 upgrade rejected
        Protocol,       // RFC 6455 violation (RSV bits, oversized frame, ...)
        Closed,         // peer sent Close or socket EOF
        SessionInvalid, // op 9 with d=false → re-IDENTIFY required
        ResumeFailed,   // op 9 with d=true on resume → re-IDENTIFY required
        Json,           // payload not valid JSON or schema mismatch
        Timeout,        // heartbeat ack missed twice
    };
    Kind          kind;
    std::string   message;
    std::uint16_t close_code = 0; // RFC 6455 §5.5.1 if Kind == Closed
};

// One inbound dispatch event. We only surface MESSAGE_CREATE today;
// other event types are dropped silently by recv_event() so callers
// don't block forever waiting for a Hello-scoped READY.
struct GatewayEvent {
    std::string    type;       // e.g. "MESSAGE_CREATE"
    nlohmann::json data;       // raw "d" object
    std::int64_t   sequence;   // "s" field; -1 if absent (non-dispatch)
};

// Stateful resume bookkeeping. Lives in the gateway between connection
// attempts. session_id + last sequence let us send op 6 on reconnect;
// resume_url is the per-session URL Discord hands us in the READY
// payload (it overrides gateway.discord.gg).
struct ResumeState {
    std::string  session_id;
    std::string  resume_url;
    std::int64_t last_sequence = -1;
};

struct GatewayConfig {
    std::string   token;                         // bot token (NOT logged)
    std::uint32_t intents = kIntentDefault;
    std::string   os      = "linux";
    std::string   browser = "halo-agent";
    std::string   device  = "halo-agent";
    // Override gateway URL (defaults to wss://gateway.discord.gg).
    // The READY payload may hand us a session-specific URL we MUST use
    // for resumes; recv_event() updates this automatically.
    std::string   gateway_host = "gateway.discord.gg";
    std::uint16_t gateway_port = 443;
    std::string   gateway_path = "/?v=10&encoding=json";
};

// Pure framer — pulled out of Gateway::Impl for testability. Builds a
// CLIENT-side RFC 6455 frame: FIN | opcode, then masked payload (per
// §5.1, all client → server frames MUST be masked). The mask key is
// drawn from the supplied 4-byte buffer; tests pass a deterministic
// key, production passes random bytes.
[[nodiscard]] std::vector<std::uint8_t>
build_client_frame(WsOpcode             opcode,
                   const std::uint8_t*  payload,
                   std::size_t          len,
                   const std::uint8_t   mask_key[4]);

// Helper for tests + production: produce a 4-byte mask using the
// supplied seed. Production uses /dev/urandom; tests can pass a
// fixed seed for reproducibility.
void fill_mask_key(std::uint8_t out[4]);

// Pure JSON serializers — separated so doctest can assert on the wire
// payload without standing up a TLS server.
[[nodiscard]] std::string serialize_identify(const GatewayConfig& cfg);
[[nodiscard]] std::string serialize_heartbeat(std::int64_t last_seq);
[[nodiscard]] std::string serialize_resume(const std::string& token,
                                           const std::string& session_id,
                                           std::int64_t       last_seq);

// Parse one inbound JSON gateway frame into op / sequence / type / data.
// Returns Json kind on malformed input. Used by recv_event() and tested
// in isolation.
struct ParsedFrame {
    GatewayOp     op;
    std::int64_t  sequence = -1;
    std::string   type;
    nlohmann::json data;
};
[[nodiscard]] std::expected<ParsedFrame, GatewayError>
parse_gateway_frame(std::string_view json_text);

// Frame-header parsing: server → client frames are NOT masked (RFC 6455
// §5.1). Returns the resolved length + opcode + offset where the
// payload begins. Caller hands us at least the first 14 bytes.
struct FrameHeader {
    WsOpcode      opcode;
    bool          fin;
    std::uint64_t payload_len;
    std::size_t   header_size;   // bytes consumed (2, 4, or 10)
};
[[nodiscard]] std::expected<FrameHeader, GatewayError>
parse_frame_header(const std::uint8_t* buf, std::size_t available);

// The gateway client itself. pImpl so the OpenSSL state stays out of
// the public header (Rule F / I.27).
class DiscordGateway {
public:
    explicit DiscordGateway(GatewayConfig cfg);
    ~DiscordGateway();

    DiscordGateway(const DiscordGateway&)            = delete;
    DiscordGateway& operator=(const DiscordGateway&) = delete;
    DiscordGateway(DiscordGateway&&) noexcept;
    DiscordGateway& operator=(DiscordGateway&&) noexcept;

    // Open TCP, complete TLS handshake, complete HTTP-101 WebSocket
    // upgrade, read HELLO, send IDENTIFY (or RESUME if state present),
    // start the heartbeat thread. Idempotent: safe to call after a
    // disconnect to reconnect.
    [[nodiscard]] std::expected<void, GatewayError> connect();

    // Block reading from the socket until a Dispatch event arrives.
    // Internally consumes Hello, HeartbeatAck, Reconnect, and
    // InvalidSession opcodes; only Dispatch escapes to the caller.
    // Reconnect / InvalidSession surface as the corresponding
    // GatewayError kinds so the caller (DiscordAdapter) can backoff +
    // re-connect with the right strategy.
    [[nodiscard]] std::expected<GatewayEvent, GatewayError>
    recv_event(std::chrono::milliseconds timeout);

    // Asynchronous shutdown — closes the socket so recv_event unblocks
    // with Kind::Closed. Idempotent and thread-safe.
    void stop() noexcept;

    // Resume bookkeeping (read-only externally).
    [[nodiscard]] const ResumeState& resume_state() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace onebit::agent::discord
