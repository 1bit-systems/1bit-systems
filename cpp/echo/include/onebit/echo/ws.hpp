#pragma once

// Minimal RFC 6455 WebSocket helpers.
//
// cpp-httplib does not ship WebSocket support today. This module owns a
// hand-rolled server-side handshake + frame codec — just enough to
// receive Text/Binary frames from a browser client and send Text/Binary
// or Close frames back. No client side, no extensions, no
// fragmentation-on-send (we only emit single-frame messages), no SSL.
//
// Threading: a single connection is owned by a single thread. Multiple
// connections live in different std::thread instances.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::echo {

enum class WsOpcode : std::uint8_t {
    Continuation = 0x0,
    Text         = 0x1,
    Binary       = 0x2,
    Close        = 0x8,
    Ping         = 0x9,
    Pong         = 0xA,
};

struct WsMessage {
    WsOpcode                  opcode  = WsOpcode::Text;
    std::vector<std::uint8_t> payload;
    [[nodiscard]] std::string_view text() const noexcept {
        return std::string_view{reinterpret_cast<const char*>(payload.data()),
                                payload.size()};
    }
};

// RFC 6455 §7.4 close codes we use at the framing layer.
inline constexpr std::uint16_t kWsCloseNormal       = 1000;
inline constexpr std::uint16_t kWsCloseProtocolErr  = 1002;
inline constexpr std::uint16_t kWsCloseTooBig       = 1009;

struct WsError {
    enum class Kind { Recv, Send, Closed, Protocol, Handshake };
    Kind          kind;
    std::string   message;
    // For Protocol errors, the close code we should send to the peer.
    // Zero = no code in the close frame. Defaults to 1002 for Protocol.
    std::uint16_t close_code = 0;
};

// Compute the Sec-WebSocket-Accept header value for a given key.
[[nodiscard]] std::string ws_accept_key(std::string_view sec_websocket_key);

// Server-side handshake: read the HTTP upgrade request from `fd`, validate
// it (RFC 6455 §4.2.1: GET, Upgrade: websocket, Connection: Upgrade,
// Sec-WebSocket-Version: 13, base64 16-byte Sec-WebSocket-Key), send the
// 101 response. Returns the parsed request path on success. On any
// validation failure the server emits a 400 response and returns a
// Handshake error.
[[nodiscard]] std::expected<std::string, WsError> ws_server_handshake(int fd);

// Read the next frame. Handles ping → pong replies internally; control
// frames returned only for Close. Continues reading until a complete
// non-control message is received. On RFC 6455 protocol violations
// (unmasked client frame, reserved opcode, RSV bits set, 64-bit length
// MSB=1, frame > 64 MiB) sends a Close 1002 frame BEFORE returning the
// error so the peer learns why.
[[nodiscard]] std::expected<WsMessage, WsError> ws_recv(int fd);

// Send a Text frame.
[[nodiscard]] std::expected<void, WsError>
ws_send_text(int fd, std::string_view text);

// Send a Binary frame.
[[nodiscard]] std::expected<void, WsError>
ws_send_binary(int fd, const std::uint8_t* data, std::size_t len);

// Send a Close frame with no status code.
[[nodiscard]] std::expected<void, WsError> ws_send_close(int fd);

// Send a Close frame with an RFC 6455 status code (§5.5.1).
[[nodiscard]] std::expected<void, WsError>
ws_send_close_code(int fd, std::uint16_t code);

// =====================================================================
// Test-only entry points.
// Exposed publicly so doctest can drive the codec without opening sockets.
// `detail::ByteReader` is a callable that fills `n` bytes into `dst` and
// returns true on success / false on EOF or error. Tests pass a reader
// backed by a `std::vector<std::uint8_t>` cursor.
// =====================================================================
namespace detail {

using ByteReader = std::function<bool(std::uint8_t* dst, std::size_t n)>;

// Maximum payload size we will accept on a single frame (DoS bound).
inline constexpr std::uint64_t kMaxFramePayload = 64ULL * 1024ULL * 1024ULL;

// Validate the bits drawn from the first two header bytes plus the
// already-decoded length. Returns Protocol with close_code=1002 on any
// RFC 6455 §5.2 violation. `is_server_side` mandates the client→server
// MASK=1 rule (§5.1).
[[nodiscard]] std::expected<void, WsError>
validate_frame_header(std::uint8_t  b0,
                      std::uint8_t  b1,
                      std::uint64_t payload_len,
                      bool          is_server_side);

// Parse a single frame from `read`. On Protocol violations returns the
// error WITHOUT having dispatched a close frame (the caller is the one
// holding the fd). Allocates payload only after the length cap clears.
[[nodiscard]] std::expected<WsMessage, WsError>
parse_frame(const ByteReader& read, bool is_server_side);

// base64 decode (strict, RFC 4648). Used to validate Sec-WebSocket-Key
// must decode to exactly 16 bytes per RFC 6455 §4.1.
[[nodiscard]] std::expected<std::vector<std::uint8_t>, std::string>
base64_decode(std::string_view in);

} // namespace detail
} // namespace onebit::echo
