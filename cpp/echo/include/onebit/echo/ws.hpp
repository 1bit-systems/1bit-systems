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

struct WsError {
    enum class Kind { Recv, Send, Closed, Protocol, Handshake };
    Kind        kind;
    std::string message;
};

// Compute the Sec-WebSocket-Accept header value for a given key.
[[nodiscard]] std::string ws_accept_key(std::string_view sec_websocket_key);

// Server-side handshake: read the HTTP upgrade request from `fd`, validate
// it, send the 101 response. Returns the parsed request path on success.
[[nodiscard]] std::expected<std::string, WsError> ws_server_handshake(int fd);

// Read the next frame. Handles ping → pong replies internally; control
// frames returned only for Close. Continues reading until a complete
// non-control message is received.
[[nodiscard]] std::expected<WsMessage, WsError> ws_recv(int fd);

// Send a Text frame.
[[nodiscard]] std::expected<void, WsError>
ws_send_text(int fd, std::string_view text);

// Send a Binary frame.
[[nodiscard]] std::expected<void, WsError>
ws_send_binary(int fd, const std::uint8_t* data, std::size_t len);

// Send a Close frame (no status code; cheap polite teardown).
[[nodiscard]] std::expected<void, WsError> ws_send_close(int fd);

} // namespace onebit::echo
