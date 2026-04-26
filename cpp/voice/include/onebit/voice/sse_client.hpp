#pragma once

// Minimal streaming SSE client.
//
// cpp-httplib's POST API does not expose a streaming response handler,
// so we own a small TCP-based POST → response-stream loop with one
// purpose only: feed an OpenAI-compat /v1/chat/completions SSE response
// to a callback as bytes arrive, with NO per-event copy of the body.
//
// HTTPS not supported here — runtime path is loopback to lemond on
// 127.0.0.1:8180. If a remote LLM is wanted later, route via a local
// HTTPS-capable client (httplib::SSLClient is fine for non-streaming).

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <string>
#include <string_view>

namespace onebit::voice {

struct SseError {
    enum class Kind { Connect, Resolve, Send, Recv, Status, Closed };
    Kind        kind;
    int         http_status = 0;   // valid when kind == Status
    std::string message;
};

// Streamed POST: opens TCP, writes request, reads chunked/identity body,
// invokes `on_event` for each SSE event block (text between two
// consecutive blank lines, NO trailing "\n\n"). `on_event` is given a
// non-owning std::string_view; the underlying buffer is the streamer's
// internal scratch — DO NOT retain past the callback.
//
// Returning false from `on_event` aborts the read and returns a clean
// success.
//
// Headers: Content-Type defaults to application/json; Accept set to
// text/event-stream. body is sent as-is.
[[nodiscard]] std::expected<void, SseError>
post_sse(std::string_view  host,
         std::uint16_t     port,
         std::string_view  path,
         std::string_view  body,
         std::uint32_t     timeout_secs,
         const std::function<bool(std::string_view event)>& on_event);

} // namespace onebit::voice
