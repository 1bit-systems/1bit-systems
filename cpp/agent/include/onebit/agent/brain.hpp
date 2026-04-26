#pragma once

// 1bit-agent — brain client. POSTs OpenAI-compat /v1/chat/completions
// at lemond (default :8180) and returns either final text or a list of
// tool_calls.
//
// Two transport modes:
//   * non-streaming: one request -> one JSON body -> BrainReply.
//   * streaming SSE: same endpoint with stream=true; we accumulate the
//     `delta.content` string and any `tool_calls` deltas into a single
//     BrainReply. Streaming exists so a future adapter (web, voice)
//     can subscribe to chunks without re-implementing the SSE parser.
//
// The dispatch path is synchronous per turn (one outstanding HTTP
// request at a time). Threads belong to the adapter recv loop, never
// here.

#include "onebit/agent/error.hpp"
#include "onebit/agent/event.hpp"

#include <chrono>
#include <expected>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace onebit::agent {

// One message in the OpenAI-compat history window.
struct ChatMessage {
    std::string role;          // "system" | "user" | "assistant" | "tool"
    std::string content;
    std::string tool_call_id;  // role=tool only
    std::vector<ToolCall> tool_calls; // role=assistant only
};

// One inference turn input.
struct BrainRequest {
    std::vector<ChatMessage> history;
    std::vector<nlohmann::json> tools;     // OpenAI tools[] schema
    std::string                 model;
    double                      temperature = 0.2;
    bool                        stream      = false;
    std::chrono::milliseconds   timeout{60000};
};

// Optional per-chunk callback for streaming. Only called during
// stream=true requests. Errors thrown from the callback are caught
// and converted to AgentError::brain(...).
using StreamCallback = std::function<void(std::string_view delta)>;

// IBrain: virtual seam so the loop can be exercised with a stub.
// Concrete production impl is `Brain` below.
class IBrain {
public:
    virtual ~IBrain() = default;

    [[nodiscard]] virtual std::expected<BrainReply, AgentError>
    chat(const BrainRequest& req, const StreamCallback& on_chunk = {}) = 0;
};

class Brain final : public IBrain {
public:
    explicit Brain(std::string base_url);
    ~Brain() override;
    Brain(const Brain&)            = delete;
    Brain& operator=(const Brain&) = delete;
    Brain(Brain&&) noexcept;
    Brain& operator=(Brain&&) noexcept;

    [[nodiscard]] const std::string& base_url() const noexcept;

    // Non-streaming. Single POST; parses the JSON body.
    [[nodiscard]] std::expected<BrainReply, AgentError>
    complete(const BrainRequest& req);

    // Streaming. POST with stream=true; consumes SSE; calls `on_chunk`
    // with each text delta as it arrives, then returns the assembled
    // BrainReply (text + any tool_calls deltas).
    [[nodiscard]] std::expected<BrainReply, AgentError>
    complete_streaming(const BrainRequest& req, const StreamCallback& on_chunk);

    // Dispatches to the streaming or non-streaming path based on
    // req.stream. Adapter-friendly single entry point.
    [[nodiscard]] std::expected<BrainReply, AgentError>
    chat(const BrainRequest& req, const StreamCallback& on_chunk = {}) override;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// Helpers (free functions; unit-tested standalone).
[[nodiscard]] nlohmann::json
build_request_body(const BrainRequest& req);

// Parses one full /v1/chat/completions response body (non-streaming).
[[nodiscard]] std::expected<BrainReply, AgentError>
parse_response_body(std::string_view json_body);

// Parses one SSE line of the form "data: {...}" or "data: [DONE]" and
// applies its delta to `acc`. Mirrors helm/src/stream.cpp parser shape
// but extended to also surface assistant-side `tool_calls` deltas.
struct SseStep {
    std::string text_delta;          // empty if this line carried only tool deltas
    bool        done = false;        // line was "data: [DONE]"
    bool        had_tool_delta = false;
};
[[nodiscard]] SseStep
apply_sse_line(std::string_view line, BrainReply& acc);

} // namespace onebit::agent
