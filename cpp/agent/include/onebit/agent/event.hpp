#pragma once

// 1bit-agent — pure event types crossing the adapter / brain / tool /
// loop seams. Header-only; depends on STL + nlohmann/json only so
// adapters and tool registries (sibling agents) can include this
// without pulling the loop in.

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace onebit::agent {

// One inbound message off an adapter (Discord DM, Telegram, web).
// channel = surface-specific routing identifier (DM channel id, room,
// websocket session). user_id is opaque; we never key on user_name.
struct Attachment {
    std::string url;
    std::string mime_type;
    std::uint64_t bytes = 0;
};

struct IncomingMessage {
    std::string channel;
    std::string user_id;
    std::string user_name;
    std::string text;
    std::vector<Attachment> attachments;
};

// One tool call requested by the brain. args_json is the raw JSON
// object the model emitted; the tool registry validates the schema.
struct ToolCall {
    std::string id;          // OpenAI tool_call_id; round-trips back in the result
    std::string name;
    nlohmann::json args_json;
};

struct ToolResult {
    bool        success = false;
    std::string content; // text body; gets stuffed into role=tool message
};

// One brain response. Either text (final answer for this turn) or a
// non-empty tool_calls list (loop must execute, feed back, re-prompt).
// Both fields populated is allowed by the OpenAI spec; loop treats any
// non-empty tool_calls as "tool round" and ignores text until the
// follow-up turn.
struct BrainReply {
    std::string text;
    std::vector<ToolCall> tool_calls;
};

// One outbound reply destined for an adapter. tool_calls retained for
// tests + audit; adapters only need text.
struct OutgoingReply {
    std::string text;
    std::vector<ToolCall> tool_calls;
};

} // namespace onebit::agent
