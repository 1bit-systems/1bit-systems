// 1bit-helm — OpenAI-compatible SSE stream parser.
//
// One pure function: walk one logical SSE line, return a parser
// outcome. Caller does framing. Mirrors crates/1bit-helm/src/stream.rs.

#pragma once

#include <string>
#include <string_view>
#include <variant>

namespace onebit::helm {

// One logical parser outcome for a single SSE line.
struct SseDelta { std::string content; };
struct SseDone {};
struct SseIgnore {};

using SseEvent = std::variant<SseIgnore, SseDelta, SseDone>;

[[nodiscard]] SseEvent parse_sse_line(std::string_view line);

// Helpers — pull-out predicates so test sites + tray code don't have
// to spell std::holds_alternative.
[[nodiscard]] inline bool is_done(const SseEvent& e) noexcept
{
    return std::holds_alternative<SseDone>(e);
}
[[nodiscard]] inline bool is_delta(const SseEvent& e) noexcept
{
    return std::holds_alternative<SseDelta>(e);
}

} // namespace onebit::helm
