// 1bit.cpp — chat-template renderer.
//
// Byte-exact C++20 mirror of `crates/1bit-server/src/chat_template.rs`. The
// HTTP leak bench traced one of the sampler-pipe IPC costs back to Rust
// `String` allocation inside the template renderer; this C++ port lives on
// `std::string_view` inputs and a single pre-reserved `std::string` output
// so the hot path never touches malloc after the initial reserve.
//
// Variants:
//   Llama3 — canonical; wrap every turn in
//            `<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>`
//            and append a trailing assistant marker. Wire-compatible with
//            OpenAI SDKs.
//   Short  — drop header framing; concatenate `{content}<|eot_id|>` per turn.
//   Raw    — pass-through concatenation, no framing.
//
// Sanitization: any `<|...|>` sequence in user-supplied content is replaced
// with the 10-byte UTF-8 literal `«scrubbed»` (same bytes as the Rust side,
// so the test can diff the output verbatim). Rationale matches the Rust doc
// — prevents role-impersonation via injected special-token markers.

#ifndef ONEBIT_CPP_CHAT_TEMPLATE_HPP
#define ONEBIT_CPP_CHAT_TEMPLATE_HPP

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

namespace onebit::cpp {

struct ChatMessage {
    std::string role;
    std::string content;
};

enum class ChatTemplate {
    Llama3,
    Short,
    Raw,
};

// Map header / env-var text (case-insensitive) to a template. Unknown values
// return `false`; caller decides whether to fall back to Llama3 or surface
// an error. Mirrors `ChatTemplate::from_str_opt` in the Rust side.
bool chat_template_from_str(std::string_view s, ChatTemplate& out) noexcept;

// Read `HALO_CHAT_TEMPLATE` once. Unknown value → log-to-stderr + Llama3
// default (matches Rust behaviour; we don't have `tracing::warn!` here so
// stderr is the pragmatic swap).
ChatTemplate chat_template_from_env() noexcept;

// Render the message list into a prompt under `tpl`. See module comment for
// the exact bytes each variant emits. Output is freshly allocated per call;
// the result is safe to move into the tokenizer.
std::string chat_template_render(ChatTemplate tpl,
                                 std::span<const ChatMessage> messages);

// Exposed for the unit test. Scrubs `<|...|>` sequences → `«scrubbed»`.
std::string sanitize(std::string_view in);

// Shorthand convenience for the three variants. Kept as free functions so
// the call site reads `onebit::cpp::llama3_render(msgs)` rather than a
// redundant `render(Llama3, msgs)` — mirrors the Rust module surface.
std::string llama3_render(std::span<const ChatMessage> messages);
std::string short_render(std::span<const ChatMessage> messages);
std::string raw_render(std::span<const ChatMessage> messages);

}  // namespace onebit::cpp

#endif  // ONEBIT_CPP_CHAT_TEMPLATE_HPP
