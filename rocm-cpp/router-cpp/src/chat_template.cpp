// 1bit.cpp — chat-template renderer (impl).
//
// Byte-exact mirror of `crates/1bit-server/src/chat_template.rs`. The
// smoke test round-trips an `"hi"` user message through `llama3_render`
// and diffs the bytes against the Rust reference string. Zero diff is
// load-bearing — if this diverges, the HTTP server switching to this
// path will also diverge, and the shadow-burnin harness will start
// flagging every request.

#include "onebit_cpp/chat_template.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

namespace onebit::cpp {

namespace {

// Llama-3 template framing.
constexpr std::string_view kLlama3HeaderOpen  = "<|start_header_id|>";
constexpr std::string_view kLlama3HeaderClose = "<|end_header_id|>\n\n";
constexpr std::string_view kLlama3EotId       = "<|eot_id|>";
constexpr std::string_view kLlama3AsstTail    =
    "<|start_header_id|>assistant<|end_header_id|>\n\n";

// `«scrubbed»` as UTF-8 bytes. Mirrors the Rust literal; both « and » are
// 2-byte code points so the sequence is 10 bytes total.
// c2 ab s c r u b b e d c2 bb  → 10 bytes (« + "scrubbed" + »)
constexpr std::string_view kScrub = "\xc2\xabscrubbed\xc2\xbb";

// ASCII-only lowercase — matches Rust's `to_ascii_lowercase` exactly.
std::string ascii_lower_trim(std::string_view s) {
    // Trim leading/trailing whitespace (ASCII; Rust `.trim()` trims
    // Unicode whitespace but the accepted values are ASCII-only, so
    // this is equivalent for the input set we advertise).
    std::size_t begin = 0;
    std::size_t end   = s.size();
    while (begin < end && static_cast<unsigned char>(s[begin]) <= 0x20) ++begin;
    while (end > begin && static_cast<unsigned char>(s[end - 1]) <= 0x20) --end;

    std::string out;
    out.reserve(end - begin);
    for (std::size_t i = begin; i < end; ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c >= 'A' && c <= 'Z') c = static_cast<unsigned char>(c + ('a' - 'A'));
        out.push_back(static_cast<char>(c));
    }
    return out;
}

}  // namespace

std::string sanitize(std::string_view in) {
    std::string out;
    out.reserve(in.size());
    const char* data = in.data();
    const std::size_t n = in.size();

    std::size_t i = 0;
    while (i < n) {
        // Look for `<|`. If found, look for the closing `|>` AFTER the
        // `<|` (two-byte skip to match Rust `s[i + 2..].find("|>")`).
        if (i + 1 < n && data[i] == '<' && data[i + 1] == '|') {
            const char* search_begin = data + i + 2;
            const std::size_t search_len = n - (i + 2);
            std::string_view remainder(search_begin, search_len);
            std::size_t end = remainder.find("|>");
            if (end != std::string_view::npos) {
                out.append(kScrub.data(), kScrub.size());
                // Advance past `<|<inner>|>` — mirror the Rust offset math:
                //   i += 2 + end + 2;
                i += 2 + end + 2;
                continue;
            }
        }
        // Emit byte as-is. Rust decodes a UTF-8 `char` here and pushes
        // it; that's equivalent byte-for-byte to copying the bytes
        // because `push(char)` writes the char's UTF-8 encoding — which
        // is the same bytes we already have.
        out.push_back(data[i]);
        ++i;
    }
    return out;
}

bool chat_template_from_str(std::string_view s, ChatTemplate& out) noexcept {
    const std::string lower = ascii_lower_trim(s);
    if (lower == "llama3" || lower == "llama-3" || lower == "llama_3") {
        out = ChatTemplate::Llama3;
        return true;
    }
    if (lower == "short") {
        out = ChatTemplate::Short;
        return true;
    }
    if (lower == "raw") {
        out = ChatTemplate::Raw;
        return true;
    }
    return false;
}

ChatTemplate chat_template_from_env() noexcept {
    const char* v = std::getenv("HALO_CHAT_TEMPLATE");
    if (v == nullptr) return ChatTemplate::Llama3;
    ChatTemplate out;
    if (chat_template_from_str(v, out)) return out;
    // Log-to-stderr approximates the Rust `tracing::warn!`.
    std::fprintf(stderr,
                 "[onebit::cpp] HALO_CHAT_TEMPLATE=%s unrecognized; "
                 "falling back to llama3\n",
                 v);
    return ChatTemplate::Llama3;
}

std::string llama3_render(std::span<const ChatMessage> messages) {
    // Rough upper bound — header + eot per turn plus the tail marker.
    std::size_t cap = kLlama3AsstTail.size();
    for (const auto& m : messages) {
        cap += kLlama3HeaderOpen.size() + m.role.size()
             + kLlama3HeaderClose.size() + m.content.size() + kLlama3EotId.size();
    }
    std::string prompt;
    prompt.reserve(cap);

    for (const auto& m : messages) {
        prompt.append(kLlama3HeaderOpen);
        prompt.append(m.role);
        prompt.append(kLlama3HeaderClose);
        const std::string clean = sanitize(m.content);
        prompt.append(clean);
        prompt.append(kLlama3EotId);
    }
    prompt.append(kLlama3AsstTail);
    return prompt;
}

std::string short_render(std::span<const ChatMessage> messages) {
    std::size_t cap = 0;
    for (const auto& m : messages) cap += m.content.size() + kLlama3EotId.size();

    std::string prompt;
    prompt.reserve(cap);
    for (const auto& m : messages) {
        prompt.append(sanitize(m.content));
        prompt.append(kLlama3EotId);
    }
    return prompt;
}

std::string raw_render(std::span<const ChatMessage> messages) {
    std::size_t cap = 0;
    for (const auto& m : messages) cap += m.content.size();

    std::string prompt;
    prompt.reserve(cap);
    for (const auto& m : messages) prompt.append(sanitize(m.content));
    return prompt;
}

std::string chat_template_render(ChatTemplate tpl,
                                 std::span<const ChatMessage> messages) {
    switch (tpl) {
        case ChatTemplate::Llama3: return llama3_render(messages);
        case ChatTemplate::Short:  return short_render(messages);
        case ChatTemplate::Raw:    return raw_render(messages);
    }
    // Unreachable; enum is exhaustive.
    return {};
}

}  // namespace onebit::cpp
