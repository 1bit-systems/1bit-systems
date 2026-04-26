#pragma once

// Stateful sentence-boundary splitter — port of crates/1bit-voice/src/splitter.rs.
//
// Fed one LLM delta at a time, emits complete sentences as they close.
// A "sentence" runs up to and includes one of `.`, `!`, `?`, `\n`. Bytewise
// scanning is safe for UTF-8 because none of the four boundary bytes appear
// inside multi-byte UTF-8 sequences (all ≤ 0x7F, never a continuation byte).
//
// Edge cases (matched bit-exact against the Rust test matrix):
//   * Mid-word delta — buffered until next boundary or finish().
//   * Multiple sentences in one delta — emitted in order, buffer drained.
//   * Lone punctuation (`!` after the previous sentence already closed) —
//     dropped: TTS otherwise synthesizes a click.
//   * Ellipsis (`Hello...`) — emits one speakable sentence (`Hello.`),
//     trailing `..` is punctuation-only and gets dropped on flush.
//   * Trailing partial sentence with no boundary — flushed by finish().
//   * Whitespace between sentences swallowed via trim.
//
// We intentionally accept the abbreviation false positive (`e.g.` →
// `e.`, `g.`) — see docs/wiki/Why-1bit-voice.md.

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::voice {

// Boundary bytes — order doesn't matter, all four equally weighted.
inline constexpr std::array<char, 4> kBoundaryBytes = {'.', '!', '?', '\n'};

// Streaming splitter. Holds a small `std::string` buffer; not thread-safe.
// Wrap in std::mutex if cross-thread use ever lands.
class SentenceSplitter {
public:
    SentenceSplitter() noexcept = default;

    SentenceSplitter(const SentenceSplitter&)            = default;
    SentenceSplitter(SentenceSplitter&&) noexcept        = default;
    SentenceSplitter& operator=(const SentenceSplitter&) = default;
    SentenceSplitter& operator=(SentenceSplitter&&) noexcept = default;
    ~SentenceSplitter()                                  = default;

    // Feed a new delta. Returns 0+ complete sentences in stream order.
    // Partial-sentence remainder stays in the buffer until a boundary
    // or finish() arrives. `delta` may be a non-owning view; we copy
    // only the unflushed remainder, never the whole feed.
    [[nodiscard]] std::vector<std::string> feed(std::string_view delta);

    // Flush any buffered partial sentence. Call once at end-of-stream.
    // Returns std::nullopt when the buffer is empty or whitespace-only.
    [[nodiscard]] std::optional<std::string> finish();

    // Peek the current unflushed buffer (mostly for tests).
    [[nodiscard]] std::string_view buffered() const noexcept { return buf_; }

private:
    std::string buf_{};
};

// Free helper: does the slice contain any alphanumeric glyph? If not, it's
// punctuation/whitespace only — TTS would otherwise click on it. Exposed
// for unit testing; uses the C locale (matches Rust char::is_alphanumeric
// for ASCII; non-ASCII is treated permissively as speakable to mirror
// Rust's UTF-8-aware behavior).
[[nodiscard]] bool has_speakable(std::string_view s) noexcept;

} // namespace onebit::voice
