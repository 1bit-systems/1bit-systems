#include "onebit/retrieval/chunker.hpp"

#include <algorithm>
#include <optional>
#include <string>

namespace onebit::retrieval {

namespace {

// Trim ASCII whitespace from both ends of a string view.
[[nodiscard]] std::string_view trim_view(std::string_view s) noexcept
{
    std::size_t a = 0;
    std::size_t b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) {
        ++a;
    }
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) {
        --b;
    }
    return s.substr(a, b - a);
}

// Parse `# foo`, `## foo`, ... at the start of an already-left-trimmed
// line. Returns the heading text (without leading hashes) when present.
[[nodiscard]] std::optional<std::string_view> parse_heading(std::string_view line) noexcept
{
    std::size_t hashes = 0;
    for (char ch : line) {
        if (ch == '#') {
            ++hashes;
            if (hashes > 6) {
                return std::nullopt;
            }
        } else {
            break;
        }
    }
    if (hashes == 0) {
        return std::nullopt;
    }
    auto rest = line.substr(hashes);
    if (rest.empty() || (rest[0] != ' ' && rest[0] != '\t')) {
        return std::nullopt;
    }
    return trim_view(rest);
}

// Char-boundary helpers for valid UTF-8 strings (any byte that is not a
// continuation byte 10xxxxxx is a boundary).
[[nodiscard]] bool is_char_boundary(const std::string& s, std::size_t i) noexcept
{
    if (i == 0 || i >= s.size()) {
        return true;
    }
    return (static_cast<unsigned char>(s[i]) & 0xC0) != 0x80;
}

// Split a buffer that has exceeded the cap. Prefer last `\n` before cap,
// else last space, else cap itself. Result lengths sum to buf.size().
struct SplitParts {
    std::string emit;
    std::string tail;
};

[[nodiscard]] SplitParts split_at_boundary(const std::string& buf, std::size_t cap)
{
    std::size_t safe_cap = std::min(cap, buf.size());
    while (safe_cap > 0 && !is_char_boundary(buf, safe_cap)) {
        --safe_cap;
    }
    // Returns the index of the matched char (Rust-`rfind` semantics), or npos.
    auto rfind_in = [&](char ch) -> std::size_t {
        for (std::size_t i = safe_cap; i > 0; --i) {
            if (buf[i - 1] == ch) {
                return i - 1;
            }
        }
        return std::string::npos;
    };
    // Mirror Rust split_at(idx + 1): newline goes into emit, tail starts after.
    if (auto nl = rfind_in('\n'); nl != std::string::npos && nl > 0) {
        return {buf.substr(0, nl + 1), buf.substr(nl + 1)};
    }
    if (auto sp = rfind_in(' '); sp != std::string::npos && sp > 0) {
        return {buf.substr(0, sp + 1), buf.substr(sp + 1)};
    }
    return {buf.substr(0, safe_cap), buf.substr(safe_cap)};
}

// Walk backwards from `start` until whitespace, return the index just
// after it. Falls back to `start` if no whitespace is found within 64
// bytes.
[[nodiscard]] std::size_t find_word_boundary(const std::string& s, std::size_t start) noexcept
{
    start = std::min(start, s.size());
    while (start > 0 && !is_char_boundary(s, start)) {
        --start;
    }
    if (start == 0 || start == s.size()) {
        return start;
    }
    const std::size_t min = (start > 64) ? (start - 64) : 0;
    std::size_t       i   = start;
    while (i > min) {
        --i;
        while (i > 0 && !is_char_boundary(s, i)) {
            --i;
        }
        const auto b = static_cast<unsigned char>(s[i]);
        if (b == ' ' || b == '\n' || b == '\t') {
            return i + 1;
        }
    }
    return start;
}

void flush(std::string& buf, const std::string& heading, std::vector<Chunk>& out)
{
    auto t = trim_view(buf);
    if (!t.empty()) {
        out.push_back(Chunk{heading, std::string{t}});
    }
    buf.clear();
}

} // namespace

std::vector<Chunk> chunk_markdown(std::string_view src)
{
    std::vector<Chunk> out;
    std::string        cur_heading;
    std::string        buf;

    auto next_line = [](std::string_view s, std::size_t pos)
        -> std::pair<std::string_view, std::size_t> {
        const auto nl = s.find('\n', pos);
        if (nl == std::string_view::npos) {
            return {s.substr(pos), s.size()};
        }
        return {s.substr(pos, nl - pos), nl + 1};
    };

    std::size_t pos = 0;
    while (pos < src.size() || (pos == 0 && !src.empty())) {
        if (pos >= src.size()) {
            break;
        }
        auto [line, next] = next_line(src, pos);
        pos               = next;

        // Left-trim for heading detection; the original line is preserved
        // for the buffer.
        std::size_t lt = 0;
        while (lt < line.size() && (line[lt] == ' ' || line[lt] == '\t')) {
            ++lt;
        }
        const auto stripped = line.substr(lt);

        if (auto h = parse_heading(stripped)) {
            flush(buf, cur_heading, out);
            cur_heading.assign(*h);
            continue;
        }

        buf.append(line);
        buf.push_back('\n');

        if (buf.size() >= SOFT_CAP) {
            auto parts        = split_at_boundary(buf, SOFT_CAP);
            auto emit_trimmed = trim_view(parts.emit);
            if (!emit_trimmed.empty()) {
                out.push_back(Chunk{cur_heading, std::string{emit_trimmed}});
            }
            std::size_t overlap_start = (parts.tail.size() > OVERLAP)
                                          ? parts.tail.size() - OVERLAP
                                          : 0;
            overlap_start             = find_word_boundary(parts.tail, overlap_start);
            buf                       = parts.tail.substr(overlap_start);
        }
    }

    flush(buf, cur_heading, out);
    return out;
}

} // namespace onebit::retrieval
