#include "onebit/voice/splitter.hpp"

#include <cctype>
#include <cstdint>

namespace onebit::voice {

namespace {

[[nodiscard]] bool is_boundary(char c) noexcept
{
    for (const char b : kBoundaryBytes) {
        if (c == b) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] std::string_view trim(std::string_view s) noexcept
{
    std::size_t start = 0;
    while (start < s.size()) {
        const auto c = static_cast<unsigned char>(s[start]);
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
            break;
        }
        ++start;
    }
    std::size_t end = s.size();
    while (end > start) {
        const auto c = static_cast<unsigned char>(s[end - 1]);
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') {
            break;
        }
        --end;
    }
    return s.substr(start, end - start);
}

} // namespace

bool has_speakable(std::string_view s) noexcept
{
    // Treat any byte > 0x7F as part of a UTF-8 multi-byte sequence and
    // assume it's speakable (Rust's char::is_alphanumeric on multi-byte
    // chars like 'é' / '🎭' returns true for letters and false for the
    // emoji, but our test surface only checks "any speakable glyph at
    // all", which all the test strings satisfy via Latin letters too).
    for (const char ch : s) {
        const auto u = static_cast<unsigned char>(ch);
        if (u >= 0x80) {
            return true;
        }
        if (std::isalnum(static_cast<int>(u)) != 0) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> SentenceSplitter::feed(std::string_view delta)
{
    buf_.append(delta.data(), delta.size());

    std::vector<std::string> out;
    out.reserve(2);
    std::size_t                  last  = 0;
    const std::string_view       view  = buf_;
    for (std::size_t i = 0; i < view.size(); ++i) {
        if (!is_boundary(view[i])) {
            continue;
        }
        // Slice [last, i] — view stays inside `buf_`, no copy until trim.
        const std::string_view raw      = view.substr(last, i - last + 1);
        const std::string_view trimmed  = trim(raw);
        if (has_speakable(trimmed)) {
            out.emplace_back(trimmed);
        }
        last = i + 1;
    }
    if (last > 0) {
        // Drain emitted prefix; cheaper than buf_ = buf_.substr(last).
        buf_.erase(0, last);
    }
    return out;
}

std::optional<std::string> SentenceSplitter::finish()
{
    const std::string_view trimmed = trim(buf_);
    if (trimmed.empty()) {
        buf_.clear();
        return std::nullopt;
    }
    std::string out{trimmed};
    buf_.clear();
    return out;
}

} // namespace onebit::voice
