#pragma once

// In-memory BM25 index over a wiki directory. Constructed once via
// WikiIndex::load and queried repeatedly via WikiIndex::top_k.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace onebit::retrieval {

struct RetrievedChunk {
    std::string file;
    std::string heading;
    std::string text;
    float       score{0.0F};
};

// Error variants. Mirrors the Rust `RetrievalError` shape.
struct ErrorWikiDirMissing {
    std::filesystem::path path;
};
struct ErrorIo {
    std::filesystem::path path;
    std::string           message;
};
struct ErrorNoMarkdown {
    std::filesystem::path path;
};

class RetrievalError {
public:
    using Variant = std::variant<ErrorWikiDirMissing, ErrorIo, ErrorNoMarkdown>;

    explicit RetrievalError(Variant v) : v_{std::move(v)} {}

    [[nodiscard]] const Variant& variant() const noexcept { return v_; }
    [[nodiscard]] std::string    what() const;

    static RetrievalError wiki_dir_missing(std::filesystem::path p)
    {
        return RetrievalError{ErrorWikiDirMissing{std::move(p)}};
    }
    static RetrievalError io(std::filesystem::path p, std::string msg)
    {
        return RetrievalError{ErrorIo{std::move(p), std::move(msg)}};
    }
    static RetrievalError no_markdown(std::filesystem::path p)
    {
        return RetrievalError{ErrorNoMarkdown{std::move(p)}};
    }

private:
    Variant v_;
};

// Lower-case, split on non-alnum, drop stopwords. Retains '-' and '_'
// inside tokens (so `fish-shell` and `gfx1151` survive). Public for tests.
[[nodiscard]] std::vector<std::string> tokenize(std::string_view s);

// Slugify a heading to GitHub-style anchor (`Distro policy` →
// `Distro-policy`). Public for tests.
[[nodiscard]] std::string slugify(std::string_view s);

class WikiIndex {
public:
    // Opaque pImpl — special members declared here, defined in .cpp per
    // ISO C++ Core Guideline I.27.
    WikiIndex();
    WikiIndex(const WikiIndex&)            = delete;
    WikiIndex& operator=(const WikiIndex&) = delete;
    WikiIndex(WikiIndex&&) noexcept;
    WikiIndex& operator=(WikiIndex&&) noexcept;
    ~WikiIndex();

    [[nodiscard]] static std::expected<WikiIndex, RetrievalError>
        load(const std::filesystem::path& wiki_dir);

    [[nodiscard]] std::vector<RetrievedChunk> top_k(std::string_view query, std::size_t k) const;

    [[nodiscard]] std::size_t len() const noexcept;
    [[nodiscard]] bool        is_empty() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Format retrieved chunks for injection into a system prompt.
[[nodiscard]] std::string format_for_system_prompt(const std::vector<RetrievedChunk>& chunks);

} // namespace onebit::retrieval
