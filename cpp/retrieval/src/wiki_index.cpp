#include "onebit/retrieval/wiki_index.hpp"

#include "onebit/retrieval/chunker.hpp"
#include "onebit/retrieval/stopwords.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace onebit::retrieval {

namespace {

[[nodiscard]] std::string lc(std::string_view s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        out.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

[[nodiscard]] std::string trim_dashunder(std::string_view s)
{
    std::size_t a = 0;
    std::size_t b = s.size();
    while (a < b && (s[a] == '-' || s[a] == '_')) {
        ++a;
    }
    while (b > a && (s[b - 1] == '-' || s[b - 1] == '_')) {
        --b;
    }
    return std::string{s.substr(a, b - a)};
}

void flush_token(std::string& cur, std::vector<std::string>& out)
{
    if (cur.empty()) {
        return;
    }
    auto trimmed = trim_dashunder(cur);
    if (!trimmed.empty() && !is_stopword(trimmed)) {
        out.push_back(std::move(trimmed));
    }
    cur.clear();
}

} // namespace

std::vector<std::string> tokenize(std::string_view s)
{
    std::vector<std::string> out;
    std::string              cur;
    for (char ch : s) {
        const auto uch = static_cast<unsigned char>(ch);
        const bool keep =
            (std::isalnum(uch) != 0) || ch == '-' || ch == '_';
        if (keep) {
            cur.push_back(static_cast<char>(std::tolower(uch)));
        } else {
            flush_token(cur, out);
        }
    }
    flush_token(cur, out);
    return out;
}

std::string slugify(std::string_view s)
{
    std::string out;
    bool        last_dash = true;
    for (char ch : s) {
        const auto uch = static_cast<unsigned char>(ch);
        if (std::isalnum(uch) != 0) {
            out.push_back(ch);
            last_dash = false;
        } else if (!last_dash) {
            out.push_back('-');
            last_dash = true;
        }
    }
    while (!out.empty() && out.back() == '-') {
        out.pop_back();
    }
    return out;
}

std::string RetrievalError::what() const
{
    return std::visit(
        [](const auto& e) -> std::string {
            using T = std::decay_t<decltype(e)>;
            if constexpr (std::is_same_v<T, ErrorWikiDirMissing>) {
                return "wiki directory not found: " + e.path.string();
            } else if constexpr (std::is_same_v<T, ErrorIo>) {
                return "io error reading " + e.path.string() + ": " + e.message;
            } else if constexpr (std::is_same_v<T, ErrorNoMarkdown>) {
                return "wiki directory loaded zero markdown files: " + e.path.string();
            }
        },
        v_);
}

// ----------------------------------------------------------------------
// Impl

struct WikiIndex::Impl {
    struct StoredChunk {
        std::string   file;
        std::string   heading;
        std::string   text;
        std::uint32_t len_tokens{0};
    };
    struct Posting {
        std::uint32_t                                          doc_freq{0};
        std::vector<std::pair<std::uint32_t, std::uint32_t>>   entries; // (chunk_idx, tf)
    };

    std::vector<StoredChunk>                  chunks;
    std::unordered_map<std::string, Posting>  postings;
    float                                     avgdl{1.0F};
    float                                     n_chunks{0.0F};
    float                                     k1{1.5F};
    float                                     b{0.75F};
};

WikiIndex::WikiIndex() : impl_{std::make_unique<Impl>()} {}
WikiIndex::WikiIndex(WikiIndex&&) noexcept            = default;
WikiIndex& WikiIndex::operator=(WikiIndex&&) noexcept = default;
WikiIndex::~WikiIndex()                               = default;

std::size_t WikiIndex::len() const noexcept
{
    return impl_ ? impl_->chunks.size() : 0;
}

bool WikiIndex::is_empty() const noexcept
{
    return len() == 0;
}

std::expected<WikiIndex, RetrievalError>
WikiIndex::load(const std::filesystem::path& wiki_dir)
{
    namespace fs = std::filesystem;

    std::error_code ec;
    if (!fs::exists(wiki_dir, ec) || ec) {
        return std::unexpected(RetrievalError::wiki_dir_missing(wiki_dir));
    }

    WikiIndex idx;
    auto&     chunks = idx.impl_->chunks;

    // Recursive walk; deterministic order via sort. We use the throwing
    // iterator overload + `increment(ec)` so a permission error in a
    // subdir doesn't surface as an exception.
    std::vector<fs::path> md_files;
    auto                   it  = fs::recursive_directory_iterator(
        wiki_dir, fs::directory_options::skip_permission_denied, ec);
    if (ec) {
        return std::unexpected(RetrievalError::io(wiki_dir, ec.message()));
    }
    const auto end = fs::recursive_directory_iterator{};
    while (it != end) {
        std::error_code step_ec;
        if (it->is_regular_file(step_ec)) {
            const auto& p = it->path();
            if (p.extension() == ".md") {
                md_files.push_back(p);
            }
        }
        it.increment(step_ec);
        if (step_ec) {
            break;
        }
    }
    std::sort(md_files.begin(), md_files.end());

    for (const auto& path : md_files) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return std::unexpected(
                RetrievalError::io(path, "open failed"));
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        std::string body = std::move(ss).str();

        const auto rel_path = fs::relative(path, wiki_dir, ec);
        std::string rel = (ec ? path.string() : rel_path.generic_string());

        for (auto& ch : chunk_markdown(body)) {
            const auto toks       = tokenize(ch.text);
            const auto len_tokens = static_cast<std::uint32_t>(toks.size());
            if (len_tokens == 0) {
                continue;
            }
            chunks.push_back(Impl::StoredChunk{
                rel, std::move(ch.heading), std::move(ch.text), len_tokens});
        }
    }

    if (chunks.empty()) {
        return std::unexpected(RetrievalError::no_markdown(wiki_dir));
    }

    // Build postings.
    auto& postings = idx.impl_->postings;
    for (std::uint32_t i = 0; i < chunks.size(); ++i) {
        std::unordered_map<std::string, std::uint32_t> tf;
        for (auto& tok : tokenize(chunks[i].text)) {
            ++tf[std::move(tok)];
        }
        for (auto& [term, freq] : tf) {
            auto& p = postings[term];
            p.doc_freq += 1;
            p.entries.emplace_back(i, freq);
        }
    }

    std::uint64_t total_tokens = 0;
    for (const auto& c : chunks) {
        total_tokens += c.len_tokens;
    }
    const float n_chunks = static_cast<float>(chunks.size());
    const float avgdl    = (n_chunks > 0.0F)
                               ? (static_cast<float>(total_tokens) / n_chunks)
                               : 1.0F;
    idx.impl_->n_chunks = n_chunks;
    idx.impl_->avgdl    = std::max(avgdl, 1.0F);

    return idx;
}

std::vector<RetrievedChunk>
WikiIndex::top_k(std::string_view query, std::size_t k) const
{
    if (k == 0 || !impl_) {
        return {};
    }
    auto q_terms_raw = tokenize(query);
    if (q_terms_raw.empty()) {
        return {};
    }

    // Dedupe the query so "rust rust rust" doesn't triple-count.
    std::unordered_set<std::string> seen;
    std::vector<std::string>        q_terms;
    q_terms.reserve(q_terms_raw.size());
    for (auto& t : q_terms_raw) {
        if (seen.insert(t).second) {
            q_terms.push_back(std::move(t));
        }
    }

    std::unordered_map<std::uint32_t, float> scores;
    for (const auto& term : q_terms) {
        auto it = impl_->postings.find(term);
        if (it == impl_->postings.end()) {
            continue;
        }
        const auto& posting = it->second;
        const float df      = static_cast<float>(posting.doc_freq);
        const float idf =
            std::log(((impl_->n_chunks - df + 0.5F) / (df + 0.5F)) + 1.0F);
        for (const auto& [chunk_idx, tf_u32] : posting.entries) {
            const float dl   = static_cast<float>(impl_->chunks[chunk_idx].len_tokens);
            const float tf_f = static_cast<float>(tf_u32);
            const float norm = 1.0F - impl_->b + impl_->b * (dl / impl_->avgdl);
            const float bm   = idf * ((tf_f * (impl_->k1 + 1.0F))
                                    / (tf_f + impl_->k1 * norm));
            scores[chunk_idx] += bm;
        }
    }

    if (scores.empty()) {
        return {};
    }

    std::vector<std::pair<std::uint32_t, float>> ranked(scores.begin(), scores.end());
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) {
                  if (a.second != b.second) {
                      return a.second > b.second;
                  }
                  return a.first < b.first;
              });
    if (ranked.size() > k) {
        ranked.resize(k);
    }

    std::vector<RetrievedChunk> out;
    out.reserve(ranked.size());
    for (const auto& [i, score] : ranked) {
        const auto& c = impl_->chunks[i];
        out.push_back(RetrievedChunk{c.file, c.heading, c.text, score});
    }
    return out;
}

std::string format_for_system_prompt(const std::vector<RetrievedChunk>& chunks)
{
    if (chunks.empty()) {
        return {};
    }
    std::string out;
    out.append("RELEVANT DOCS (top ");
    out.append(std::to_string(chunks.size()));
    out.append("):\n\n");
    for (const auto& c : chunks) {
        const auto anchor = slugify(c.heading);
        // Trim ASCII whitespace for the body view.
        auto       body   = std::string_view{c.text};
        std::size_t a = 0;
        std::size_t b = body.size();
        while (a < b && (body[a] == ' ' || body[a] == '\t' || body[a] == '\r' || body[a] == '\n')) {
            ++a;
        }
        while (b > a && (body[b - 1] == ' ' || body[b - 1] == '\t' || body[b - 1] == '\r' || body[b - 1] == '\n')) {
            --b;
        }
        body = body.substr(a, b - a);
        out.push_back('[');
        out.append(c.file);
        if (!anchor.empty()) {
            out.push_back('#');
            out.append(anchor);
        }
        out.append("] ");
        out.append(body);
        out.push_back('\n');
    }
    return out;
}

} // namespace onebit::retrieval
