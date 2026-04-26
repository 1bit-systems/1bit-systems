#include "onebit/cli/burnin.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <map>
#include <sstream>
#include <unordered_map>

namespace onebit::cli {

namespace {
using json = nlohmann::json;

[[nodiscard]] std::string truncate_chars(std::string_view s, std::size_t n)
{
    // Naive char-count truncation — matches Rust's `chars().take(n)`. Works
    // byte-by-byte; UTF-8 handling is "good enough for prompts in JSON".
    if (s.size() <= n) return std::string(s);
    return std::string(s.substr(0, n)) + "…";
}
}  // namespace

std::expected<std::vector<BurninRow>, Error>
load_rows(const std::filesystem::path& path)
{
    std::ifstream f(path);
    if (!f) {
        return std::unexpected(Error::io("cannot open " + path.string()));
    }
    std::vector<BurninRow> rows;
    std::string line;
    while (std::getline(f, line)) {
        // Trim CR/LF + leading/trailing whitespace.
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r' ||
                                  line.back() == ' '  || line.back() == '\t')) {
            line.pop_back();
        }
        if (line.empty()) continue;
        try {
            json j = json::parse(line);
            BurninRow r;
            r.ts                 = j.value("ts", "");
            r.prompt_idx         = j.value("prompt_idx", 0U);
            r.prompt_snippet     = j.value("prompt_snippet", "");
            r.prefix_match_chars = j.value("prefix_match_chars", 0U);
            r.full_match         = j.value("full_match", false);
            r.v1_ms              = j.value("v1_ms", 0ULL);
            r.v2_ms              = j.value("v2_ms", 0ULL);
            r.v1_text            = j.value("v1_text", "");
            r.v2_text            = j.value("v2_text", "");
            rows.push_back(std::move(r));
        } catch (const json::exception&) {
            // Drop bad rows silently (matches Rust behavior — service may
            // be mid-write on the last line).
        }
    }
    return rows;
}

BurninStats compute_stats(const std::vector<BurninRow>& rows)
{
    BurninStats s{};
    s.total = rows.size();
    s.pass  = static_cast<std::size_t>(
        std::count_if(rows.begin(), rows.end(),
                      [](const BurninRow& r) { return r.full_match; }));
    s.fail = s.total - s.pass;
    s.pct  = s.total ? 100.0 * static_cast<double>(s.pass) / static_cast<double>(s.total) : 0.0;
    std::uint64_t v1 = 0, v2 = 0;
    for (const auto& r : rows) { v1 += r.v1_ms; v2 += r.v2_ms; }
    s.mean_v1_ms = s.total ? static_cast<double>(v1) / static_cast<double>(s.total) : 0.0;
    s.mean_v2_ms = s.total ? static_cast<double>(v2) / static_cast<double>(s.total) : 0.0;
    return s;
}

std::vector<DriftBucket>
compute_drift(const std::vector<BurninRow>& rows, std::size_t top)
{
    std::map<std::uint32_t, std::vector<const BurninRow*>> by_idx;
    for (const auto& r : rows) {
        if (!r.full_match) by_idx[r.prompt_idx].push_back(&r);
    }
    std::vector<DriftBucket> out;
    out.reserve(by_idx.size());
    for (auto& [idx, bucket] : by_idx) {
        std::unordered_map<std::uint32_t, std::size_t> hist;
        for (const auto* r : bucket) ++hist[r->prefix_match_chars];
        std::uint32_t typ = 0;
        std::size_t   best = 0;
        for (const auto& [off, n] : hist) {
            if (n > best) { best = n; typ = off; }
        }
        const BurninRow* head = bucket.front();
        out.push_back(DriftBucket{
            idx,
            head->prompt_snippet,
            bucket.size(),
            typ,
            truncate_chars(head->v1_text, 60),
            truncate_chars(head->v2_text, 60),
        });
    }
    std::sort(out.begin(), out.end(), [](const DriftBucket& a, const DriftBucket& b) {
        if (a.fail_count != b.fail_count) return a.fail_count > b.fail_count;
        return a.prompt_idx < b.prompt_idx;
    });
    if (out.size() > top) out.resize(top);
    return out;
}

std::vector<BurninRow>
tail_rows(const std::vector<BurninRow>& rows, std::size_t n)
{
    if (n >= rows.size()) return rows;
    return std::vector<BurninRow>(rows.end() - static_cast<std::ptrdiff_t>(n),
                                   rows.end());
}

std::vector<BurninRow>
filter_since(const std::vector<BurninRow>& rows, std::string_view cutoff)
{
    std::vector<BurninRow> out;
    out.reserve(rows.size());
    for (const auto& r : rows) {
        if (std::string_view(r.ts) >= cutoff) out.push_back(r);
    }
    return out;
}

}  // namespace onebit::cli
