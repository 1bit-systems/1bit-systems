#pragma once

// `1bit burnin` — pure-function analyzer for the shadow-burnin JSONL log.

#include "onebit/cli/error.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::cli {

struct BurninRow {
    std::string   ts;
    std::uint32_t prompt_idx        = 0;
    std::string   prompt_snippet;
    std::uint32_t prefix_match_chars = 0;
    bool          full_match         = false;
    std::uint64_t v1_ms              = 0;
    std::uint64_t v2_ms              = 0;
    std::string   v1_text;
    std::string   v2_text;
};

struct BurninStats {
    std::size_t total = 0;
    std::size_t pass  = 0;
    std::size_t fail  = 0;
    double      pct        = 0.0;
    double      mean_v1_ms = 0.0;
    double      mean_v2_ms = 0.0;
};

struct DriftBucket {
    std::uint32_t prompt_idx     = 0;
    std::string   prompt_snippet;
    std::size_t   fail_count     = 0;
    std::uint32_t typical_offset = 0;
    std::string   sample_v1;
    std::string   sample_v2;
};

[[nodiscard]] std::expected<std::vector<BurninRow>, Error>
load_rows(const std::filesystem::path& path);

[[nodiscard]] BurninStats compute_stats(const std::vector<BurninRow>& rows);
[[nodiscard]] std::vector<DriftBucket>
compute_drift(const std::vector<BurninRow>& rows, std::size_t top);
[[nodiscard]] std::vector<BurninRow>
tail_rows(const std::vector<BurninRow>& rows, std::size_t n);
[[nodiscard]] std::vector<BurninRow>
filter_since(const std::vector<BurninRow>& rows, std::string_view cutoff);

// 95.0% — used by `1bit burnin` (no subcommand) to decide exit code.
inline constexpr double PASS_THRESHOLD_PCT = 95.0;

}  // namespace onebit::cli
