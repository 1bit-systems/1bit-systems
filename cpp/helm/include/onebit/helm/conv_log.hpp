// 1bit-helm — JSONL conversation logs under
// `~/.halo/helm/conversations/<unix_ts>.jsonl`.
//
// Mirrors crates/1bit-helm/src/conv_log.rs. Pure file I/O — no Qt,
// no async.

#pragma once

#include "onebit/helm/conversation.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace onebit::helm {

struct LogEntry {
    std::string   role;
    std::string   content;
    std::uint64_t ts{0};
};

// `~/.halo/helm/conversations/`. Falls back to `./` if HOME is unset.
[[nodiscard]] std::filesystem::path default_log_root();

// Write `conv` as a JSONL file named `<unix_ts>.jsonl` under `root`.
// Returns the absolute path written, or the failure reason. Creates
// `root` if missing.
[[nodiscard]] std::expected<std::filesystem::path, std::string>
write_session(const std::filesystem::path& root, const Conversation& conv);

// Parse a JSONL log back. Skips malformed lines.
[[nodiscard]] std::expected<std::vector<LogEntry>, std::string>
read_session(const std::filesystem::path& path);

} // namespace onebit::helm
