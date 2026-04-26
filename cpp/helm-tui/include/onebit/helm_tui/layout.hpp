// 1bit-helm-tui — layout persistence (load/save the pane tree).
//
// Mirrors crates/1bit-helm-tui/src/layout.rs. On any error
// (missing file, parse failure) we fall back to
// `Node::default_layout()` so the TUI always shows something.

#pragma once

#include "onebit/helm_tui/pane.hpp"

#include <expected>
#include <filesystem>
#include <string>

namespace onebit::helm_tui {

// Best-effort load. Falls back to Node::default_layout() on any
// failure. Never throws.
[[nodiscard]] Node load_or_default(const std::filesystem::path& path);

// Persist `node` as pretty-printed JSON. Creates parent dirs.
// Returns the failure reason on error (matches the Rust crate's
// anyhow::Result return).
[[nodiscard]] std::expected<void, std::string>
save(const std::filesystem::path& path, const Node& node);

} // namespace onebit::helm_tui
