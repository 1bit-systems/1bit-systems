#pragma once

// HTTP poll helpers for GitHub + HuggingFace, plus a hook runner that
// shells out to per-entry on_merge / on_bump argv arrays.

#include "onebit/watchdog/config.hpp"
#include "onebit/watchdog/state.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace onebit::watchdog {

enum class PollError {
    Network,
    HttpStatus,
    BadJson,
    MissingSha,
};

// Fetch the current default-branch SHA for a "owner/repo" GitHub slug.
// Honors the GH_TOKEN env var when present.
[[nodiscard]] std::optional<std::string>
poll_github(std::string_view repo,
            std::string_view branch_or_empty,
            PollError*       err = nullptr);

// Fetch the default-revision SHA for a HuggingFace Hub model.
[[nodiscard]] std::optional<std::string>
poll_huggingface(std::string_view repo, PollError* err = nullptr);

// Run on_merge (Github) or on_bump (Huggingface) hooks. With dry_run set,
// hooks are logged but not executed. Returns false on the first non-zero
// exit; remaining hooks are skipped (matching Rust's anyhow::bail!).
[[nodiscard]] bool run_hooks(const WatchEntry& entry, bool dry_run);

// One poll cycle for a single entry. Records the resulting transition in
// `state` and triggers hooks on SoakComplete (unless dry_run).
[[nodiscard]] bool poll_entry(const WatchEntry& entry,
                              State&            state,
                              bool              dry_run);

} // namespace onebit::watchdog
