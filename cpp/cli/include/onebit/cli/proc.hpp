#pragma once

// Tiny subprocess helpers — popen/fork+execvp wrappers. No third-party
// subprocess libraries; the entire CLI's host-touching surface is gated
// through this file so future tightening (cgroup membership, fd hygiene)
// has one place to land.

#include "onebit/cli/error.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::cli {

struct CommandResult {
    int          exit_code = -1;
    bool         signaled  = false;
    std::string  stdout_text;
    std::string  stderr_text;
};

// Spawn argv via fork/execvp, capture stdout + stderr, wait for exit.
// `cwd` defaults to the parent's cwd if empty.
[[nodiscard]] std::expected<CommandResult, Error>
run_capture(const std::vector<std::string>& argv,
            const std::filesystem::path& cwd = {});

// Spawn argv inheriting stdio (so the user sees output live). Returns
// the exit code; signal-terminated children produce -1.
[[nodiscard]] std::expected<int, Error>
run_inherit(const std::vector<std::string>& argv,
            const std::filesystem::path& cwd = {});

// True iff `bin` is on $PATH.
[[nodiscard]] bool which(std::string_view bin);

// Expand a leading `~/` or bare `~` against $HOME.
[[nodiscard]] std::string expand_tilde(std::string_view raw);

}  // namespace onebit::cli
