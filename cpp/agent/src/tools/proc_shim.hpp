#pragma once

// Tool-side subprocess shim.
//
// The cli already owns a hardened run_capture (cpp/cli/src/proc.cpp). We
// reuse that exact symbol via cpp/cli/include/onebit/cli/proc.hpp; the
// agent library link line declares onebit::cli as a PRIVATE dependency
// so the resolution is unambiguous and ODR-safe. Tests bypass run_capture
// entirely via tools::set_test_run_capture(...).

#include "onebit/agent/error.hpp"
#include "onebit/agent/tools/registry.hpp"
#include "onebit/cli/proc.hpp"

#include <expected>
#include <string>
#include <vector>

namespace onebit::agent::tools {

// Internal: fetch the test stub if one is set (registry.cpp owns it).
RunCaptureFn current_test_run_capture();

// Run argv via the test stub if set, else the real run_capture. Returns
// the StubResult shape the tools want (exit_code + stdout + stderr) or
// an AgentError on subprocess setup failure (fork, pipe, exec missing).
[[nodiscard]] inline std::expected<StubResult, AgentError>
run_or_stub(const std::string& tool_name,
            const std::vector<std::string>& argv)
{
    if (auto fn = current_test_run_capture()) {
        auto r = fn(argv);
        if (!r) {
            return std::unexpected(AgentError::tool(tool_name, r.error()));
        }
        return std::move(*r);
    }
    auto r = onebit::cli::run_capture(argv);
    if (!r) {
        return std::unexpected(AgentError::tool(tool_name, r.error().message));
    }
    return StubResult{
        r->exit_code,
        r->signaled,
        std::move(r->stdout_text),
        std::move(r->stderr_text),
    };
}

} // namespace onebit::agent::tools
