#pragma once

// 1bit-agent — concrete ToolRegistry implementation.
//
// The autonomous-agent loop (sibling-authored) holds an `IToolRegistry*`
// and asks it two questions per turn:
//
//   1. `list_tools_openai_format()` — returns the JSON schema array shipped
//      to the brain in `BrainRequest.tools`. Each entry is a complete
//      OpenAI function-calling definition (`{"type":"function","function":
//      {"name":..., "description":..., "parameters":{...}}}`).
//   2. `call(ToolCall)` — dispatches by `name`, validates args against
//      the tool's schema, runs with a wall-clock deadline, returns
//      `ToolResult{success, content}`. Output content is *always* trimmed
//      and capped (4 KB default) — never raw subprocess bytes.
//
// Tools are registered by name from `[tools] enabled = [...]` in the
// agent's TOML config. Unknown names are skipped with a warning so a stale
// config doesn't brick the daemon.
//
// Concurrency: registry is read-only after `build(...)`. `call(...)` is
// reentrant (each invocation owns its argv + buffers); the underlying
// subprocess helpers in cpp/cli/src/proc.cpp do not share state.

#include "onebit/agent/error.hpp"
#include "onebit/agent/event.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstddef>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::agent {

// Sibling will land an identical interface in loop.hpp; until then we own
// the abstract shape so the registry compiles + tests in isolation. When
// sibling's loop.hpp arrives, that header forward-declares the same type
// in this same namespace and the ODR collapses.
class IToolRegistry {
public:
    virtual ~IToolRegistry() = default;

    [[nodiscard]] virtual std::vector<nlohmann::json>
    list_tools_openai_format() const = 0;

    [[nodiscard]] virtual std::expected<ToolResult, AgentError>
    call(const ToolCall& c) = 0;
};

namespace tools {

// One tool-registry entry. Authored once per concrete tool below
// (repo_search.cpp, bench_lookup.cpp, ...). The `schema` blob is the
// full OpenAI function-calling object so we can ship it verbatim to the
// brain. `invoke` is a synchronous functor; the registry wraps every
// call in a chrono deadline before handing the result back.
struct ToolDef {
    std::string    name;
    nlohmann::json schema;        // OpenAI function-calling JSON
    std::function<std::expected<ToolResult, AgentError>(const nlohmann::json& args)>
                   invoke;
};

// Default deadline if a tool doesn't override. 5s is the cap for any one
// call; anything longer is a config bug (slow rg root, network gh, ...).
inline constexpr std::chrono::milliseconds kDefaultDeadline{5'000};
// Hard cap on tool output content, post-trim. 4 KB is the spec; brain
// will receive role=tool messages bounded to this length.
inline constexpr std::size_t kMaxContentBytes = 4 * 1024;

// ---- Tool factories ------------------------------------------------------
//
// Each factory returns the ToolDef the registry registers under `name`.
// Factories own zero state — the lambdas they return capture by value.
// Test stubs swap out the proc shim via `set_test_run_capture(...)`.

[[nodiscard]] ToolDef make_repo_search();
[[nodiscard]] ToolDef make_bench_lookup();
[[nodiscard]] ToolDef make_install_runbook();
[[nodiscard]] ToolDef make_gh_issue_create();

// ---- Test seam -----------------------------------------------------------
//
// Production path shells out via cpp/cli/src/proc.cpp::run_capture. Tests
// inject a fake by setting the function pointer below; nullptr means
// "use the real subprocess path". We keep this in the header so test
// translation units can flip it without touching the registry impl.
struct StubResult {
    int          exit_code = 0;
    bool         signaled  = false;
    std::string  stdout_text;
    std::string  stderr_text;
};
using RunCaptureFn = std::function<
    std::expected<StubResult, std::string>(const std::vector<std::string>& argv)>;

void set_test_run_capture(RunCaptureFn fn);   // pass {} to clear

// ---- Helpers (exposed for tests) ----------------------------------------

// Validates `args` against a JSON schema's required/properties/types.
// Lightweight — supports object/string/integer/number/boolean/array(of
// string). Returns empty optional on success; otherwise the message that
// the registry stamps into ToolResult{success=false, content=...}.
[[nodiscard]] std::optional<std::string>
validate_args(const nlohmann::json& schema, const nlohmann::json& args);

// Trim trailing whitespace + cap to kMaxContentBytes. Drops a trailing
// `... [truncated N bytes]` notice when capped so the brain sees it.
[[nodiscard]] std::string trim_and_cap(std::string_view raw);

} // namespace tools

// ---- ToolRegistry --------------------------------------------------------

class ToolRegistry final : public IToolRegistry {
public:
    ToolRegistry();
    ~ToolRegistry() override;

    ToolRegistry(const ToolRegistry&)            = delete;
    ToolRegistry& operator=(const ToolRegistry&) = delete;
    ToolRegistry(ToolRegistry&&) noexcept;
    ToolRegistry& operator=(ToolRegistry&&) noexcept;

    // Build a registry from a list of enabled tool names. Unknown names
    // are recorded in `warnings` (not fatal) so a stale config still
    // boots. `gh_issue_auto_confirm` flips the gh_issue_create tool's
    // confirm-gate default — set true only when the operator explicitly
    // opts into "let the agent file issues without asking".
    struct BuildOptions {
        bool gh_issue_auto_confirm = false;
    };
    struct BuildOutcome {
        std::vector<std::string> warnings;
    };

    [[nodiscard]] BuildOutcome
    build(const std::vector<std::string>& enabled);

    [[nodiscard]] BuildOutcome
    build(const std::vector<std::string>& enabled, const BuildOptions& opts);

    // For tests / hot-reload paths.
    void register_tool(tools::ToolDef def);

    // IToolRegistry --------------------------------------------------------

    [[nodiscard]] std::vector<nlohmann::json>
    list_tools_openai_format() const override;

    [[nodiscard]] std::expected<ToolResult, AgentError>
    call(const ToolCall& c) override;

    // Test-only — peek at how many tools are wired.
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool        has(std::string_view name) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;

    void ensure_impl();
};

} // namespace onebit::agent
