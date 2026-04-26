// 1bit-halo-ralph CLI front-end. Mirrors the Rust crate's flag surface +
// exit codes. Default base_url / model / system prompt are tracked in
// ralph.hpp for easy auditing against the Rust source.

#include "onebit/halo_ralph/ralph.hpp"

#include <CLI/CLI.hpp>

#include <cstdint>
#include <cstdlib>
#include <string>

using onebit::halo_ralph::Args;
using onebit::halo_ralph::run_loop;
using onebit::halo_ralph::RunStatus;

namespace {

const char* env_or_null(const char* k)
{
    const char* v = std::getenv(k);
    return (v && *v) ? v : nullptr;
}

} // namespace

int main(int argc, char** argv)
{
    CLI::App app{
        "Minimal ralph-loop agent pointed at an OpenAI-compatible endpoint",
        "1bit-halo-ralph"};
    app.set_version_flag("--version", "1bit-halo-ralph 0.1.0");

    Args a;

    // Default-from-env mirrors the Rust crate's `env = "RALPH_*"` directives.
    if (auto* e = env_or_null("RALPH_BASE_URL")) a.base_url = e;
    if (auto* e = env_or_null("RALPH_MODEL"))    a.model    = e;
    if (auto* e = env_or_null("RALPH_API_KEY"))  a.api_key  = std::string(e);

    app.add_option("--task", a.task,
                   "Task prompt for the agent (wrap in quotes)")->required();
    app.add_option("--base-url",   a.base_url,
                   "OpenAI-compatible base URL")->capture_default_str();
    app.add_option("--model",      a.model,
                   "Model id")->capture_default_str();
    std::string api_key_buf;
    auto*       api_key_opt = app.add_option(
        "--api-key", api_key_buf, "Bearer token, if the endpoint requires one");
    app.add_option("--max-iter",   a.max_iter,
                   "Maximum loop iterations")->capture_default_str();
    std::string test_cmd_buf;
    auto*       test_cmd_opt = app.add_option(
        "--test-cmd", test_cmd_buf,
        "Optional shell command to run each iteration");
    std::string system_buf;
    auto*       system_opt = app.add_option(
        "--system", system_buf, "Override the system prompt");
    app.add_option("--temperature", a.temperature,
                   "Sampling temperature")->capture_default_str();

    CLI11_PARSE(app, argc, argv);

    if (api_key_opt->count() > 0) a.api_key  = api_key_buf;
    if (test_cmd_opt->count() > 0) a.test_cmd = test_cmd_buf;
    if (system_opt->count()   > 0) a.system   = system_buf;

    switch (run_loop(a)) {
        case RunStatus::TestsPassed: return 0;
        case RunStatus::NoTestCmd:   return 0;
        case RunStatus::GaveUp:      return 2;
        case RunStatus::HttpError:   return 1;
    }
    return 1;
}
