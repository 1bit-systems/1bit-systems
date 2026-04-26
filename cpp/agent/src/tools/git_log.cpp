// git_log — recent commit history of the 1bit-systems repo. Pure read.
// Bounded to last 30 commits, oneline format. Optional file-path filter.

#include "onebit/agent/tools/registry.hpp"
#include "proc_shim.hpp"

#include <nlohmann/json.hpp>

#include <string>

namespace onebit::agent::tools {

namespace {

constexpr std::string_view kRepoRoot = "/home/bcloud/repos/1bit-systems";
constexpr int kDefaultN = 15;
constexpr int kMaxN     = 30;

[[nodiscard]] nlohmann::json make_schema()
{
    return nlohmann::json{
        {"type", "function"},
        {"function", {
            {"name", "git_log"},
            {"description",
             "Recent commits in the 1bit-systems repo, oneline format. "
             "Defaults to last 15 commits across the whole repo. Pass "
             "`path` to restrict to commits that touched that file/dir."},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"n", {
                        {"type", "integer"},
                        {"description", "Number of commits (default 15, cap 30)."},
                    }},
                    {"path", {
                        {"type", "string"},
                        {"description", "Optional path filter (relative to repo root)."},
                    }},
                }},
                {"required", nlohmann::json::array()},
            }},
        }},
    };
}

} // namespace

ToolDef make_git_log()
{
    ToolDef d;
    d.name   = "git_log";
    d.schema = make_schema();
    d.invoke = [](const nlohmann::json& args)
        -> std::expected<ToolResult, AgentError>
    {
        int n = kDefaultN;
        if (args.contains("n") && args.at("n").is_number_integer()) {
            n = args.at("n").get<int>();
        }
        if (n <= 0)     n = kDefaultN;
        if (n > kMaxN)  n = kMaxN;

        std::vector<std::string> argv{
            "git", "-C", std::string(kRepoRoot),
            "log", "--oneline", "--no-decorate",
            "-n", std::to_string(n),
        };
        if (args.contains("path") && args.at("path").is_string()) {
            const std::string path = args.at("path").get<std::string>();
            if (!path.empty()) {
                argv.emplace_back("--");
                argv.emplace_back(path);
            }
        }
        auto r = run_or_stub("git_log", argv);
        if (!r) return std::unexpected(std::move(r.error()));
        if (r->exit_code != 0) {
            return ToolResult{false,
                "git log failed (exit " + std::to_string(r->exit_code) + "): "
                    + r->stderr_text};
        }
        if (r->stdout_text.empty()) {
            return ToolResult{true, "no commits"};
        }
        std::string out = "```\n";
        out += r->stdout_text;
        if (!out.empty() && out.back() != '\n') out += '\n';
        out += "```";
        return ToolResult{true, std::move(out)};
    };
    return d;
}

} // namespace onebit::agent::tools
