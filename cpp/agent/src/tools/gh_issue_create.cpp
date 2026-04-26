#include "onebit/agent/tools/registry.hpp"
#include "proc_shim.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::agent::tools {

namespace {

// Hard cap on user-provided strings before they hit `gh`. Titles >256
// chars are GitHub-side rejected anyway; bodies above ~32 KB are nearly
// always brain hallucination spam.
constexpr std::size_t kMaxTitleBytes = 256;
constexpr std::size_t kMaxBodyBytes  = 32 * 1024;

[[nodiscard]] nlohmann::json make_schema()
{
    return nlohmann::json{
        {"type", "function"},
        {"function", {
            {"name", "gh_issue_create"},
            {"description",
             "Open a GitHub issue on the active repo via `gh issue "
             "create`. REFUSES TO FIRE without `confirm: true`. "
             "Autonomous agents must ask the human to confirm first, "
             "or the operator must set `gh_issue_auto_confirm` in the "
             "registry build options. The repo `gh` targets is whichever "
             "default it resolves from cwd (don't call this from outside "
             "a 1bit-systems checkout)."},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"title", {
                        {"type", "string"},
                        {"description", "Issue title. Bounded to 256 bytes."},
                    }},
                    {"body", {
                        {"type", "string"},
                        {"description", "Issue body markdown. Bounded to 32 KB."},
                    }},
                    {"labels", {
                        {"type", "array"},
                        {"description",
                         "Optional labels. Each must already exist on the "
                         "repo or `gh` will refuse to create the issue."},
                    }},
                    {"confirm", {
                        {"type", "boolean"},
                        {"description",
                         "MUST be true. The agent should ask the human "
                         "first; without confirm:true this tool refuses "
                         "and returns success=false content='not "
                         "confirmed'."},
                    }},
                }},
                {"required", nlohmann::json::array({"title", "body", "confirm"})},
            }},
        }},
    };
}

[[nodiscard]] std::string clip(std::string_view s, std::size_t cap)
{
    if (s.size() <= cap) return std::string(s);
    return std::string(s.substr(0, cap));
}

} // namespace

ToolDef make_gh_issue_create()
{
    ToolDef d;
    d.name   = "gh_issue_create";
    d.schema = make_schema();
    d.invoke = [](const nlohmann::json& args)
        -> std::expected<ToolResult, AgentError>
    {
        // Confirm-gate first. Refusing here is a *successful* tool call
        // (the brain sees the message and can re-prompt with confirm).
        const bool confirm = args.value("confirm", false);
        if (!confirm) {
            return ToolResult{false,
                "not confirmed: gh_issue_create requires confirm:true. "
                "Ask the human in plain text first; only then re-call "
                "this tool with confirm:true."};
        }

        const std::string title = clip(args.at("title").get<std::string>(),
                                       kMaxTitleBytes);
        const std::string body  = clip(args.at("body").get<std::string>(),
                                       kMaxBodyBytes);
        if (title.empty()) {
            return ToolResult{false, "bad args: title is empty"};
        }
        if (body.empty()) {
            return ToolResult{false, "bad args: body is empty"};
        }

        // Argv form — every label and every `--body` byte rides as a
        // separate slot. If the brain emits backticks / $(...) / ;
        // they're inert because no shell ever sees them.
        std::vector<std::string> argv{
            "gh", "issue", "create",
            "--title", title,
            "--body",  body,
        };
        if (auto it = args.find("labels"); it != args.end() && it->is_array()) {
            for (const auto& l : *it) {
                if (!l.is_string()) continue;
                const std::string lab = l.get<std::string>();
                if (lab.empty()) continue;
                argv.push_back("--label");
                argv.push_back(lab);
            }
        }

        auto r = run_or_stub("gh_issue_create", argv);
        if (!r) return std::unexpected(std::move(r.error()));
        if (r->exit_code != 0) {
            std::ostringstream m;
            m << "gh issue create failed (exit " << r->exit_code << ")";
            if (!r->stderr_text.empty()) {
                m << ":\n" << r->stderr_text;
            }
            return ToolResult{false, m.str()};
        }
        // gh prints the issue URL on stdout when --json isn't passed.
        std::string url = r->stdout_text;
        while (!url.empty()
               && (url.back() == '\n' || url.back() == '\r')) {
            url.pop_back();
        }
        std::ostringstream out;
        out << "issue created: " << (url.empty() ? "(no URL on stdout)" : url);
        return ToolResult{true, out.str()};
    };
    return d;
}

} // namespace onebit::agent::tools
