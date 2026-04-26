#include "onebit/agent/tools/registry.hpp"
#include "proc_shim.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <sstream>
#include <string>
#include <string_view>

namespace onebit::agent::tools {

namespace {

constexpr std::string_view kRepoRoot = "/home/bcloud/repos/1bit-systems";
constexpr int kDefaultLimit = 20;
constexpr int kMaxLimit     = 100;

[[nodiscard]] nlohmann::json make_schema()
{
    return nlohmann::json{
        {"type", "function"},
        {"function", {
            {"name", "repo_search"},
            {"description",
             "Search the 1bit-systems source tree (under "
             "/home/bcloud/repos/1bit-systems) for a literal string. "
             "Returns top-N hits as a markdown table of {file, line, "
             "snippet}. Use this before quoting code paths in DM replies."},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"query", {
                        {"type", "string"},
                        {"description", "Literal string or regex to search for. Not shell-quoted; passed as a single argv slot."},
                    }},
                    {"limit", {
                        {"type", "integer"},
                        {"description", "Max hits to return (default 20, hard cap 100)."},
                    }},
                }},
                {"required", nlohmann::json::array({"query"})},
            }},
        }},
    };
}

// Parse `rg --json` line-delimited output. Each line is a JSON record.
// We only care about `type == "match"` entries, which carry the file
// path, line number, and the matched line text.
[[nodiscard]] std::string format_rg_json(std::string_view raw, int limit)
{
    std::ostringstream out;
    out << "| file | line | snippet |\n|---|---:|---|\n";

    int rows = 0;
    std::size_t pos = 0;
    while (pos < raw.size() && rows < limit) {
        const auto nl = raw.find('\n', pos);
        const std::string_view line =
            (nl == std::string_view::npos)
                ? raw.substr(pos)
                : raw.substr(pos, nl - pos);
        pos = (nl == std::string_view::npos) ? raw.size() : nl + 1;
        if (line.empty()) continue;

        nlohmann::json j;
        try {
            j = nlohmann::json::parse(line);
        } catch (...) {
            continue;
        }
        if (!j.is_object()) continue;
        auto type_it = j.find("type");
        if (type_it == j.end() || !type_it->is_string()) continue;
        if (type_it->get<std::string>() != "match") continue;

        const auto& data = j.value("data", nlohmann::json::object());
        std::string file =
            data.value("path", nlohmann::json::object())
                .value("text", std::string{});
        const int  line_no = data.value("line_number", 0);
        std::string snippet =
            data.value("lines", nlohmann::json::object())
                .value("text", std::string{});

        // Strip trailing newline + collapse internal pipes so the
        // markdown table doesn't blow out.
        while (!snippet.empty()
               && (snippet.back() == '\n' || snippet.back() == '\r')) {
            snippet.pop_back();
        }
        std::replace(snippet.begin(), snippet.end(), '|', '/');
        // Also bound the snippet length so a giant minified file
        // doesn't spam one row across the entire 4 KB cap.
        if (snippet.size() > 160) {
            snippet = snippet.substr(0, 157) + "...";
        }

        // Strip the repo root prefix to keep file column readable.
        if (file.starts_with(kRepoRoot)) {
            file = file.substr(kRepoRoot.size());
            if (!file.empty() && file.front() == '/') file.erase(0, 1);
        }

        out << "| `" << file << "` | " << line_no << " | `"
            << snippet << "` |\n";
        ++rows;
    }
    if (rows == 0) {
        return "no hits";
    }
    return out.str();
}

} // namespace

ToolDef make_repo_search()
{
    ToolDef d;
    d.name   = "repo_search";
    d.schema = make_schema();
    d.invoke = [](const nlohmann::json& args)
        -> std::expected<ToolResult, AgentError>
    {
        const std::string query = args.at("query").get<std::string>();
        if (query.empty()) {
            return ToolResult{false, "bad args: query is empty"};
        }
        int limit = kDefaultLimit;
        if (args.contains("limit") && args.at("limit").is_number_integer()) {
            limit = args.at("limit").get<int>();
        }
        if (limit <= 0)         limit = kDefaultLimit;
        if (limit > kMaxLimit)  limit = kMaxLimit;

        // Argv form — every byte of `query` lands in a single argv slot.
        // No shell metacharacter risk, no glob, no traversal-by-shell.
        // --max-count caps per-file matches; --json gives us structured
        // output we can parse without scraping.
        std::vector<std::string> argv{
            "rg",
            "--json",
            "--max-count", "20",
            "--", query, std::string(kRepoRoot),
        };
        auto r = run_or_stub("repo_search", argv);
        if (!r) return std::unexpected(std::move(r.error()));

        // rg exit 0=match, 1=no-match, 2=error. Treat 1 as success
        // with empty body; only >=2 is a real failure.
        if (r->exit_code >= 2) {
            return ToolResult{false,
                "rg failed (exit " + std::to_string(r->exit_code) + "): "
                    + r->stderr_text};
        }
        if (r->exit_code == 1 || r->stdout_text.empty()) {
            return ToolResult{true, "no hits"};
        }

        return ToolResult{true, format_rg_json(r->stdout_text, limit)};
    };
    return d;
}

} // namespace onebit::agent::tools
