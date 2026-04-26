// list_files — glob the 1bit-systems tree for paths matching a pattern.
// Backed by ripgrep's --files flag so .gitignore is respected and
// performance scales with the repo size. Caps at 200 hits.

#include "onebit/agent/tools/registry.hpp"
#include "proc_shim.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <sstream>
#include <string>

namespace onebit::agent::tools {

namespace {

constexpr std::string_view kRepoRoot = "/home/bcloud/repos/1bit-systems";
constexpr int kMaxHits   = 200;
constexpr int kDefaultHits = 50;

[[nodiscard]] nlohmann::json make_schema()
{
    return nlohmann::json{
        {"type", "function"},
        {"function", {
            {"name", "list_files"},
            {"description",
             "List files in the 1bit-systems source tree matching a glob "
             "pattern (e.g. \"cpp/agent/src/**.cpp\" or \"strixhalo/systemd/*\"). "
             "Respects .gitignore. Returns up to 50 paths by default, hard "
             "cap 200."},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"glob", {
                        {"type", "string"},
                        {"description", "ripgrep-style glob pattern."},
                    }},
                    {"limit", {
                        {"type", "integer"},
                        {"description", "Max paths to return (default 50, cap 200)."},
                    }},
                }},
                {"required", nlohmann::json::array({"glob"})},
            }},
        }},
    };
}

} // namespace

ToolDef make_list_files()
{
    ToolDef d;
    d.name   = "list_files";
    d.schema = make_schema();
    d.invoke = [](const nlohmann::json& args)
        -> std::expected<ToolResult, AgentError>
    {
        const std::string glob = args.at("glob").get<std::string>();
        if (glob.empty()) {
            return ToolResult{false, "bad args: glob is empty"};
        }
        int limit = kDefaultHits;
        if (args.contains("limit") && args.at("limit").is_number_integer()) {
            limit = args.at("limit").get<int>();
        }
        if (limit <= 0)        limit = kDefaultHits;
        if (limit > kMaxHits)  limit = kMaxHits;

        std::vector<std::string> argv{
            "rg", "--files", "--glob", glob,
            "--", std::string(kRepoRoot),
        };
        auto r = run_or_stub("list_files", argv);
        if (!r) return std::unexpected(std::move(r.error()));
        if (r->exit_code >= 2) {
            return ToolResult{false,
                "rg failed (exit " + std::to_string(r->exit_code) + "): "
                    + r->stderr_text};
        }
        if (r->stdout_text.empty()) {
            return ToolResult{true, "no matches"};
        }

        std::ostringstream out;
        int rows = 0;
        std::size_t pos = 0;
        while (pos < r->stdout_text.size() && rows < limit) {
            const auto nl = r->stdout_text.find('\n', pos);
            const std::string line = (nl == std::string::npos)
                ? r->stdout_text.substr(pos)
                : r->stdout_text.substr(pos, nl - pos);
            std::string p = line;
            const std::string root_pfx = std::string(kRepoRoot) + "/";
            if (p.starts_with(root_pfx)) p.erase(0, root_pfx.size());
            if (!p.empty()) {
                out << "- `" << p << "`\n";
                ++rows;
            }
            if (nl == std::string::npos) break;
            pos = nl + 1;
        }
        if (rows == 0) return ToolResult{true, "no matches"};
        return ToolResult{true, out.str()};
    };
    return d;
}

} // namespace onebit::agent::tools
