// read_file — read a tracked file under the 1bit-systems tree, capped at
// 64 KiB. Path is resolved relative to /home/bcloud/repos/1bit-systems
// and a symlink/traversal guard refuses anything that escapes the repo.
//
// The brain uses this to quote source/config/runbook content verbatim
// in its replies. Pairs with repo_search: search → read.

#include "onebit/agent/tools/registry.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace onebit::agent::tools {

namespace {

constexpr std::string_view kRepoRoot = "/home/bcloud/repos/1bit-systems";
constexpr std::size_t      kMaxBytes = 64 * 1024;

[[nodiscard]] nlohmann::json make_schema()
{
    return nlohmann::json{
        {"type", "function"},
        {"function", {
            {"name", "read_file"},
            {"description",
             "Read a file from the 1bit-systems tree (under "
             "/home/bcloud/repos/1bit-systems). Path may be absolute or "
             "relative to the repo root. Returns up to the first 64 KiB. "
             "Refuses paths that resolve outside the repo."},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"path", {
                        {"type", "string"},
                        {"description", "Repo-relative or absolute path inside the 1bit-systems tree."},
                    }},
                }},
                {"required", nlohmann::json::array({"path"})},
            }},
        }},
    };
}

} // namespace

ToolDef make_read_file()
{
    ToolDef d;
    d.name   = "read_file";
    d.schema = make_schema();
    d.invoke = [](const nlohmann::json& args)
        -> std::expected<ToolResult, AgentError>
    {
        const std::string raw_path = args.at("path").get<std::string>();
        if (raw_path.empty()) {
            return ToolResult{false, "bad args: path is empty"};
        }
        std::filesystem::path p(raw_path);
        if (!p.is_absolute()) {
            p = std::filesystem::path(kRepoRoot) / p;
        }
        std::error_code ec;
        const auto canon = std::filesystem::weakly_canonical(p, ec);
        if (ec) {
            return ToolResult{false,
                "read_file: cannot canonicalize: " + ec.message()};
        }
        const std::string canon_s = canon.string();
        if (canon_s.compare(0, kRepoRoot.size(), kRepoRoot) != 0) {
            return ToolResult{false,
                "read_file: refused — path resolves outside repo root"};
        }
        std::ifstream f(canon, std::ios::binary);
        if (!f) {
            return ToolResult{false,
                "read_file: cannot open " + canon_s};
        }
        std::ostringstream buf;
        buf << f.rdbuf();
        std::string body = buf.str();
        const bool truncated = body.size() > kMaxBytes;
        if (truncated) {
            body.resize(kMaxBytes);
            body += "\n\n…(truncated at 64 KiB cap)";
        }
        std::string out;
        out.reserve(body.size() + 96);
        out += "```\n";
        out += "// ";
        out += canon_s;
        out += "\n";
        out += body;
        out += "\n```";
        return ToolResult{true, std::move(out)};
    };
    return d;
}

} // namespace onebit::agent::tools
