#include "onebit/agent/tools/registry.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::agent::tools {

namespace {

// Hardcoded path is fine per the spec — when this file moves, swap to
// config-driven via Config::ToolsSection later. The benchmarks dir is
// canonical at the repo root.
constexpr std::string_view kBenchPath =
    "/home/bcloud/repos/1bit-systems/benchmarks/RESULTS-1bit-2026-04-26.md";

[[nodiscard]] nlohmann::json make_schema()
{
    return nlohmann::json{
        {"type", "function"},
        {"function", {
            {"name", "bench_lookup"},
            {"description",
             "Look up a model's benchmark row in the canonical "
             "RESULTS-1bit-*.md table on Strix Halo gfx1151. Returns the "
             "matched table row(s) plus the headline tok/s. Use to back "
             "up perf claims in DM replies with real numbers."},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"model", {
                        {"type", "string"},
                        {"description",
                         "Model name to grep for (case-insensitive). "
                         "Examples: 'lily-bonsai-1.7B', 'trilm', 'gianni'."},
                    }},
                }},
                {"required", nlohmann::json::array({"model"})},
            }},
        }},
    };
}

[[nodiscard]] std::string to_lower(std::string_view s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        out.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

[[nodiscard]] std::string read_file(const std::filesystem::path& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Walk the markdown text. Find the table header row (`| Model | Quant |
// ...`), then collect every subsequent `|` row whose first cell matches
// the query (substring, case-insensitive). Stop on the first blank line
// after the table, since the file has prose afterwards.
[[nodiscard]] std::string scan_table(std::string_view body, std::string_view model_lc)
{
    std::ostringstream out;
    bool in_table = false;
    bool seen_header = false;
    int  rows_found = 0;

    std::size_t pos = 0;
    while (pos < body.size()) {
        const auto nl = body.find('\n', pos);
        const std::string_view line =
            (nl == std::string_view::npos)
                ? body.substr(pos)
                : body.substr(pos, nl - pos);
        pos = (nl == std::string_view::npos) ? body.size() : nl + 1;

        const bool is_table_row = !line.empty() && line.front() == '|';

        if (!in_table) {
            if (is_table_row && line.find("Model") != std::string_view::npos
                && line.find("tok/s") != std::string_view::npos) {
                in_table = true;
                out << line << "\n";
            }
            continue;
        }

        // In-table: separator row (|---|...|) once, then data rows.
        if (is_table_row) {
            if (!seen_header) {
                seen_header = true;
                out << line << "\n";
                continue;
            }
            // First cell — between first and second `|`.
            const auto p1 = line.find('|', 1);
            if (p1 == std::string_view::npos) continue;
            const auto cell = line.substr(1, p1 - 1);
            if (to_lower(cell).find(model_lc) != std::string::npos) {
                out << line << "\n";
                ++rows_found;
            }
            continue;
        }
        // Empty / non-pipe line ends the table.
        if (line.empty() || line.front() != '|') break;
    }

    if (rows_found == 0) {
        return {};
    }
    out << "\n_" << rows_found << " row";
    if (rows_found != 1) out << "s";
    out << " matched._\n";
    return out.str();
}

// Tail of the file: the `## Headline` block. Cheap to grab; gives the
// brain a one-line "fastest decode = ..." in every reply.
[[nodiscard]] std::string extract_headline(std::string_view body)
{
    const auto h = body.find("## Headline");
    if (h == std::string_view::npos) return {};
    const auto next_h = body.find("\n## ", h + 1);
    return std::string(body.substr(
        h, (next_h == std::string_view::npos ? body.size() : next_h) - h));
}

} // namespace

ToolDef make_bench_lookup()
{
    ToolDef d;
    d.name   = "bench_lookup";
    d.schema = make_schema();
    d.invoke = [](const nlohmann::json& args)
        -> std::expected<ToolResult, AgentError>
    {
        const std::string model = args.at("model").get<std::string>();
        if (model.empty()) {
            return ToolResult{false, "bad args: model is empty"};
        }
        const std::string body = read_file(kBenchPath);
        if (body.empty()) {
            return ToolResult{false,
                "bench file unavailable: " + std::string(kBenchPath)};
        }
        const std::string model_lc = to_lower(model);
        const std::string rows     = scan_table(body, model_lc);
        if (rows.empty()) {
            return ToolResult{true,
                "no benchmark row matches `" + model
                + "` in " + std::string(kBenchPath)};
        }
        std::string out = rows;
        out += "\n";
        out += extract_headline(body);
        return ToolResult{true, std::move(out)};
    };
    return d;
}

} // namespace onebit::agent::tools
