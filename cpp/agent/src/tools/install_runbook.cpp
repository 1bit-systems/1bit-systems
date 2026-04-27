#include "onebit/agent/tools/registry.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>

#ifndef ONEBIT_AGENT_RUNBOOK_DIR
#define ONEBIT_AGENT_RUNBOOK_DIR "configs/runbooks"
#endif

namespace onebit::agent::tools {

namespace {

[[nodiscard]] std::string home_subdir(const char* suffix)
{
    const char* home = std::getenv("HOME");
    if (!home || !*home) return {};
    return std::string(home) + suffix;
}

// Runbooks ship in-tree at cpp/agent/configs/runbooks/, baked at build
// time via ONEBIT_AGENT_RUNBOOK_DIR. The CLI's install path drops them
// into ${HOME}/.local/share/1bit-agent/runbooks/ — we look at both,
// repo-tree first because dev iteration loads from there.
[[nodiscard]] std::string repo_runbooks_dir()
{
    return ONEBIT_AGENT_RUNBOOK_DIR;
}

[[nodiscard]] std::string installed_runbooks_dir()
{
    return home_subdir("/.local/share/1bit-agent/runbooks");
}

// Optional buglog the brain has seen us write to. If present, we grep
// for similar errors and append the matching fix block.
[[nodiscard]] std::string buglog_path()
{
    return home_subdir("/.claude/projects/-home-bcloud/memory/buglog.json");
}

// Whitelist — components must match a fixed set so the agent can't be
// tricked into reading arbitrary files via path traversal in `component`.
const std::unordered_set<std::string>&
allowed_components()
{
    static const std::unordered_set<std::string> set{
        "core", "voice", "echo", "mcp", "helm", "landing", "power",
        "tunnel", "whisper-kokoro", "sd", "lemonade", "burnin", "npu",
        "stt-engine", "tts-engine", "image-engine", "ingest", "agents",
    };
    return set;
}

[[nodiscard]] nlohmann::json make_schema()
{
    return nlohmann::json{
        {"type", "function"},
        {"function", {
            {"name", "install_runbook"},
            {"description",
             "Fetch a canned install runbook for one of the 1bit-systems "
             "components. Returns prereqs, common errors + fixes, "
             "rollback procedure, and log locations. If `error_text` is "
             "provided, also greps the local buglog.json for similar "
             "past failures and appends the matched fix."},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"component", {
                        {"type", "string"},
                        {"description",
                         "Component name from packages.toml. One of: "
                         "core, voice, echo, mcp, helm, landing, power, "
                         "tunnel, whisper-kokoro, sd, lemonade, burnin, "
                         "npu, stt-engine, tts-engine, image-engine, "
                         "ingest, agents."},
                    }},
                    {"error_text", {
                        {"type", "string"},
                        {"description",
                         "Optional. The error message the user pasted. "
                         "If present, used to grep buglog.json for past "
                         "matching incidents."},
                    }},
                }},
                {"required", nlohmann::json::array({"component"})},
            }},
        }},
    };
}

[[nodiscard]] std::string read_file(const std::filesystem::path& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Extract a few "anchor" tokens from an error string — alpha-numeric
// runs of length >= 5 are kept, lowercased, deduped. Skips noisy words
// (the, with, error, ...) so the buglog scan focuses on signal.
[[nodiscard]] std::vector<std::string> tokenize_error(std::string_view err)
{
    static const std::unordered_set<std::string> stop{
        "error", "errors", "with", "this", "that", "have",
        "would", "could", "should", "found", "while",
        "warning", "failed", "failure",
    };
    std::vector<std::string> tokens;
    std::set<std::string>    seen;
    std::string              acc;
    auto flush = [&] {
        if (acc.size() < 5) { acc.clear(); return; }
        std::string lc = acc;
        std::transform(lc.begin(), lc.end(), lc.begin(),
            [](unsigned char c) { return std::tolower(c); });
        acc.clear();
        if (stop.count(lc)) return;
        if (seen.insert(lc).second) tokens.push_back(std::move(lc));
    };
    for (char c : err) {
        const unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) || c == '_' || c == '-' || c == '.') {
            acc.push_back(c);
        } else {
            flush();
        }
    }
    flush();
    return tokens;
}

// buglog.json is a JSON array of {error_message, root_cause, fix, tags}.
// Score each entry by how many tokens of the user's error appear in
// either error_message or tags; return the top match (if any).
[[nodiscard]] std::string scan_buglog(std::string_view error_text)
{
    if (error_text.empty()) return {};
    auto tokens = tokenize_error(error_text);
    if (tokens.empty()) return {};

    const std::string body = read_file(buglog_path());
    if (body.empty()) return {};

    nlohmann::json j;
    try {
        j = nlohmann::json::parse(body);
    } catch (...) {
        return {};
    }
    if (!j.is_array()) return {};

    int               best_score = 0;
    const nlohmann::json* best   = nullptr;
    for (const auto& entry : j) {
        if (!entry.is_object()) continue;
        const std::string em = entry.value("error_message", std::string{});
        std::string em_lc = em;
        std::transform(em_lc.begin(), em_lc.end(), em_lc.begin(),
            [](unsigned char c) { return std::tolower(c); });

        // Tags can be array or string; normalise to lowercase blob.
        std::string tags_lc;
        if (auto t = entry.find("tags"); t != entry.end()) {
            if (t->is_array()) {
                for (const auto& x : *t) {
                    if (x.is_string()) tags_lc.append(x.get<std::string>()).push_back(' ');
                }
            } else if (t->is_string()) {
                tags_lc = t->get<std::string>();
            }
            std::transform(tags_lc.begin(), tags_lc.end(), tags_lc.begin(),
                [](unsigned char c) { return std::tolower(c); });
        }

        int score = 0;
        for (const auto& tok : tokens) {
            if (em_lc.find(tok) != std::string::npos)   ++score;
            if (tags_lc.find(tok) != std::string::npos) ++score;
        }
        if (score > best_score) {
            best_score = score;
            best       = &entry;
        }
    }
    if (!best || best_score < 2) return {};

    std::ostringstream out;
    out << "\n\n---\n## Matched buglog entry (score " << best_score << ")\n";
    if (auto e = best->find("error_message"); e != best->end() && e->is_string()) {
        out << "**error**: `" << e->get<std::string>() << "`\n";
    }
    if (auto e = best->find("root_cause"); e != best->end() && e->is_string()) {
        out << "**root cause**: " << e->get<std::string>() << "\n";
    }
    if (auto e = best->find("fix"); e != best->end() && e->is_string()) {
        out << "**fix**: " << e->get<std::string>() << "\n";
    }
    return out.str();
}

[[nodiscard]] std::string load_runbook(const std::string& component)
{
    // Repo tree wins so dev edits land without a reinstall.
    auto path = std::filesystem::path(repo_runbooks_dir()) / (component + ".md");
    auto body = read_file(path);
    if (!body.empty()) return body;

    const auto installed = installed_runbooks_dir();
    if (!installed.empty()) {
        path = std::filesystem::path(installed) / (component + ".md");
        body = read_file(path);
    }
    return body;
}

} // namespace

ToolDef make_install_runbook()
{
    ToolDef d;
    d.name   = "install_runbook";
    d.schema = make_schema();
    d.invoke = [](const nlohmann::json& args)
        -> std::expected<ToolResult, AgentError>
    {
        const std::string component = args.at("component").get<std::string>();
        if (component.empty()) {
            return ToolResult{false, "bad args: component is empty"};
        }
        // Path-traversal guard. The whitelist *also* excludes anything
        // with a `/` or `..` so even if a future runbook lands outside
        // the directory we don't open it.
        if (!allowed_components().count(component)
            || component.find('/')  != std::string::npos
            || component.find("..") != std::string::npos) {
            return ToolResult{false,
                "bad args: unknown component `" + component + "`"};
        }

        std::string body = load_runbook(component);
        if (body.empty()) {
            return ToolResult{false,
                "no runbook found for `" + component + "` (looked in "
                + repo_runbooks_dir() + ")"};
        }

        std::string error_text;
        if (auto it = args.find("error_text");
            it != args.end() && it->is_string()) {
            error_text = it->get<std::string>();
        }
        if (!error_text.empty()) {
            body += scan_buglog(error_text);
        }
        return ToolResult{true, std::move(body)};
    };
    return d;
}

} // namespace onebit::agent::tools
