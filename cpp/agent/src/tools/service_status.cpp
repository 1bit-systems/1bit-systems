// service_status — query systemctl --user for the status of a 1bit-*
// service. Read-only, no start/stop. Operator can use this tool to ask
// Halo "is the lemonade up?" or "did halo-coder die?".

#include "onebit/agent/tools/registry.hpp"
#include "proc_shim.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <string>
#include <string_view>

namespace onebit::agent::tools {

namespace {

// Allowlist of unit-name prefixes the brain can poke. Anything else is
// refused so a prompt-injected user can't make Halo enumerate the
// operator's whole systemd state.
constexpr std::array<std::string_view, 5> kAllowedPrefixes = {
    "1bit-halo-",
    "halo-agent@",
    "halo-",
    "strix-",
    "1bit-",
};

[[nodiscard]] bool unit_allowed(std::string_view name)
{
    for (const auto& pfx : kAllowedPrefixes) {
        if (name.starts_with(pfx)) return true;
    }
    return false;
}

[[nodiscard]] nlohmann::json make_schema()
{
    return nlohmann::json{
        {"type", "function"},
        {"function", {
            {"name", "service_status"},
            {"description",
             "Query systemctl --user for the active state + last few "
             "log lines of a 1bit-* / halo-* / strix-* user service. "
             "Read-only; no start/stop/restart. Use this when the user "
             "asks whether a service is up, why it crashed, etc."},
            {"parameters", {
                {"type", "object"},
                {"properties", {
                    {"unit", {
                        {"type", "string"},
                        {"description", "Unit name, e.g. \"halo-agent@halo-helpdesk\" or \"1bit-halo-lemonade\"."},
                    }},
                }},
                {"required", nlohmann::json::array({"unit"})},
            }},
        }},
    };
}

} // namespace

ToolDef make_service_status()
{
    ToolDef d;
    d.name   = "service_status";
    d.schema = make_schema();
    d.invoke = [](const nlohmann::json& args)
        -> std::expected<ToolResult, AgentError>
    {
        const std::string unit = args.at("unit").get<std::string>();
        if (unit.empty()) {
            return ToolResult{false, "bad args: unit is empty"};
        }
        if (!unit_allowed(unit)) {
            return ToolResult{false,
                "service_status: refused — unit not on allowlist (must start with "
                "1bit-halo- / halo-agent@ / halo- / strix- / 1bit-)"};
        }
        // Strip any caller-injected shell metas defensively even though
        // we go via argv.
        for (char c : unit) {
            if (c == ';' || c == '|' || c == '&' || c == '`' || c == '$') {
                return ToolResult{false,
                    "service_status: refused — shell metacharacter in unit"};
            }
        }
        std::vector<std::string> argv{
            "systemctl", "--user", "status", unit, "--no-pager", "-n", "20",
        };
        auto r = run_or_stub("service_status", argv);
        if (!r) return std::unexpected(std::move(r.error()));
        // systemctl status exit codes: 0=active, 3=inactive, 4=no-such-unit.
        // Treat all <128 as "report and let the brain decide".
        std::string out;
        out.reserve(r->stdout_text.size() + r->stderr_text.size() + 64);
        out += "exit=" + std::to_string(r->exit_code) + "\n```\n";
        out += r->stdout_text;
        if (!r->stderr_text.empty()) {
            out += "\n--- stderr ---\n";
            out += r->stderr_text;
        }
        if (!out.empty() && out.back() != '\n') out += '\n';
        out += "```";
        return ToolResult{true, std::move(out)};
    };
    return d;
}

} // namespace onebit::agent::tools
