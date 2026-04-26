#include "onebit/agent/tools/registry.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <map>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace onebit::agent {

namespace tools {

namespace {

std::mutex             g_test_mu;
RunCaptureFn           g_test_run_capture;

} // namespace

void set_test_run_capture(RunCaptureFn fn)
{
    std::lock_guard lk(g_test_mu);
    g_test_run_capture = std::move(fn);
}

// Internal accessor for the per-tool source files.
RunCaptureFn current_test_run_capture();
RunCaptureFn current_test_run_capture()
{
    std::lock_guard lk(g_test_mu);
    return g_test_run_capture;
}

std::string trim_and_cap(std::string_view raw)
{
    // Trim trailing whitespace (incl. \n) — leading whitespace stays so
    // diff-style and table-style outputs render correctly when a brain
    // pastes the content back.
    auto end = raw.size();
    while (end > 0) {
        const unsigned char c = static_cast<unsigned char>(raw[end - 1]);
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
            --end;
            continue;
        }
        break;
    }
    std::string_view body(raw.data(), end);

    if (body.size() <= kMaxContentBytes) {
        return std::string(body);
    }
    const std::size_t dropped = body.size() - kMaxContentBytes;
    std::string out;
    out.reserve(kMaxContentBytes + 64);
    out.append(body.substr(0, kMaxContentBytes));
    out.append("\n\n... [truncated ");
    out.append(std::to_string(dropped));
    out.append(" bytes]");
    return out;
}

namespace {

[[nodiscard]] bool type_matches(const nlohmann::json& value, std::string_view want)
{
    if (want == "string")  return value.is_string();
    if (want == "integer") return value.is_number_integer();
    if (want == "number")  return value.is_number();
    if (want == "boolean") return value.is_boolean();
    if (want == "array")   return value.is_array();
    if (want == "object")  return value.is_object();
    return true; // unknown type tag — accept rather than spuriously reject
}

} // namespace

std::optional<std::string>
validate_args(const nlohmann::json& schema, const nlohmann::json& args)
{
    if (!schema.is_object()) return "schema is not an object";
    if (!args.is_object())   return "args must be a JSON object";

    // required[]
    if (auto it = schema.find("required"); it != schema.end() && it->is_array()) {
        for (const auto& r : *it) {
            if (!r.is_string()) continue;
            const std::string key = r.get<std::string>();
            if (!args.contains(key)) {
                return "missing required field: " + key;
            }
        }
    }

    // properties{name: {type: ...}}
    if (auto pit = schema.find("properties"); pit != schema.end() && pit->is_object()) {
        for (auto it = pit->begin(); it != pit->end(); ++it) {
            const std::string& key = it.key();
            if (!args.contains(key)) continue;     // optional, absent is fine
            const auto& prop = it.value();
            if (!prop.is_object()) continue;
            auto type_it = prop.find("type");
            if (type_it == prop.end() || !type_it->is_string()) continue;
            const std::string want = type_it->get<std::string>();
            if (!type_matches(args.at(key), want)) {
                return "field `" + key + "` expected " + want;
            }
        }
    }
    return std::nullopt;
}

} // namespace tools

// ---- ToolRegistry::Impl --------------------------------------------------

struct ToolRegistry::Impl {
    // Insertion order matters for list_tools_openai_format() determinism
    // (brain prompts cache better when the tools[] array is stable).
    std::vector<tools::ToolDef> ordered;
    std::unordered_map<std::string, std::size_t> by_name;
};

ToolRegistry::ToolRegistry()                                   = default;
ToolRegistry::~ToolRegistry()                                  = default;
ToolRegistry::ToolRegistry(ToolRegistry&&) noexcept            = default;
ToolRegistry& ToolRegistry::operator=(ToolRegistry&&) noexcept = default;

void ToolRegistry::ensure_impl()
{
    if (!p_) p_ = std::make_unique<Impl>();
}

void ToolRegistry::register_tool(tools::ToolDef def)
{
    ensure_impl();
    auto it = p_->by_name.find(def.name);
    if (it != p_->by_name.end()) {
        // Replace existing — useful for hot reload + tests.
        p_->ordered[it->second] = std::move(def);
        return;
    }
    p_->by_name.emplace(def.name, p_->ordered.size());
    p_->ordered.push_back(std::move(def));
}

ToolRegistry::BuildOutcome
ToolRegistry::build(const std::vector<std::string>& enabled)
{
    return build(enabled, BuildOptions{});
}

ToolRegistry::BuildOutcome
ToolRegistry::build(const std::vector<std::string>& enabled,
                    const BuildOptions& opts)
{
    ensure_impl();
    BuildOutcome out;

    // Closure over the auto-confirm flag so gh_issue_create's lambda sees
    // the right default without a global. The factory returns a fresh
    // ToolDef per call, so the override only sticks for this build.
    using FactoryFn = tools::ToolDef (*)();
    const std::map<std::string, FactoryFn> factories{
        {"repo_search",      &tools::make_repo_search},
        {"bench_lookup",     &tools::make_bench_lookup},
        {"install_runbook",  &tools::make_install_runbook},
        {"gh_issue_create",  &tools::make_gh_issue_create},
    };

    for (const auto& name : enabled) {
        auto it = factories.find(name);
        if (it == factories.end()) {
            out.warnings.push_back("unknown tool in [tools] enabled: " + name);
            continue;
        }
        tools::ToolDef def = it->second();
        if (name == "gh_issue_create" && opts.gh_issue_auto_confirm) {
            // Patch the schema so the OpenAI doc reflects the policy the
            // brain is actually allowed to play under. We don't change
            // the dispatch — the lambda itself reads `confirm` from
            // args, and the operator-level auto-confirm flag is enforced
            // by injecting `confirm:true` defaults at call time below.
            // Copy out before mutating — string_view referencing the
            // json's internal buffer could dangle if assignment
            // reorganizes the underlying storage.
            std::string desc =
                def.schema["function"]["description"].get<std::string>();
            desc += " (auto-confirm enabled)";
            def.schema["function"]["description"] = std::move(desc);
            def.invoke = [orig = std::move(def.invoke)](
                             const nlohmann::json& args)
                -> std::expected<ToolResult, AgentError>
            {
                // Auto-confirm policy: force confirm:true regardless of
                // what the brain emitted. The schema still requires the
                // field, so a totally-absent confirm is caught upstream
                // by validate_args (the operator-level opt-in here is
                // "if the brain says confirm:false, override it"). Brain
                // can no longer suppress an issue creation by accident.
                auto patched = args;
                patched["confirm"] = true;
                return orig(patched);
            };
        }
        register_tool(std::move(def));
    }
    return out;
}

std::vector<nlohmann::json>
ToolRegistry::list_tools_openai_format() const
{
    if (!p_) return {};
    std::vector<nlohmann::json> out;
    out.reserve(p_->ordered.size());
    for (const auto& t : p_->ordered) {
        out.push_back(t.schema);
    }
    return out;
}

std::expected<ToolResult, AgentError>
ToolRegistry::call(const ToolCall& c)
{
    if (!p_) {
        return std::unexpected(AgentError::tool(c.name, "registry not built"));
    }
    auto it = p_->by_name.find(c.name);
    if (it == p_->by_name.end()) {
        return std::unexpected(AgentError::tool(c.name, "unknown tool"));
    }
    const auto& def = p_->ordered[it->second];

    // Schema check — every tool gets validated before dispatch. Rejected
    // calls return success=false content="bad args: ..." so the brain
    // sees a tool message and can recover, vs. an AgentError that would
    // propagate up the loop.
    const auto& schema = def.schema.at("function").at("parameters");
    if (auto bad = tools::validate_args(schema, c.args_json)) {
        return ToolResult{false, "bad args: " + *bad};
    }

    // Wall-clock deadline. `def.invoke` is synchronous; the deadline
    // here is observational — we record overrun on the result. The
    // subprocess helpers themselves do not yet honour SIGTERM-on-timeout
    // (TODO: thread a kill-on-timeout into proc.cpp), so a 5s shell
    // command that hangs forever will hang the loop. Logging the overrun
    // gives us a signal to chase before that ever happens in prod.
    const auto t0 = std::chrono::steady_clock::now();
    auto       r  = def.invoke(c.args_json);
    const auto dt = std::chrono::steady_clock::now() - t0;

    if (!r) {
        // Errors from the tool body (subprocess fail, IO fail) are real
        // AgentErrors — brain gets `role=tool content="error: ..."` via
        // the loop's existing error path.
        return std::unexpected(std::move(r.error()));
    }
    if (dt > tools::kDefaultDeadline) {
        const auto over_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            dt - tools::kDefaultDeadline).count();
        r->content += "\n\n[warning: tool ran "
                    + std::to_string(over_ms)
                    + " ms past 5s deadline]";
    }
    r->content = tools::trim_and_cap(r->content);
    return std::move(*r);
}

std::size_t ToolRegistry::size() const noexcept
{
    return p_ ? p_->ordered.size() : 0;
}

bool ToolRegistry::has(std::string_view name) const noexcept
{
    if (!p_) return false;
    return p_->by_name.find(std::string(name)) != p_->by_name.end();
}

} // namespace onebit::agent
