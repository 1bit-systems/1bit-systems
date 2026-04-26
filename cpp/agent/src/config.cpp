#include "onebit/agent/config.hpp"

#include <toml++/toml.hpp>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace onebit::agent {

namespace {

// Replaces every occurrence of "${ENV:NAME}" inside `s` with
// std::getenv("NAME"). Missing env vars expand to "" — the loader
// validates required fields after substitution.
std::string expand_env(std::string_view s)
{
    constexpr std::string_view kPrefix = "${ENV:";
    std::string out;
    out.reserve(s.size());
    std::size_t i = 0;
    while (i < s.size()) {
        auto rest = s.substr(i);
        if (rest.starts_with(kPrefix)) {
            auto end = rest.find('}', kPrefix.size());
            if (end != std::string_view::npos) {
                auto name = rest.substr(kPrefix.size(), end - kPrefix.size());
                std::string name_z(name);
                if (const char* v = std::getenv(name_z.c_str()); v != nullptr) {
                    out.append(v);
                }
                i += end + 1;
                continue;
            }
        }
        out.push_back(s[i]);
        ++i;
    }
    return out;
}

void expand_inplace(std::string& s) { s = expand_env(s); }

template <class T>
T get_or(const toml::table& tbl, std::string_view key, T fallback)
{
    if (const auto* node = tbl.get(key)) {
        if (auto v = node->value<T>()) return *v;
    }
    return fallback;
}

// String getter — toml++ can't `value<std::string>` from a non-string
// node; this normalizes that path.
std::string get_str_or(const toml::table& tbl, std::string_view key, std::string fallback)
{
    if (const auto* node = tbl.get(key)) {
        if (auto v = node->value<std::string>()) return *v;
    }
    return fallback;
}

std::expected<Config, AgentError>
parse_table(const toml::table& root)
{
    Config cfg;

    if (const auto* a = root.get_as<toml::table>("agent")) {
        cfg.agent.name              = get_str_or(*a, "name", cfg.agent.name);
        cfg.agent.brain_url         = get_str_or(*a, "brain_url", cfg.agent.brain_url);
        cfg.agent.system_prompt     = get_str_or(*a, "system_prompt", cfg.agent.system_prompt);
        cfg.agent.model             = get_str_or(*a, "model", cfg.agent.model);
        cfg.agent.max_history       = static_cast<std::int32_t>(
            get_or<std::int64_t>(*a, "max_history", cfg.agent.max_history));
        cfg.agent.max_tool_iters    = static_cast<std::int32_t>(
            get_or<std::int64_t>(*a, "max_tool_iters", cfg.agent.max_tool_iters));
        cfg.agent.request_timeout_ms = static_cast<std::int32_t>(
            get_or<std::int64_t>(*a, "request_timeout_ms", cfg.agent.request_timeout_ms));
        cfg.agent.stream            = get_or<bool>(*a, "stream", cfg.agent.stream);
        cfg.agent.temperature       = get_or<double>(*a, "temperature", cfg.agent.temperature);
    }

    if (const auto* a = root.get_as<toml::table>("adapter")) {
        cfg.adapter.kind      = get_str_or(*a, "kind", cfg.adapter.kind);
        cfg.adapter.token     = get_str_or(*a, "token", cfg.adapter.token);
        cfg.adapter.bind_host = get_str_or(*a, "bind_host", cfg.adapter.bind_host);
        auto port_64          = get_or<std::int64_t>(*a, "bind_port", 0);
        if (port_64 < 0 || port_64 > 65535) {
            return std::unexpected(AgentError::config(
                "adapter.bind_port out of range [0, 65535]"));
        }
        cfg.adapter.bind_port = static_cast<std::uint16_t>(port_64);
    }

    if (const auto* a = root.get_as<toml::table>("memory")) {
        cfg.memory.sqlite_path = get_str_or(*a, "sqlite_path", std::string{});
        cfg.memory.keep_messages = get_or<std::int64_t>(*a, "keep_messages", 0);
    }

    if (const auto* a = root.get_as<toml::table>("tools")) {
        if (const auto* arr = a->get_as<toml::array>("enabled")) {
            for (const auto& el : *arr) {
                if (auto s = el.value<std::string>()) {
                    cfg.tools.enabled.push_back(*s);
                }
            }
        }
        if (const auto* sub = a->get_as<toml::table>("agent_consult")) {
            cfg.tools.agent_consult.peer_name      = get_str_or(*sub, "peer_name", std::string{});
            cfg.tools.agent_consult.peer_brain_url = get_str_or(*sub, "peer_brain_url", std::string{});
            cfg.tools.agent_consult.peer_model     = get_str_or(*sub, "peer_model", std::string{});
        }
        if (const auto* sub = a->get_as<toml::table>("speak_to_echo")) {
            cfg.tools.speak_to_echo.echo_url   = get_str_or(*sub, "echo_url",
                cfg.tools.speak_to_echo.echo_url);
            cfg.tools.speak_to_echo.auto_speak = get_or<bool>(*sub, "auto_speak",
                cfg.tools.speak_to_echo.auto_speak);
        }
    }

    // ${ENV:...} expansion across all string fields. Done after parse
    // so `${ENV:DISCORD_TOKEN}` in TOML lands literally and we expand
    // here once.
    expand_inplace(cfg.agent.name);
    expand_inplace(cfg.agent.brain_url);
    expand_inplace(cfg.agent.system_prompt);
    expand_inplace(cfg.agent.model);
    expand_inplace(cfg.adapter.kind);
    expand_inplace(cfg.adapter.token);
    expand_inplace(cfg.adapter.bind_host);
    {
        auto p = cfg.memory.sqlite_path.string();
        expand_inplace(p);
        cfg.memory.sqlite_path = std::filesystem::path(p);
    }
    for (auto& t : cfg.tools.enabled) expand_inplace(t);

    // Schema validation. Cheap checks; deeper invariants belong in
    // the components themselves.
    if (cfg.agent.brain_url.empty()) {
        return std::unexpected(AgentError::config("agent.brain_url is required"));
    }
    if (cfg.agent.max_history < 0) {
        return std::unexpected(AgentError::config("agent.max_history must be >= 0"));
    }
    if (cfg.agent.max_tool_iters < 0) {
        return std::unexpected(AgentError::config("agent.max_tool_iters must be >= 0"));
    }
    if (cfg.agent.request_timeout_ms <= 0) {
        return std::unexpected(AgentError::config("agent.request_timeout_ms must be > 0"));
    }
    if (cfg.adapter.kind.empty()) {
        return std::unexpected(AgentError::config("adapter.kind is required"));
    }
    if (cfg.memory.sqlite_path.empty()) {
        return std::unexpected(AgentError::config("memory.sqlite_path is required"));
    }
    return cfg;
}

} // namespace

std::expected<Config, AgentError>
parse_config(std::string_view toml_text)
{
    toml::table root;
    try {
        root = toml::parse(toml_text);
    } catch (const toml::parse_error& e) {
        std::ostringstream os;
        os << "toml parse: " << e.description() << " at "
           << e.source().begin;
        return std::unexpected(AgentError::config(os.str()));
    }
    return parse_table(root);
}

std::expected<Config, AgentError>
load_config(const std::filesystem::path& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return std::unexpected(AgentError::config(
            "open " + path.string() + ": cannot read"));
    }
    std::ostringstream buf;
    buf << f.rdbuf();
    if (!f) {
        return std::unexpected(AgentError::config(
            "read " + path.string() + ": failed"));
    }
    return parse_config(buf.str());
}

} // namespace onebit::agent
