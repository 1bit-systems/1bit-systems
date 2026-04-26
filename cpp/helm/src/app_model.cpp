#include "onebit/helm/app_model.hpp"
#include "onebit/helm/conv_log.hpp"

#include <cstdlib>

namespace onebit::helm {

namespace {
// Adapter: std::getenv is `char* (*)(const char*) noexcept` in C++23.
// EnvLookup is `const char* (*)(const char*)` (no noexcept). The
// noexcept-narrowing direction can't auto-decay through the function-
// pointer type, so we route through a tiny shim that has the exact
// EnvLookup signature.
char* getenv_shim(const char* k) {
    return std::getenv(k);
}
} // namespace

std::string_view pane_label(Pane p) noexcept
{
    switch (p) {
        case Pane::Status:   return "Status";
        case Pane::Chat:     return "Chat";
        case Pane::Models:   return "Models";
        case Pane::Settings: return "Settings";
    }
    return "Status";
}

std::optional<Pane> pane_from_string(std::string_view s) noexcept
{
    if (s == "Status")   return Pane::Status;
    if (s == "Chat")     return Pane::Chat;
    if (s == "Models")   return Pane::Models;
    if (s == "Settings") return Pane::Settings;
    return std::nullopt;
}

std::optional<std::string>
env_any(EnvLookup get, std::initializer_list<const char*> keys)
{
    for (const auto* k : keys) {
        if (const auto* v = get(k); v && *v) {
            return std::string(v);
        }
    }
    return std::nullopt;
}

SessionConfig load_config_from_env()
{
    // Lemonade default :8200, mirrors crates/1bit-helm/src/main.rs.
    auto url = env_any(&getenv_shim, {"HALO_HELM_URL", "HALO_GAIA_URL"})
                   .value_or("http://127.0.0.1:8200");
    auto model = env_any(&getenv_shim, {"HALO_HELM_MODEL", "HALO_GAIA_MODEL"})
                     .value_or("1bit-monster-2b");
    SessionConfig cfg(std::move(url), std::move(model));
    cfg.bearer = env_any(&getenv_shim, {"HALO_HELM_TOKEN", "HALO_GAIA_TOKEN"});
    return cfg;
}

std::string load_landing_url_from_env()
{
    return env_any(&getenv_shim, {"HALO_HELM_LANDING", "HALO_GAIA_LANDING"})
        .value_or("http://127.0.0.1:8190");
}

AppModel make_app_model(SessionConfig cfg)
{
    AppModel m;
    m.gateway_url = cfg.server_url;
    m.cfg         = std::move(cfg);
    m.log_root    = default_log_root();
    return m;
}

nlohmann::json
build_chat_body(const SessionConfig& cfg, const Conversation& conv)
{
    auto messages = nlohmann::json::array();
    if (cfg.system_prompt) {
        messages.push_back({{"role", "system"}, {"content", *cfg.system_prompt}});
    }
    for (auto& m : conv.to_openai_messages()) {
        messages.push_back(std::move(m));
    }
    return nlohmann::json{
        {"model",    cfg.default_model},
        {"messages", std::move(messages)},
        {"stream",   true},
    };
}

bool apply_ui_msg(AppModel& m, const UiMsg& msg)
{
    return std::visit(
        [&](const auto& alt) -> bool {
            using T = std::decay_t<decltype(alt)>;
            if constexpr (std::is_same_v<T, UiTelemetrySnapshot>) {
                m.live            = alt.stats;
                m.live_connected  = true;
                m.live_last_error = std::nullopt;
                return true;
            } else if constexpr (std::is_same_v<T, UiTelemetryDisconnect>) {
                m.live_connected  = false;
                m.live_last_error = alt.reason;
                return true;
            } else if constexpr (std::is_same_v<T, UiChatDelta>) {
                if (m.chat_streaming.has_value()) {
                    m.chat_streaming->append(alt.content);
                }
                return true;
            } else if constexpr (std::is_same_v<T, UiChatDone>) {
                if (m.chat_streaming.has_value()
                    && !m.chat_streaming->empty()) {
                    m.chat_conv.push_assistant(std::move(*m.chat_streaming));
                }
                m.chat_streaming.reset();
                return true;
            } else if constexpr (std::is_same_v<T, UiChatError>) {
                m.chat_streaming.reset();
                m.last_error = "chat: " + alt.message;
                return true;
            } else if constexpr (std::is_same_v<T, UiModelsOk>) {
                m.models       = alt.cards;
                m.models_error = std::nullopt;
                return true;
            } else if constexpr (std::is_same_v<T, UiModelsErr>) {
                m.models_error = alt.message;
                return true;
            } else if constexpr (std::is_same_v<T, UiToast>) {
                m.toast = alt.text;
                return true;
            } else {
                return false;
            }
        },
        msg);
}

std::expected<std::optional<std::filesystem::path>, std::string>
flush_conversation(const AppModel& m)
{
    if (m.chat_conv.empty()) {
        return std::optional<std::filesystem::path>{};
    }
    auto rc = write_session(m.log_root, m.chat_conv);
    if (!rc) return std::unexpected(rc.error());
    return std::optional<std::filesystem::path>{*rc};
}

} // namespace onebit::helm
