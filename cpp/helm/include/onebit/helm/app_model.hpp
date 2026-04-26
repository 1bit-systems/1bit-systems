// 1bit-helm — model-layer state, no Qt UI dependencies.
//
// The Qt MainWindow (see app_window.hpp) holds one of these and binds
// signals/slots to it. Headless tests construct it directly without
// pulling QApplication. Mirrors crates/1bit-helm/src/app.rs::HelmApp
// data plus build_chat_body / apply_ui_msg / flush_conversation —
// every pure helper that previously lived on HelmApp.

#pragma once

#include "onebit/helm/conversation.hpp"
#include "onebit/helm/models.hpp"
#include "onebit/helm/session.hpp"
#include "onebit/helm/telemetry.hpp"

#include <nlohmann/json.hpp>

#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace onebit::helm {

// Top-bar pane discriminator. Persisted across restarts via QSettings.
enum class Pane : std::uint8_t { Status, Chat, Models, Settings };

[[nodiscard]] std::string_view pane_label(Pane p) noexcept;
[[nodiscard]] std::optional<Pane> pane_from_string(std::string_view s) noexcept;
inline constexpr Pane PANES_ALL[] = {
    Pane::Status, Pane::Chat, Pane::Models, Pane::Settings,
};

// Worker → UI messages. Background threads emit these; the model
// applies them through `apply_ui_msg` which is pure (so tests can
// drive the same path as the live worker).
struct UiTelemetrySnapshot { LiveStats stats; };
struct UiTelemetryDisconnect { std::string reason; };
struct UiChatDelta { std::string content; };
struct UiChatDone {};
struct UiChatError { std::string message; };
struct UiModelsOk { std::vector<ModelCard> cards; };
struct UiModelsErr { std::string message; };
struct UiToast { std::string text; };

using UiMsg = std::variant<
    UiTelemetrySnapshot, UiTelemetryDisconnect,
    UiChatDelta, UiChatDone, UiChatError,
    UiModelsOk, UiModelsErr,
    UiToast>;

// Brand strings — surfaced verbatim in the about dialog + hero strip.
inline constexpr std::string_view BRAND        = "1bit monster";
inline constexpr std::string_view BRAND_DOMAIN = "1bit.systems";

// Every field in Rust's HelmApp that wasn't a runtime/HTTP handle.
// The Qt window holds one of these by value; signals/slots pivot off
// the model and re-render. Pure data — no Qt types.
struct AppModel {
    SessionConfig            cfg{};
    std::string              gateway_url{};
    std::string              landing_url{"http://127.0.0.1:8190"};
    Pane                     current_pane{Pane::Status};

    Conversation             chat_conv{};
    std::string              chat_input{};
    std::optional<std::string> chat_streaming{};

    std::vector<ModelCard>   models{};
    std::optional<std::string> models_error{};

    LiveStats                live{};
    bool                     live_connected{false};
    std::optional<std::string> live_last_error{};

    bool                     show_bearer_modal{false};
    std::string              bearer_input{};

    std::optional<std::string> toast{};
    std::optional<std::string> last_error{};

    std::filesystem::path    log_root{};
};

// Construct a default-state model from env-derived config.
[[nodiscard]] AppModel make_app_model(SessionConfig cfg);

// Build the /v1/chat/completions JSON body. Pure — exposed for
// tests + the network worker.
[[nodiscard]] nlohmann::json build_chat_body(const SessionConfig& cfg,
                                             const Conversation&  conv);

// Apply one worker message to the model. Returns true if the model
// changed (so the UI knows to repaint).
bool apply_ui_msg(AppModel& m, const UiMsg& msg);

// Flush the current conversation to JSONL. Returns the path written
// or the failure reason. Skips entirely when the conversation is
// empty (mirrors the Rust `flush_conversation` early-return).
[[nodiscard]] std::expected<std::optional<std::filesystem::path>, std::string>
flush_conversation(const AppModel& m);

// Env-key helpers — load the `HALO_HELM_*` / `HALO_GAIA_*` fallback
// chain. Pure on a `getenv`-shaped callable so tests can inject a
// fake. Returns the first non-empty value, else nullopt.
// Note: matches std::getenv exactly so &std::getenv is convertible.
using EnvLookup = char* (*)(const char*);
[[nodiscard]] std::optional<std::string>
env_any(EnvLookup get, std::initializer_list<const char*> keys);

// Defaults for `make_app_model` based on the live process env. Real
// `std::getenv`. The CLI / main wires this in.
[[nodiscard]] SessionConfig load_config_from_env();
[[nodiscard]] std::string   load_landing_url_from_env();

} // namespace onebit::helm
