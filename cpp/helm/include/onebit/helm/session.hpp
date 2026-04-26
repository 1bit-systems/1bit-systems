// 1bit-helm — session-wide config (server URL, auth, default model,
// system prompt). Mirrors crates/1bit-helm/src/session.rs.

#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <string>

namespace onebit::helm {

struct SessionConfig {
    std::string                server_url;
    std::optional<std::string> bearer;
    std::string                default_model;
    std::optional<std::string> system_prompt;

    SessionConfig() = default;
    SessionConfig(std::string url, std::string model)
        : server_url(std::move(url)), default_model(std::move(model))
    {}
};

// Serde-compatible JSON round-trip — same field names as the Rust struct.
[[nodiscard]] nlohmann::json    to_json(const SessionConfig& c);
[[nodiscard]] SessionConfig     from_json(const nlohmann::json& j);

} // namespace onebit::helm
