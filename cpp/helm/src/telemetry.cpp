#include "onebit/helm/telemetry.hpp"

#include <nlohmann/json.hpp>

namespace onebit::helm {

namespace {

std::string_view trim_trailing_cr(std::string_view s)
{
    while (!s.empty() && s.back() == '\r') s.remove_suffix(1);
    return s;
}

std::string_view ltrim_space(std::string_view s)
{
    while (!s.empty() && s.front() == ' ') s.remove_prefix(1);
    return s;
}

} // namespace

std::optional<LiveStats> parse_stats(std::string_view data)
{
    if (data.empty()) return std::nullopt;
    nlohmann::json v;
    try {
        v = nlohmann::json::parse(data);
    } catch (const nlohmann::json::parse_error&) {
        return std::nullopt;
    }
    if (!v.is_object()) return std::nullopt;
    LiveStats s;
    s.loaded_model          = v.value("loaded_model",          std::string{});
    s.tok_s_decode          = v.value("tok_s_decode",          0.0F);
    s.gpu_temp_c            = v.value("gpu_temp_c",            0.0F);
    s.gpu_util_pct          = v.value("gpu_util_pct",          std::uint8_t{0});
    s.npu_up                = v.value("npu_up",                false);
    s.shadow_burn_exact_pct = v.value("shadow_burn_exact_pct", 0.0F);
    s.stale                 = v.value("stale",                 false);
    if (v.contains("services") && v["services"].is_array()) {
        for (const auto& row : v["services"]) {
            if (!row.is_object() || !row.contains("name")) continue;
            ServiceDot d;
            d.name   = row["name"].get<std::string>();
            d.active = row.value("active", false);
            s.services.push_back(std::move(d));
        }
    }
    return s;
}

std::optional<std::string>
extract_landing_payload(std::string_view line)
{
    line = trim_trailing_cr(line);
    if (line.empty() || line.front() == ':') return std::nullopt;
    constexpr std::string_view kPrefix = "data:";
    if (line.size() < kPrefix.size()
        || line.substr(0, kPrefix.size()) != kPrefix) {
        return std::nullopt;
    }
    auto payload = ltrim_space(line.substr(kPrefix.size()));
    if (payload == "[DONE]") return std::nullopt;
    return std::string(payload);
}

} // namespace onebit::helm
