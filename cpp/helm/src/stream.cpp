#include "onebit/helm/stream.hpp"

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

SseEvent parse_sse_line(std::string_view line)
{
    line = trim_trailing_cr(line);
    if (line.empty() || line.front() == ':') {
        return SseIgnore{};
    }
    constexpr std::string_view kPrefix = "data:";
    if (line.size() < kPrefix.size()
        || line.substr(0, kPrefix.size()) != kPrefix) {
        return SseIgnore{};
    }
    auto payload = ltrim_space(line.substr(kPrefix.size()));
    if (payload == "[DONE]") {
        return SseDone{};
    }
    try {
        auto v = nlohmann::json::parse(payload);
        if (!v.contains("choices") || !v["choices"].is_array()
            || v["choices"].empty()) {
            return SseIgnore{};
        }
        const auto& choice = v["choices"][0];
        if (!choice.contains("delta") || !choice["delta"].is_object()) {
            return SseIgnore{};
        }
        const auto& delta = choice["delta"];
        if (!delta.contains("content") || !delta["content"].is_string()) {
            return SseIgnore{};
        }
        auto s = delta["content"].get<std::string>();
        if (s.empty()) return SseIgnore{};
        return SseDelta{std::move(s)};
    } catch (const nlohmann::json::parse_error&) {
        return SseIgnore{};
    }
}

} // namespace onebit::helm
