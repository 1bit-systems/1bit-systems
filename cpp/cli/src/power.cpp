#include "onebit/cli/power.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <string>

namespace onebit::cli {

PowerEnvelope envelope_of(PowerProfile p) noexcept
{
    switch (p) {
        case PowerProfile::Inference: return {65000, 80000, 75000, 95};
        case PowerProfile::Chat:      return {45000, 65000, 55000, 90};
        case PowerProfile::Idle:      return {20000, 35000, 25000, 80};
    }
    return {0, 0, 0, 0};
}

std::string_view name_of(PowerProfile p) noexcept
{
    switch (p) {
        case PowerProfile::Inference: return "inference";
        case PowerProfile::Chat:      return "chat";
        case PowerProfile::Idle:      return "idle";
    }
    return "?";
}

std::string_view description_of(PowerProfile p) noexcept
{
    switch (p) {
        case PowerProfile::Inference:
            return "Max sustained decode tok/s — all headroom to package.";
        case PowerProfile::Chat:
            return "Interactive, balanced, low fan — default after boot.";
        case PowerProfile::Idle:
            return "Watchdog-triggered quiet-closet mode — no active requests.";
    }
    return "?";
}

std::vector<std::string> ryzenadj_argv(PowerProfile p)
{
    const auto e = envelope_of(p);
    return {
        fmt::format("--tctl-temp={}",   e.tctl_c),
        fmt::format("--slow-limit={}",  e.slow_mw),
        fmt::format("--fast-limit={}",  e.fast_mw),
        fmt::format("--stapm-limit={}", e.stapm_mw),
    };
}

std::expected<PowerProfile, Error> parse_profile(std::string_view raw)
{
    std::string s;
    s.reserve(raw.size());
    for (char c : raw) {
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
        }
    }
    if (s == "inference" || s == "decode" || s == "perf" || s == "performance") {
        return PowerProfile::Inference;
    }
    if (s == "chat" || s == "balanced" || s == "default") {
        return PowerProfile::Chat;
    }
    if (s == "idle" || s == "silent" || s == "save") {
        return PowerProfile::Idle;
    }
    return std::unexpected(Error::invalid(
        std::string("unknown power profile '") + std::string(raw) +
        "' (try `1bit power --list`)"));
}

std::vector<PowerProfile> list_profiles()
{
    return {PowerProfile::Inference, PowerProfile::Chat, PowerProfile::Idle};
}

namespace {

[[nodiscard]] std::string pick_value(std::string_view line)
{
    if (const auto bar = line.find('|'); bar != std::string_view::npos) {
        std::string_view rest = line.substr(bar + 1);
        // First non-space token.
        std::size_t i = 0;
        while (i < rest.size() && (rest[i] == ' ' || rest[i] == '\t')) ++i;
        std::size_t j = i;
        while (j < rest.size() && rest[j] != ' ' && rest[j] != '\t' && rest[j] != '\n') ++j;
        if (j > i) return std::string(rest.substr(i, j - i));
    }
    // Fallback: last whitespace-separated token.
    std::size_t end = line.size();
    while (end > 0 && (line[end - 1] == ' ' || line[end - 1] == '\t' ||
                       line[end - 1] == '\n' || line[end - 1] == '\r')) {
        --end;
    }
    std::size_t start = end;
    while (start > 0 && line[start - 1] != ' ' && line[start - 1] != '\t') --start;
    if (end > start) return std::string(line.substr(start, end - start));
    return "?";
}

[[nodiscard]] std::string lower_dashed(std::string_view in)
{
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        if (c == '-') out.push_back(' ');
        else          out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

}  // namespace

std::string summarize_info(std::string_view info)
{
    std::string stapm = "?";
    std::string fast  = "?";
    std::string slow  = "?";
    std::string tctl  = "?";

    std::size_t pos = 0;
    while (pos < info.size()) {
        const auto eol = info.find('\n', pos);
        const auto end = eol == std::string_view::npos ? info.size() : eol;
        const std::string_view line = info.substr(pos, end - pos);
        pos = (eol == std::string_view::npos) ? info.size() : eol + 1;

        const std::string norm = lower_dashed(line);
        if      (norm.find("stapm value") != std::string::npos) {
            // live reading — ignore.
        } else if (norm.find("stapm limit") != std::string::npos) {
            stapm = pick_value(line);
        } else if (norm.find("ppt limit fast") != std::string::npos) {
            fast = pick_value(line);
        } else if (norm.find("ppt limit slow") != std::string::npos) {
            slow = pick_value(line);
        } else if (norm.find("tctl temp") != std::string::npos) {
            tctl = pick_value(line);
        } else if (norm.find("thm limit core") != std::string::npos) {
            if (tctl == "?") tctl = pick_value(line);
        }
    }
    return fmt::format("stapm={} fast={} slow={} tctl={}", stapm, fast, slow, tctl);
}

}  // namespace onebit::cli
