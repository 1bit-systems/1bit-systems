#include "onebit/cli/update.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <tuple>

namespace onebit::cli {

namespace {

using json = nlohmann::json;

struct ParsedVersion {
    std::int32_t major = 0;
    std::int32_t minor = 0;
    std::int32_t patch = 0;
    std::int32_t pre_rank = INT32_MAX;  ///< INT32_MAX = release, no pre-suffix
};

[[nodiscard]] ParsedVersion parse_version(std::string_view v) noexcept
{
    ParsedVersion out;
    std::string_view core = v;
    std::string_view pre;
    if (const auto dash = v.find('-'); dash != std::string_view::npos) {
        core = v.substr(0, dash);
        pre  = v.substr(dash + 1);
    }
    auto take_num = [](std::string_view& s) -> std::int32_t {
        std::int32_t n = 0;
        std::size_t i = 0;
        while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
            n = n * 10 + (s[i] - '0');
            ++i;
        }
        // Skip the dot if any.
        if (i < s.size() && s[i] == '.') ++i;
        s.remove_prefix(std::min(i, s.size()));
        return n;
    };
    out.major = take_num(core);
    out.minor = take_num(core);
    out.patch = take_num(core);

    if (pre.empty()) {
        out.pre_rank = INT32_MAX;
    } else {
        std::int32_t n = 0;
        bool any_digit = false;
        for (char c : pre) {
            if (c >= '0' && c <= '9') {
                n = n * 10 + (c - '0');
                any_digit = true;
            }
        }
        out.pre_rank = any_digit ? n : -1;
    }
    return out;
}

[[nodiscard]] auto as_tuple(const ParsedVersion& p) noexcept
{
    return std::tie(p.major, p.minor, p.patch, p.pre_rank);
}

[[nodiscard]] ReleaseArtifact parse_artifact(const json& j)
{
    ReleaseArtifact a;
    if (j.contains("platform")) a.platform = j.value("platform", "");
    if (j.contains("kind"))     a.kind     = j.value("kind", "");
    if (j.contains("name") && j["name"].is_string()) a.name = j["name"].get<std::string>();
    a.url    = j.value("url", "");
    if (j.contains("size") && j["size"].is_number_integer()) {
        a.size = j["size"].get<std::uint64_t>();
    }
    a.sha256       = j.value("sha256", "");
    a.minisign_sig = j.value("minisign_sig", "");
    a.primary      = j.value("primary", true);
    return a;
}

}  // namespace

std::expected<Feed, Error> parse_feed(std::string_view bytes)
{
    json j;
    try {
        j = json::parse(bytes);
    } catch (const json::exception& e) {
        return std::unexpected(Error::parse(std::string("releases.json: ") + e.what()));
    }
    if (!j.is_object()) {
        return std::unexpected(Error::schema("releases.json: top-level must be object"));
    }
    Feed f;
    f.latest = j.value("latest", "");
    if (j.contains("channels") && j["channels"].is_object()) {
        for (auto it = j["channels"].begin(); it != j["channels"].end(); ++it) {
            if (it.value().is_string()) {
                f.channels.emplace(it.key(), it.value().get<std::string>());
            }
        }
    }
    if (j.contains("releases") && j["releases"].is_array()) {
        for (const auto& r : j["releases"]) {
            Release rel;
            rel.version        = r.value("version", "");
            rel.date           = r.value("date", "");
            rel.min_compatible = r.value("min_compatible", "");
            rel.notes          = r.value("notes", "");
            if (r.contains("artifacts") && r["artifacts"].is_array()) {
                for (const auto& a : r["artifacts"]) {
                    rel.artifacts.push_back(parse_artifact(a));
                }
            }
            f.releases.push_back(std::move(rel));
        }
    }
    return f;
}

bool is_newer(std::string_view newer, std::string_view older) noexcept
{
    return as_tuple(parse_version(newer)) > as_tuple(parse_version(older));
}

std::string_view current_platform() noexcept
{
#if defined(__x86_64__) && defined(__linux__)
    return "x86_64-linux-gnu";
#elif defined(__aarch64__) && defined(__linux__)
    return "aarch64-linux-gnu";
#else
    return "unknown";
#endif
}

std::optional<PickedUpdate>
pick_update(const Feed& feed, std::string_view current)
{
    if (!is_newer(feed.latest, current)) return std::nullopt;
    auto rel = std::find_if(feed.releases.begin(), feed.releases.end(),
                            [&](const Release& r) { return r.version == feed.latest; });
    if (rel == feed.releases.end()) return std::nullopt;
    const auto plat = current_platform();
    auto art = std::find_if(rel->artifacts.begin(), rel->artifacts.end(),
                            [&](const ReleaseArtifact& a) { return a.platform == plat; });
    if (art == rel->artifacts.end()) return std::nullopt;
    return PickedUpdate{*rel, *art};
}

CheckOutcome classify_check(const Feed& feed, std::string_view current)
{
    if (auto picked = pick_update(feed, current); picked) {
        return CheckAvailable{std::string(current), *picked};
    }
    return CheckUpToDate{std::string(current), feed.latest};
}

int exit_code_for(const CheckOutcome& o) noexcept
{
    if (std::holds_alternative<CheckUpToDate>(o))   return 0;
    if (std::holds_alternative<CheckAvailable>(o))  return 1;
    return 2;
}

}  // namespace onebit::cli
