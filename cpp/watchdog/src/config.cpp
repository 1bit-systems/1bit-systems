#include "onebit/watchdog/config.hpp"

#include <toml++/toml.hpp>

#include <fstream>
#include <sstream>
#include <utility>

namespace onebit::watchdog {

std::string_view to_string(WatchKind k) noexcept
{
    switch (k) {
        case WatchKind::Github:      return "github";
        case WatchKind::Huggingface: return "huggingface";
    }
    return "unknown";
}

namespace {

// Returns nullopt if the kind string is not one of the two recognised values.
std::optional<WatchKind> parse_kind(std::string_view s)
{
    if (s == "github")      return WatchKind::Github;
    if (s == "huggingface") return WatchKind::Huggingface;
    return std::nullopt;
}

// Convert a [[a, b], [c]] argv-list TOML node into vector<vector<string>>.
// Any non-string element is dropped silently — matches Rust's serde where a
// bad shape would have failed; we err on the conservative-tolerant side
// here because the manifest is hand-edited.
std::vector<std::vector<std::string>>
extract_argv_list(const toml::node* node)
{
    std::vector<std::vector<std::string>> out;
    if (!node || !node->is_array()) {
        return out;
    }
    for (const auto& outer : *node->as_array()) {
        if (!outer.is_array()) continue;
        std::vector<std::string> argv;
        for (const auto& inner : *outer.as_array()) {
            if (auto sv = inner.value<std::string>(); sv) {
                argv.push_back(*sv);
            }
        }
        out.push_back(std::move(argv));
    }
    return out;
}

} // namespace

std::optional<Manifest> Manifest::from_toml(std::string_view raw,
                                            ManifestError*   err)
{
    toml::table tbl;
    try {
        tbl = toml::parse(raw);
    } catch (const toml::parse_error&) {
        if (err) *err = ManifestError::ParseFailed;
        return std::nullopt;
    }

    Manifest m;
    const auto* watch_node = tbl.get("watch");
    if (!watch_node) {
        return m; // empty watch section is OK
    }
    if (!watch_node->is_table()) {
        if (err) *err = ManifestError::SchemaInvalid;
        return std::nullopt;
    }

    for (const auto& [key, val] : *watch_node->as_table()) {
        if (!val.is_table()) continue;
        const auto& t = *val.as_table();

        // kind is required
        const auto kind_str = t["kind"].value<std::string>();
        if (!kind_str) continue;
        const auto kind = parse_kind(*kind_str);
        if (!kind) continue;

        WatchEntry e;
        e.id   = std::string(key.str());
        e.kind = *kind;

        if (auto v = t["repo"].value<std::string>(); v) e.repo = *v;
        if (auto v = t["branch"].value<std::string>(); v) e.branch = *v;
        if (auto v = t["soak_hours"].value<std::int64_t>(); v && *v >= 0) {
            e.soak_hours = static_cast<std::uint32_t>(*v);
        } else {
            e.soak_hours = 24;
        }
        if (auto v = t["notify"].value<std::string>(); v) e.notify = *v;
        e.on_merge = extract_argv_list(t.get("on_merge"));
        e.on_bump  = extract_argv_list(t.get("on_bump"));

        m.watch.emplace(e.id, std::move(e));
    }

    return m;
}

std::optional<Manifest> Manifest::load(std::string_view path,
                                       ManifestError*   err)
{
    std::ifstream ifs{std::string(path)};
    if (!ifs.is_open()) {
        if (err) *err = ManifestError::ReadFailed;
        return std::nullopt;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return Manifest::from_toml(oss.str(), err);
}

} // namespace onebit::watchdog
