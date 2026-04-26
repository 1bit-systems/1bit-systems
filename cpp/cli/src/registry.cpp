#include "onebit/cli/registry.hpp"

#include "onebit/cli/paths.hpp"

#include <toml++/toml.hpp>

#include <cstdlib>
#include <fmt/format.h>
#include <fstream>
#include <sstream>
#include <system_error>
#include <utility>

namespace onebit::cli {

namespace {

[[nodiscard]] std::filesystem::path canonical_override_path()
{
    if (const char* env = std::getenv("ONEBIT_PACKAGES_TOML");
        env != nullptr && *env != '\0')
    {
        return std::filesystem::path(env);
    }
    return {};
}

}  // namespace

std::filesystem::path canonical_path()
{
    if (auto over = canonical_override_path(); !over.empty()) {
        return over;
    }
    const std::filesystem::path system_path("/etc/1bit/packages.toml");
    std::error_code ec;
    if (std::filesystem::is_regular_file(system_path, ec)) {
        return system_path;
    }
    // Repo-root dev convenience — current working directory's
    // packages.toml. Mirrors the Rust crate's `include_str!` resolution
    // behavior where the file is co-located with the workspace root.
    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    if (!ec) {
        const auto repo = cwd / "packages.toml";
        if (std::filesystem::is_regular_file(repo, ec)) {
            return repo;
        }
    }
    return system_path;  // canonical default — caller surfaces "not found"
}

std::filesystem::path overlay_path()
{
    return xdg_config_home() / "1bit" / "packages.local.toml";
}

Manifest& merge_into(Manifest& base, Manifest overlay)
{
    for (auto& [name, oc] : overlay.components) {
        oc.origin = PackageOrigin::Overlay;
        auto it = base.components.find(name);
        if (it == base.components.end()) {
            base.components.emplace(name, std::move(oc));
            continue;
        }
        // Field-level overwrite on every populated overlay field. We use
        // `populated` semantics rather than always-overwrite so an overlay
        // that only sets one field doesn't blank out the rest. Matches
        // Rust's "later wins per-key" with deep merge.
        Component& bc = it->second;
        if (!oc.description.empty()) bc.description = std::move(oc.description);
        if (!oc.deps.empty())        bc.deps     = std::move(oc.deps);
        if (!oc.build.empty())       bc.build    = std::move(oc.build);
        if (!oc.units.empty())       bc.units    = std::move(oc.units);
        if (!oc.packages.empty())    bc.packages = std::move(oc.packages);
        if (!oc.files.empty())       bc.files    = std::move(oc.files);
        if (!oc.check.empty())       bc.check    = std::move(oc.check);
        bc.origin = PackageOrigin::Overlay;
    }
    for (auto& [id, om] : overlay.models) {
        om.origin = PackageOrigin::Overlay;
        auto it = base.models.find(id);
        if (it == base.models.end()) {
            base.models.emplace(id, std::move(om));
            continue;
        }
        Model& bm = it->second;
        if (!om.description.empty()) bm.description = std::move(om.description);
        if (!om.hf_repo.empty())     bm.hf_repo     = std::move(om.hf_repo);
        if (!om.hf_file.empty())     bm.hf_file     = std::move(om.hf_file);
        if (!om.sha256.empty())      bm.sha256      = std::move(om.sha256);
        if (om.size_mb != 0)         bm.size_mb     = om.size_mb;
        if (!om.license.empty())     bm.license     = std::move(om.license);
        if (!om.requires_.empty())   bm.requires_   = std::move(om.requires_);
        bm.origin = PackageOrigin::Overlay;
    }
    return base;
}

std::expected<Manifest, Error> load_default()
{
    auto canon = parse_manifest_file(canonical_path(), PackageOrigin::Canonical);
    if (!canon) {
        return std::unexpected(canon.error());
    }
    Manifest m = std::move(*canon);

    const auto overlay = overlay_path();
    std::error_code ec;
    if (!std::filesystem::is_regular_file(overlay, ec)) {
        return m;  // missing overlay is not an error
    }
    auto over = parse_manifest_file(overlay, PackageOrigin::Overlay);
    if (!over) {
        return std::unexpected(over.error());
    }
    merge_into(m, std::move(*over));
    return m;
}

std::expected<void, Error>
overlay_add(const std::filesystem::path& overlay_file,
            const OverlayAddRequest& req)
{
    if (req.name.empty()) {
        return std::unexpected(Error::invalid("registry add: name is empty"));
    }

    // Read existing overlay if any (preserve unrelated entries).
    toml::table root;
    std::error_code ec;
    if (std::filesystem::is_regular_file(overlay_file, ec)) {
        try {
            root = toml::parse_file(overlay_file.string());
        } catch (const toml::parse_error& e) {
            return std::unexpected(
                Error::parse("overlay parse: " + std::string(e.description())));
        }
    }

    // Ensure [component] table exists.
    auto* comp_node = root.get("component");
    if (comp_node == nullptr) {
        root.insert("component", toml::table{});
        comp_node = root.get("component");
    }
    auto* comp_tab = comp_node->as_table();
    if (comp_tab == nullptr) {
        return std::unexpected(
            Error::schema("overlay [component] is not a table"));
    }

    toml::table entry;
    entry.insert("description", req.description);

    if (!req.deps.empty()) {
        toml::array deps;
        for (const auto& d : req.deps) deps.push_back(d);
        entry.insert("deps", std::move(deps));
    }
    if (!req.units.empty()) {
        toml::array units;
        for (const auto& u : req.units) units.push_back(u);
        entry.insert("units", std::move(units));
    }
    if (!req.check.empty()) {
        entry.insert("check", req.check);
    }

    comp_tab->insert_or_assign(req.name, std::move(entry));

    std::filesystem::create_directories(overlay_file.parent_path(), ec);
    std::ofstream out(overlay_file, std::ios::binary | std::ios::trunc);
    if (!out) {
        return std::unexpected(
            Error::io("cannot write " + overlay_file.string()));
    }
    out << root;
    return {};
}

std::vector<std::string> render_registry_list(const Manifest& m)
{
    std::vector<std::string> rows;
    rows.reserve(m.components.size() + m.models.size() + 4);
    rows.emplace_back("components:");
    for (const auto& [name, c] : m.components) {
        rows.emplace_back(fmt::format("  [{:<9}] {:<18} {}",
                                      origin_label(c.origin),
                                      name,
                                      c.description));
    }
    if (!m.models.empty()) {
        rows.emplace_back("");
        rows.emplace_back("models:");
        for (const auto& [id, mm] : m.models) {
            rows.emplace_back(fmt::format("  [{:<9}] {:<28} {}",
                                          origin_label(mm.origin),
                                          id,
                                          mm.description));
        }
    }
    return rows;
}

}  // namespace onebit::cli
