#pragma once

// Package registry: parses a canonical packages.toml and an optional
// `packages.local.toml` overlay, then deep-merges them.
//
// Merge rules (later wins per-key, deep merge for [component.X] /
// [model.X] tables):
//
//   * Whole table missing in overlay → canonical wins.
//   * Whole table missing in canonical, present in overlay → overlay wins,
//     origin = Overlay.
//   * Both sides present → field-level overwrite, origin tracked from the
//     last writer. Vector fields (deps / build / units / packages / files
//     / requires) are wholesale-overridden, not concatenated; Rust does
//     the same via `serde_with::Replace` semantics.
//
// The overlay file is OPTIONAL — if it doesn't exist on disk, parsing
// returns the canonical manifest unchanged. This matches the prompt:
// "third parties register their own services without touching the
// canonical packages.toml".

#include "onebit/cli/error.hpp"
#include "onebit/cli/package.hpp"

#include <expected>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace onebit::cli {

// Parse a single packages.toml-shaped string. Sets origin on every
// component / model based on the `origin` argument so callers can pass
// the same parser through twice.
[[nodiscard]] std::expected<Manifest, Error>
parse_manifest_str(std::string_view toml, PackageOrigin origin);

// Read + parse one TOML file; identical contract to `parse_manifest_str`.
[[nodiscard]] std::expected<Manifest, Error>
parse_manifest_file(const std::filesystem::path& path, PackageOrigin origin);

// Deep-merge `overlay` onto `base`, tracking origin per-component.
// Mutates `base` in-place and returns a reference to it for chaining.
Manifest& merge_into(Manifest& base, Manifest overlay);

// Resolve the path the overlay would live at, honoring `XDG_CONFIG_HOME`.
// Does NOT verify existence.
[[nodiscard]] std::filesystem::path overlay_path();

// Resolve the canonical manifest path, in priority order:
//   1. ONEBIT_PACKAGES_TOML override (test hook + dev override)
//   2. /etc/1bit/packages.toml
//   3. <CWD>/packages.toml (repo-root dev convenience)
[[nodiscard]] std::filesystem::path canonical_path();

// Convenience: load canonical + optional overlay and deep-merge. Returns
// the merged manifest. Missing overlay is NOT an error.
[[nodiscard]] std::expected<Manifest, Error> load_default();

// `1bit registry add <name> --url <url> --systemd <unit>` writes a new
// minimal Component into the overlay file. If the overlay file does not
// exist it is created (with parent dirs); existing entries are preserved.
struct OverlayAddRequest {
    std::string name;
    std::string description;
    std::vector<std::string> units;
    std::vector<std::string> deps;
    std::string check;          ///< optional health URL
};

[[nodiscard]] std::expected<void, Error>
overlay_add(const std::filesystem::path& overlay_file,
            const OverlayAddRequest& req);

// Render the `1bit registry list` table to stdout-style lines. Returned
// as a vector<string> so callers can choose how to print (println vs
// CLI11 helper). Each row is one component, columns:
//   <origin-tag>  <name>  <description>
[[nodiscard]] std::vector<std::string>
render_registry_list(const Manifest& m);

}  // namespace onebit::cli
