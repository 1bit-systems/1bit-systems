#pragma once

// On-disk shape of `packages.toml` + `packages.local.toml`. The TOML schema
// matches the Rust crate's `install::{Component, Model, FileEntry}` exactly
// so existing manifests round-trip 1:1.
//
// Two `FileEntry` shapes are accepted (matching the Rust `#[serde(untagged)]`
// enum):
//
//   files = [["src", "dest"], ...]                          # legacy pair
//   files = [{ src = "src", dst = "dest",                   # new table
//              substitute = { USER = "$USER" } }, ...]
//
// The `Origin` field tracks which file the entry came from so
// `1bit registry list` can mark canonical vs overlay rows.

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::cli {

enum class PackageOrigin : std::uint8_t {
    Canonical = 0,  ///< /etc/1bit/packages.toml or repo-root packages.toml
    Overlay   = 1,  ///< $XDG_CONFIG_HOME/1bit/packages.local.toml
};

[[nodiscard]] constexpr std::string_view origin_label(PackageOrigin o) noexcept
{
    return o == PackageOrigin::Canonical ? "canonical" : "overlay";
}

// One entry under `files = [...]` on a component. We collapse the two
// untagged-enum shapes into a single struct because the C++ side reads
// only the {src, dst, substitute} triple.
struct FileEntry {
    std::string                        src;
    std::string                        dst;
    std::map<std::string, std::string> substitute;  ///< @KEY@ → expansion seed
};

// One [component.<name>] table.
struct Component {
    std::string                                  name;
    std::string                                  description;
    std::vector<std::string>                     deps;
    std::vector<std::vector<std::string>>        build;
    std::vector<std::string>                     units;
    std::vector<std::string>                     packages;
    std::vector<FileEntry>                       files;
    std::string                                  check;
    PackageOrigin                                origin = PackageOrigin::Canonical;
};

// One [model.<id>] table — instant-load weight slot.
struct Model {
    std::string              id;
    std::string              description;
    std::string              hf_repo;
    std::string              hf_file;
    std::string              sha256;       ///< "" / "UPSTREAM" / "PENDING-*" / hex
    std::uint64_t            size_mb = 0;
    std::string              license;
    std::vector<std::string> requires_;     ///< `requires` is a C++ keyword adjacent
    PackageOrigin            origin = PackageOrigin::Canonical;
};

// Parsed in-memory manifest. Components + models are keyed by name.
// std::map gives us deterministic ordering for `--list` output, matching
// the Rust BTreeMap behavior.
struct Manifest {
    std::map<std::string, Component> components;
    std::map<std::string, Model>     models;
};

// True iff `expand_placeholder` would rewrite this seed (currently
// "$USER" / "$HOME" — anything else is treated as a literal).
[[nodiscard]] bool is_placeholder_seed(std::string_view raw) noexcept;

// Resolve a substitute seed string to its concrete value. Reads from the
// process environment (`USER` / `HOME`) when applicable; literals pass
// through unchanged.
[[nodiscard]] std::string expand_placeholder(std::string_view raw);

}  // namespace onebit::cli
