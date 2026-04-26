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

// True iff `expand_placeholder` would rewrite this seed when matched as
// the whole input (currently "$USER" / "$HOME" — anything else is treated
// as a literal seed). The substring forms supported by
// `expand_placeholder` (e.g. `${HOME}/foo`) are NOT classified here —
// `is_placeholder_seed` is used by manifest-parse code to decide whether
// the seed itself is a known token, not whether expansion will happen.
[[nodiscard]] bool is_placeholder_seed(std::string_view raw) noexcept;

// Substring-substitute every supported placeholder in `raw` and return
// the rewritten string. Recognized tokens (longest-prefix-first):
//
//   ${XDG_CONFIG_HOME}   ${XDG_DATA_HOME}   ${HOME}   ${USER}
//   $HOME                $USER
//
// Resolution reads the process environment (`HOME`, `USER`,
// `XDG_CONFIG_HOME`, `XDG_DATA_HOME`); the two XDG tokens fall back to
// `$HOME/.config` / `$HOME/.local/share` when the env var is unset, to
// match `paths.cpp`. Strict: no shell-style `${VAR:-default}`, no
// command substitution `$(cmd)`, no glob — every match is a plain
// substring replace. Pass-through for unrecognized literals.
[[nodiscard]] std::string expand_placeholder(std::string_view raw);

}  // namespace onebit::cli
