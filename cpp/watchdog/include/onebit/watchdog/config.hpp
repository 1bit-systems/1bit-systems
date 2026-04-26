#pragma once

// packages.toml watch-section parser. Reads ONLY the [watch.*] tables from
// the workspace manifest; ignores [component.*] / [model.*] so this module
// stays narrow.

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::watchdog {

enum class WatchKind {
    Github,
    Huggingface,
};

struct WatchEntry {
    std::string                            id;
    WatchKind                              kind = WatchKind::Github;
    std::string                            repo;
    std::optional<std::string>             branch;
    std::uint32_t                          soak_hours = 24;
    std::vector<std::vector<std::string>>  on_merge;
    std::vector<std::vector<std::string>>  on_bump;
    std::string                            notify;
};

enum class ManifestError {
    ReadFailed,
    ParseFailed,
    SchemaInvalid,
};

struct Manifest {
    // Sorted by id, mirrors Rust BTreeMap iteration order.
    std::map<std::string, WatchEntry> watch;

    // Load from disk. Returns std::nullopt on failure (sets *err if non-null).
    static std::optional<Manifest> load(std::string_view path,
                                        ManifestError* err = nullptr);

    // Parse from an in-memory TOML string. Same nullopt-on-failure contract.
    static std::optional<Manifest> from_toml(std::string_view raw,
                                             ManifestError* err = nullptr);
};

[[nodiscard]] std::string_view to_string(WatchKind k) noexcept;

} // namespace onebit::watchdog
