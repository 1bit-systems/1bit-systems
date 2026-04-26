#pragma once

// `1bit update` — release feed parser + sha256 verifier.
//
// Full minisign verification stays Rust-only for now (no header-only
// minisign C++ lib of comparable maturity); the C++ port wires sha256 +
// feed-fetch + signature-block-stub so the path is end-to-end and a
// minisign C library can drop in later.

#include "onebit/cli/error.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace onebit::cli {

struct ReleaseArtifact {
    std::string platform;       ///< "x86_64-linux-gnu"
    std::string kind;           ///< "appimage", "tarball", "deb", ...
    std::optional<std::string> name;
    std::string url;
    std::optional<std::uint64_t> size;
    std::string sha256;          ///< lowercase hex; "" on auxiliary
    std::string minisign_sig;    ///< full .minisig block; "" on auxiliary
    bool        primary = true;
};

struct Release {
    std::string version;
    std::string date;
    std::vector<ReleaseArtifact> artifacts;
    std::string min_compatible;
    std::string notes;
};

struct Feed {
    std::string                        latest;
    std::map<std::string, std::string> channels;   ///< channel name → version
    std::vector<Release>               releases;
};

[[nodiscard]] std::expected<Feed, Error> parse_feed(std::string_view json_bytes);

// Best-effort semver compare on `major.minor.patch[-rcN]`. Returns true
// iff `newer > older`. Pre-releases sort before their unsuffixed peer.
[[nodiscard]] bool is_newer(std::string_view newer, std::string_view older) noexcept;

// Return the current platform triple in the format the feed uses.
[[nodiscard]] std::string_view current_platform() noexcept;

// Pick the matching artifact (if any) when `feed.latest > current` and
// `release.artifacts[platform == current_platform()]` exists.
struct PickedUpdate {
    Release         release;
    ReleaseArtifact artifact;
};

[[nodiscard]] std::optional<PickedUpdate>
pick_update(const Feed& feed, std::string_view current);

struct CheckUpToDate { std::string current; std::string latest; };
struct CheckAvailable{ std::string current; PickedUpdate picked; };
struct CheckFeedError{ std::string message; };

using CheckOutcome = std::variant<CheckUpToDate, CheckAvailable, CheckFeedError>;

[[nodiscard]] CheckOutcome classify_check(const Feed& feed, std::string_view current);

// 0/1/2 to mirror the Rust CLI's exit semantics.
[[nodiscard]] int exit_code_for(const CheckOutcome& o) noexcept;

// Hex-lower sha256 of a file. Returns hash on success.
[[nodiscard]] std::expected<std::string, Error>
sha256_file(const std::filesystem::path& path);

[[nodiscard]] std::expected<void, Error>
verify_sha256(const std::filesystem::path& path, std::string_view expect_hex);

}  // namespace onebit::cli
