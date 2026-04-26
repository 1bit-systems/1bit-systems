#pragma once

// `1bit registry push` — upload catalog deltas to Cloudflare R2
// (S3-compatible). Stateful: only files whose SHA256 changed since last
// push are sent. State at $XDG_STATE_HOME/1bit/r2-pushed.json.

#include "onebit/cli/error.hpp"

#include <chrono>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::cli::push {

struct R2Config {
    std::string account_id;
    std::string access_key_id;
    std::string secret_access_key;
    std::string bucket;
    std::string region          = "auto";       // R2 default
    std::string endpoint_host;                   // <account>.r2.cloudflarestorage.com
    std::chrono::seconds timeout{120};

    [[nodiscard]] static std::expected<R2Config, Error>
    from_env_file(const std::filesystem::path& env_path);
};

enum class Tier { Lossy, Lossless, Both };

struct PushOptions {
    std::filesystem::path        catalogs_root;     // /var/lib/1bit/catalogs
    std::filesystem::path        state_file;        // $XDG_STATE_HOME/1bit/r2-pushed.json
    std::optional<std::string>   only_slug;         // empty = all catalogs
    Tier                         tier = Tier::Both;
    bool                         dry_run = false;
    bool                         verbose = true;
};

struct PushReport {
    std::size_t scanned   = 0;
    std::size_t uploaded  = 0;
    std::size_t skipped   = 0;     // unchanged sha256
    std::size_t failed    = 0;
    std::vector<std::string> failures;  // "key: reason"
    std::uint64_t bytes_uploaded = 0;
};

// Drive the push. Reads state, walks catalogs, computes sha256s, uploads
// changed objects via S3 PUT (SigV4-signed), persists new state.
[[nodiscard]] std::expected<PushReport, Error>
run(const R2Config& cfg, const PushOptions& opts);

// SigV4 helper — exposed for unit tests against AWS test vectors.
[[nodiscard]] std::string
sigv4_sign_put(std::string_view              host,
               std::string_view              bucket,
               std::string_view              key,
               std::string_view              region,
               std::string_view              access_key_id,
               std::string_view              secret_access_key,
               std::string_view              payload_sha256_hex,
               std::string_view              amz_date,    // YYYYMMDDTHHMMSSZ
               std::string_view              date_stamp); // YYYYMMDD

}  // namespace onebit::cli::push
