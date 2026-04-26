#pragma once

// Read-side wrapper around onebit::ingest::validate(). Holds the path +
// the validated manifest + section index so the HTTP layer can stream a
// sub-range without re-parsing.

#include "onebit/ingest/ingest.hpp"

#include <expected>
#include <filesystem>
#include <string>
#include <vector>

namespace onebit::stream {

namespace tag {
inline constexpr std::uint8_t MODEL_GGUF      = onebit::ingest::tag::MODEL_GGUF;
inline constexpr std::uint8_t COVER           = onebit::ingest::tag::COVER;
inline constexpr std::uint8_t TRACK_LYRICS    = onebit::ingest::tag::TRACK_LYRICS;
inline constexpr std::uint8_t ATTRIBUTION_TXT = onebit::ingest::tag::ATTRIBUTION_TXT;
inline constexpr std::uint8_t LICENSE_TXT     = onebit::ingest::tag::LICENSE_TXT;
inline constexpr std::uint8_t RESIDUAL_BLOB   = onebit::ingest::tag::RESIDUAL_BLOB;
inline constexpr std::uint8_t RESIDUAL_INDEX  = onebit::ingest::tag::RESIDUAL_INDEX;
} // namespace tag

struct Section {
    std::uint8_t  tag{0};
    std::uint64_t offset{0};
    std::uint64_t length{0};

    [[nodiscard]] bool is_lossy_tier() const noexcept
    {
        return tag != onebit::ingest::tag::RESIDUAL_BLOB &&
               tag != onebit::ingest::tag::RESIDUAL_INDEX;
    }
};

struct Catalog {
    std::filesystem::path path;
    onebit::ingest::Manifest manifest;
    std::vector<Section>   sections;
    std::uint64_t          total_bytes{0};
    std::uint64_t          footer_offset{0};

    [[nodiscard]] const std::string& slug() const noexcept
    {
        return manifest.catalog;
    }
};

[[nodiscard]] std::expected<Catalog, onebit::ingest::IngestError>
                                     open_catalog(const std::filesystem::path& path);

// Materialize a lossy-only `.1bl` in memory: magic + original CBOR header
// + lossy-tier sections + recomputed SHA-256 footer.
[[nodiscard]] std::expected<std::vector<std::uint8_t>, onebit::ingest::IngestError>
                                     build_lossy_bytes(const Catalog& cat);

} // namespace onebit::stream
