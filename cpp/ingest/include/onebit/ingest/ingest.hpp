#pragma once

// 1bit-ingest — source-side tooling for `.1bl` catalog curators.
//
// Four verbs:
//   prepare       — walk a FLAC dir, sidecar a JSON manifest, tar the corpus.
//   pack          — assemble a .1bl from a trained GGUF + catalog.toml + sidecars.
//   validate      — verify footer SHA-256, pretty-print the manifest.
//   add-residual  — append RESIDUAL_BLOB + RESIDUAL_INDEX to a lossy .1bl.
//
// See docs/wiki/1bl-container-spec.md for the byte layout this crate writes.

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace onebit::ingest {

// ----------------------------------------------------------------------
// .1bl format constants

inline constexpr std::array<std::uint8_t, 4> MAGIC{'1', 'B', 'L', 0x01};
inline constexpr std::uint8_t                VERSION = 0x01;

namespace tag {
inline constexpr std::uint8_t MODEL_GGUF      = 0x01;
inline constexpr std::uint8_t COVER           = 0x02;
inline constexpr std::uint8_t TRACK_LYRICS    = 0x03;
inline constexpr std::uint8_t ATTRIBUTION_TXT = 0x04;
inline constexpr std::uint8_t LICENSE_TXT     = 0x05;
inline constexpr std::uint8_t RESIDUAL_BLOB   = 0x10;
inline constexpr std::uint8_t RESIDUAL_INDEX  = 0x11;
} // namespace tag

// ----------------------------------------------------------------------
// Error

struct ErrIo {
    std::filesystem::path path;
    std::string           message;
};
struct ErrBadMagic {
    std::array<std::uint8_t, 4> got;
};
struct ErrBadVersion {
    std::uint8_t version;
};
struct ErrFooterHashMismatch { };
struct ErrTruncated {
    const char* what;
};
struct ErrHeaderTooLarge {
    std::uint64_t bytes;
};
struct ErrCbor {
    std::string message;
};
struct ErrToml {
    std::string message;
};
struct ErrInvalid {
    std::string message;
};

class IngestError {
public:
    using Variant = std::variant<ErrIo, ErrBadMagic, ErrBadVersion,
                                 ErrFooterHashMismatch, ErrTruncated,
                                 ErrHeaderTooLarge, ErrCbor, ErrToml,
                                 ErrInvalid>;

    explicit IngestError(Variant v) : v_{std::move(v)} {}

    [[nodiscard]] const Variant& variant() const noexcept { return v_; }
    [[nodiscard]] std::string    what() const;

    static IngestError io(std::filesystem::path p, std::string m)
    {
        return IngestError{ErrIo{std::move(p), std::move(m)}};
    }
    static IngestError bad_magic(std::array<std::uint8_t, 4> g)
    {
        return IngestError{ErrBadMagic{g}};
    }
    static IngestError bad_version(std::uint8_t v) { return IngestError{ErrBadVersion{v}}; }
    static IngestError footer_mismatch() { return IngestError{ErrFooterHashMismatch{}}; }
    static IngestError truncated(const char* w) { return IngestError{ErrTruncated{w}}; }
    static IngestError header_too_large(std::uint64_t b)
    {
        return IngestError{ErrHeaderTooLarge{b}};
    }
    static IngestError cbor(std::string m) { return IngestError{ErrCbor{std::move(m)}}; }
    static IngestError toml(std::string m) { return IngestError{ErrToml{std::move(m)}}; }
    static IngestError invalid(std::string m) { return IngestError{ErrInvalid{std::move(m)}}; }

private:
    Variant v_;
};

// ----------------------------------------------------------------------
// prepare

struct FlacEntry {
    std::string   rel_path;
    std::uint64_t size_bytes{0};
    std::string   sha256;
};

struct CorpusManifest {
    std::string            version{"0.1"};
    std::uint64_t          created_unix{0};
    std::string            tool;
    std::vector<FlacEntry> entries;
};

struct PrepareSummary {
    std::size_t   flac_count{0};
    std::uint64_t total_bytes{0};
};

[[nodiscard]] std::expected<PrepareSummary, IngestError>
prepare(const std::filesystem::path& src_dir, const std::filesystem::path& out_path);

// ----------------------------------------------------------------------
// pack

struct CodecMeta {
    std::string   audio;
    std::uint32_t sample_rate{0};
    std::uint32_t channels{0};
};

struct ModelMeta {
    std::string   arch;
    std::uint64_t params{0};
    double        bpw{0.0};
    std::string   sha256; // populated by pack()
};

struct Track {
    std::string   id;
    std::string   title;
    std::uint64_t length_ms{0};
    std::string   sha256;
};

struct Manifest {
    std::string                v{"0.1"};
    std::string                catalog;
    std::string                title;
    std::string                artist;
    std::string                license;
    std::optional<std::string> license_url;
    std::optional<std::string> attribution;
    std::optional<std::string> source;
    std::string                created;
    std::string                tier{"lossy"};
    CodecMeta                  codec{};
    ModelMeta                  model{};
    std::vector<Track>         tracks;
    bool                       residual_present{false};
    std::optional<std::string> residual_sha256;
};

struct CatalogToml {
    std::string                catalog;
    std::string                title;
    std::string                artist;
    std::string                license;
    std::optional<std::string> license_url;
    std::optional<std::string> attribution;
    std::optional<std::string> source;
    std::string                created;
    std::string                tier{"lossy"};
    CodecMeta                  codec{};
    ModelMeta                  model{};
    std::vector<Track>         tracks;
    std::optional<std::string> license_txt;
    std::optional<std::string> license_txt_path;
    std::optional<std::string> attribution_txt;
};

struct PackSummary {
    std::size_t   section_count{0};
    std::uint64_t total_bytes{0};
};

// Public for tests.
[[nodiscard]] std::expected<CatalogToml, IngestError>
parse_catalog_toml(std::string_view text);

[[nodiscard]] std::expected<PackSummary, IngestError>
pack(const std::filesystem::path&                 model_path,
     const std::filesystem::path&                 manifest_toml_path,
     const std::optional<std::filesystem::path>&  cover_path,
     const std::optional<std::filesystem::path>&  lyrics_path,
     const std::filesystem::path&                 out_path);

// ----------------------------------------------------------------------
// validate

struct SectionRecord {
    std::uint8_t  tag{0};
    std::uint64_t offset{0};
    std::uint64_t length{0};
};

struct ValidateReport {
    std::uint64_t              total_bytes{0};
    std::uint8_t               version{0};
    Manifest                   manifest{};
    std::vector<SectionRecord> sections;
    bool                       footer_ok{false};
};

[[nodiscard]] std::expected<ValidateReport, IngestError>
validate(const std::filesystem::path& path);

[[nodiscard]] std::expected<ValidateReport, IngestError>
parse_bytes(std::span<const std::uint8_t> buf);

[[nodiscard]] std::string format_report(const ValidateReport& r);

// ----------------------------------------------------------------------
// add-residual

struct ResidualSummary {
    std::uint64_t residual_bytes{0};
    std::uint64_t index_bytes{0};
};

[[nodiscard]] std::expected<ResidualSummary, IngestError>
add_residual(const std::filesystem::path& input,
             const std::filesystem::path& residual,
             const std::filesystem::path& index,
             const std::filesystem::path& out);

} // namespace onebit::ingest
