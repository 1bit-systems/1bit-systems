#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <variant>
#include <vector>

namespace onebit::core {

// Variants chosen to mirror the failure modes of the underlying C++ loaders
// (rcpp_status_t): I/O, bad-magic, unsupported-version, truncated, invalid
// config, unknown tokenizer piece, sampler.
struct ErrorIo {
    std::filesystem::path path;
    std::error_code       ec;
};
struct ErrorRawIo {
    std::error_code ec;
};
struct ErrorBadMagic {
    std::array<std::uint8_t, 4> expected;
    std::array<std::uint8_t, 4> got;
};
struct ErrorUnsupportedVersion {
    std::int32_t version;
    std::int32_t min;
    std::int32_t max;
};
struct ErrorTruncated {
    std::size_t offset;
    std::size_t needed;
    std::size_t have;
};
struct ErrorInvalidConfig {
    const char* msg;
};
struct ErrorUnknownBytePiece {
    std::vector<std::uint8_t> piece;
};
struct ErrorSampler {
    const char* msg;
};

class HaloError {
public:
    using Variant = std::variant<
        ErrorIo,
        ErrorRawIo,
        ErrorBadMagic,
        ErrorUnsupportedVersion,
        ErrorTruncated,
        ErrorInvalidConfig,
        ErrorUnknownBytePiece,
        ErrorSampler>;

    explicit HaloError(Variant v) : v_(std::move(v)) {}

    [[nodiscard]] const Variant& variant() const noexcept { return v_; }

    [[nodiscard]] std::string what() const;

    static HaloError io(std::filesystem::path path, std::error_code ec)
    {
        return HaloError{ErrorIo{std::move(path), ec}};
    }
    static HaloError raw_io(std::error_code ec)
    {
        return HaloError{ErrorRawIo{ec}};
    }
    static HaloError bad_magic(std::array<std::uint8_t, 4> expected,
                               std::array<std::uint8_t, 4> got)
    {
        return HaloError{ErrorBadMagic{expected, got}};
    }
    static HaloError unsupported_version(std::int32_t v, std::int32_t mn,
                                         std::int32_t mx)
    {
        return HaloError{ErrorUnsupportedVersion{v, mn, mx}};
    }
    static HaloError truncated(std::size_t off, std::size_t need, std::size_t have)
    {
        return HaloError{ErrorTruncated{off, need, have}};
    }
    static HaloError invalid_config(const char* msg)
    {
        return HaloError{ErrorInvalidConfig{msg}};
    }
    static HaloError unknown_byte_piece(std::vector<std::uint8_t> piece)
    {
        return HaloError{ErrorUnknownBytePiece{std::move(piece)}};
    }
    static HaloError sampler(const char* msg)
    {
        return HaloError{ErrorSampler{msg}};
    }

private:
    Variant v_;
};

} // namespace onebit::core
