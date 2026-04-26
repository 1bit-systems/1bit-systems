// onebit::kokoro::Error — typed surface for the TTS engine.
//
// One-to-one mapping with KokoroError in the Rust crate so callers
// can lift errors out of either implementation interchangeably.

#pragma once

#include <string>
#include <string_view>

namespace onebit::kokoro {

enum class ErrorKind {
    UnsupportedStub,    // build did not link halo-kokoro
    ModelLoadFailed,    // path / corrupt weights / OOM
    VoiceNotFound,      // voice id not in voice pack
    InvalidText,        // empty / whitespace-only
    InvalidVoice,       // empty
    InvalidSpeed,       // <= 0, > 4, NaN, or +inf
    ShimError,          // halo-kokoro returned non-zero
    InvalidPath,        // interior NUL byte in a string
};

class Error {
public:
    Error(ErrorKind kind, std::string detail) noexcept
        : kind_{kind}, detail_{std::move(detail)} {}

    [[nodiscard]] ErrorKind         kind()   const noexcept { return kind_; }
    [[nodiscard]] std::string_view  detail() const noexcept { return detail_; }
    [[nodiscard]] std::string       what()   const;

    // For ShimError specifically — the shim's status code is preserved
    // in `code_` so callers can match on it without parsing detail_.
    Error(ErrorKind kind, std::string detail, int code) noexcept
        : kind_{kind}, detail_{std::move(detail)}, code_{code} {}
    [[nodiscard]] int code() const noexcept { return code_; }

    // For InvalidSpeed — preserve the offending value so server logs
    // can echo it back without re-parsing detail_.
    Error(ErrorKind kind, std::string detail, float speed) noexcept
        : kind_{kind}, detail_{std::move(detail)}, speed_{speed} {}
    [[nodiscard]] float speed() const noexcept { return speed_; }

private:
    ErrorKind   kind_;
    std::string detail_{};
    int         code_{0};
    float       speed_{0.0f};
};

[[nodiscard]] std::string_view label(ErrorKind kind) noexcept;

} // namespace onebit::kokoro
