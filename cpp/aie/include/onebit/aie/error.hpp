// onebit::aie::Error — typed surface for the NPU backend.
//
// Mirrors AieError from the Rust crate. NotYetWired is the canonical
// "no real hardware/library available" return; dispatch tests pivot on
// it.

#pragma once

#include <string>
#include <string_view>

namespace onebit::aie {

enum class ErrorKind {
    XclbinNotFound,
    ShapeMismatch,
    DtypeMismatch,
    Xrt,                    // libxrt runtime error; carries xrt error code
    NotYetWired,            // skeleton path / no libxrt at runtime
    LibraryUnavailable,     // dlopen(libxrt_coreutil.so) failed
};

class Error {
public:
    Error(ErrorKind kind, std::string detail) noexcept
        : kind_{kind}, detail_{std::move(detail)} {}

    Error(ErrorKind kind, std::string detail, int xrt_code) noexcept
        : kind_{kind}, detail_{std::move(detail)}, xrt_code_{xrt_code} {}

    [[nodiscard]] ErrorKind        kind()     const noexcept { return kind_; }
    [[nodiscard]] std::string_view detail()   const noexcept { return detail_; }
    [[nodiscard]] int              xrt_code() const noexcept { return xrt_code_; }
    [[nodiscard]] std::string      what()     const;

private:
    ErrorKind   kind_;
    std::string detail_{};
    int         xrt_code_{0};
};

[[nodiscard]] std::string_view label(ErrorKind kind) noexcept;

} // namespace onebit::aie
