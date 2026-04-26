// onebit::onnx::Error — typed surface for the ONNX lane.
//
// Mirrors the Rust crate's `OnnxError` enum so callers in C++ can
// distinguish "operator pointed us at the wrong directory" from "ORT
// runtime not installed" from "VitisAI EP not registered" without
// string matching.
//
// Used as the `E` in `std::expected<T, Error>` throughout this module.

#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace onebit::onnx {

// Discriminant for Error. One per failure mode the Rust crate exposes,
// plus an Ok sentinel that should never be observed (use std::expected
// — Ok is implicit on the success branch).
enum class ErrorKind {
    MissingArtifact,        // a required file in the artifact dir is gone
    NotAnArtifactDir,       // root has no model.onnx — wrong directory
    InvalidGenAiConfig,     // genai_config.json failed to parse
    OrtRuntimeUnavailable,  // libonnxruntime.so not loadable / not built
    VitisaiUnavailable,     // EP requested but not registered
    SessionInit,            // ORT session creation / forward-pass failure
    TokenizerLoad,          // tokenizer.json could not be loaded/decoded
    Io,                     // generic filesystem error
};

// Typed error. `path` is set when `kind` carries a path payload
// (MissingArtifact, NotAnArtifactDir); empty otherwise. `detail` is a
// human-readable string suitable for log lines and HTTP error bodies.
class Error {
public:
    Error(ErrorKind kind, std::string detail) noexcept
        : kind_{kind}, detail_{std::move(detail)} {}

    // Path-bearing overload. The path parameter is taken as
    // `std::filesystem::path` by value; the 3-arg form is required when a
    // path is set so a 2-arg call (kind + string-literal) routes
    // unambiguously to the (kind, detail) overload above. Callers that
    // want a path with no detail should pass an explicit empty string:
    // `Error{kind, path, {}}`.
    Error(ErrorKind kind, std::filesystem::path path, std::string detail) noexcept
        : kind_{kind}, path_{std::move(path)}, detail_{std::move(detail)} {}

    [[nodiscard]] ErrorKind                       kind()   const noexcept { return kind_; }
    [[nodiscard]] const std::filesystem::path&    path()   const noexcept { return path_; }
    [[nodiscard]] std::string_view                detail() const noexcept { return detail_; }

    // Human-readable one-liner. Stable enough for log greps.
    [[nodiscard]] std::string what() const;

private:
    ErrorKind             kind_;
    std::filesystem::path path_{};
    std::string           detail_{};
};

// Stable label per kind (e.g. "missing_artifact"). Used for /metrics
// labels and server log fields.
[[nodiscard]] std::string_view label(ErrorKind kind) noexcept;

} // namespace onebit::onnx
