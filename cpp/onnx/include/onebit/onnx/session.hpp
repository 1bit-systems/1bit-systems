// onebit::onnx::Session — thin wrapper around Ort::Session that selects
// the EP at runtime (VitisAI first → CPU fallback).
//
// Mirrors the Rust `OnnxSession` split:
//   * `load_config_only(root)`   — parses tokenizer + genai_config,
//                                   never touches libonnxruntime.so.
//   * `load(root)`               — full session init, only available
//                                   when the build linked ORT.
//
// pImpl: the public header is ORT-agnostic so callers don't transitively
// pull in onnxruntime_cxx_api.h. Implementation in src/session.cpp.

#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <string_view>

#include "onebit/onnx/error.hpp"
#include "onebit/onnx/model.hpp"

namespace onebit::onnx {

// Which EP the session actually landed on. Distinct from the *requested*
// EP — fallback to CPU is expected on Linux STX-H today.
enum class ExecutionLane {
    Cpu,
    Vitisai,
};

[[nodiscard]] constexpr std::string_view label(ExecutionLane lane) noexcept {
    return lane == ExecutionLane::Vitisai ? "ort-vitisai" : "ort-cpu";
}

// Session state. Holds artifact paths, parsed config, and (when ORT is
// linked) a live Ort::Session via pImpl.
class Session {
public:
    // Move-only — Ort::Session is not copyable.
    Session(Session&&) noexcept;
    Session& operator=(Session&&) noexcept;
    Session(const Session&)            = delete;
    Session& operator=(const Session&) = delete;
    ~Session();

    // Tokenizer + config only. Always available, even without ORT.
    [[nodiscard]] static std::expected<Session, Error>
    load_config_only(const std::filesystem::path& root);

    // Full session including Ort::Session. Returns
    // ErrorKind::OrtRuntimeUnavailable if this build did not link ORT.
    [[nodiscard]] static std::expected<Session, Error>
    load(const std::filesystem::path& root);

    [[nodiscard]] const ArtifactPaths& paths()  const noexcept { return paths_; }
    [[nodiscard]] const GenAiConfig&   config() const noexcept { return config_; }
    [[nodiscard]] ExecutionLane        lane()   const noexcept { return lane_; }

    // True when this build was linked against ORT and `load` would
    // succeed if the artifact existed.
    [[nodiscard]] static bool runtime_available() noexcept;

    // True when *this instance* has a live Ort::Session attached
    // (i.e. constructed via `load`, not `load_config_only`).
    [[nodiscard]] bool has_runtime() const noexcept;

private:
    Session() = default;

    ArtifactPaths   paths_{};
    GenAiConfig     config_{};
    ExecutionLane   lane_{ExecutionLane::Cpu};

    // pImpl carries Ort::Session + Ort::Env in the .cpp file so callers
    // never include onnxruntime_cxx_api.h transitively. Null when this
    // is a config-only session OR when the build didn't link ORT.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace onebit::onnx
