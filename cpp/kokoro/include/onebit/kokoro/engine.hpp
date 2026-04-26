// onebit::kokoro::Engine — safe C++ facade over halo-kokoro.
//
// Engine owns one libkokoro context (when halo-kokoro is linked) and
// validates all inputs at the C++ boundary so bad inputs never cross
// into the C++ TTS engine.

#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

#include "onebit/kokoro/error.hpp"

namespace onebit::kokoro {

// Metadata describing a successful synthesis call. Mirrors
// `SynthesisInfo` from the Rust crate.
struct SynthesisInfo {
    std::uint32_t sample_rate{22'050};  // kokoro v1 fixed
    std::uint16_t channels{1};          // mono
    std::uint64_t samples{0};           // per-channel sample count
    std::uint64_t duration_ms{0};
};

// The synth output: PCM samples + metadata.
struct SynthesisOutput {
    std::vector<std::int16_t> pcm;
    SynthesisInfo             info;
};

// Default speed clamp range — half-open (0, 4]. Matches kokoro.cpp's
// upstream clamp; we mirror it so bad inputs never cross the FFI.
inline constexpr float kSpeedMin = 0.0f;  // exclusive
inline constexpr float kSpeedMax = 4.0f;  // inclusive

// True when this build was linked against halo-kokoro and `Engine::create`
// would return a real engine. False in stub builds.
[[nodiscard]] bool runtime_available() noexcept;

class Engine {
public:
    // Move-only — owns a unique pImpl.
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;
    Engine(const Engine&)            = delete;
    Engine& operator=(const Engine&) = delete;
    ~Engine();

    // Load a kokoro ONNX model from disk via halo-kokoro. In stub builds
    // returns ErrorKind::UnsupportedStub without touching the filesystem.
    [[nodiscard]] static std::expected<Engine, Error>
    create(const std::filesystem::path& model);

    // Synthesize `text` as `voice` at `speed`. Validates inputs first
    // (text non-empty, voice non-empty, speed in (0, 4]), then routes
    // to halo-kokoro.
    [[nodiscard]] std::expected<SynthesisOutput, Error>
    synthesize(std::string_view text,
               std::string_view voice,
               float            speed);

    // True when this engine has a live halo-kokoro context. False when
    // the build was stub-only.
    [[nodiscard]] bool has_runtime() const noexcept;

private:
    Engine() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Validation primitives — exposed for testing without constructing an
// engine. Used internally by Engine::synthesize.
namespace detail {
[[nodiscard]] std::expected<void, Error> validate_text(std::string_view text)   noexcept;
[[nodiscard]] std::expected<void, Error> validate_voice(std::string_view voice) noexcept;
[[nodiscard]] std::expected<void, Error> validate_speed(float speed)            noexcept;
} // namespace detail

} // namespace onebit::kokoro
