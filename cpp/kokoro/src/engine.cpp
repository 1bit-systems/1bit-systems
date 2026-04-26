// engine.cpp — Engine lifecycle + synthesize().
//
// When ONEBIT_KOKORO_HAVE_HALO == 0 every entry point returns
// UnsupportedStub. When the upstream halo-kokoro target is linked,
// Impl wraps its engine handle and dispatches.
//
// We deliberately don't include halo-kokoro's header in the public
// surface — pImpl keeps onebit::kokoro::Engine ABI-stable across
// halo-kokoro upstream churn.

#include "onebit/kokoro/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef ONEBIT_KOKORO_HAVE_HALO
#define ONEBIT_KOKORO_HAVE_HALO 0
#endif

#if ONEBIT_KOKORO_HAVE_HALO
// halo-kokoro upstream is single-header. The actual header name on
// disk varies across versions ("kokoro.hpp", "halo_kokoro.hpp",
// "halo-kokoro/engine.hpp"). We probe via an indirect include set so
// the build adapts.
#  if __has_include(<halo_kokoro/engine.hpp>)
#    include <halo_kokoro/engine.hpp>
#  elif __has_include(<halo-kokoro/engine.hpp>)
#    include <halo-kokoro/engine.hpp>
#  elif __has_include("halo_kokoro.hpp")
#    include "halo_kokoro.hpp"
#  elif __has_include("kokoro.hpp")
#    include "kokoro.hpp"
#  else
#    warning "ONEBIT_KOKORO_HAVE_HALO=1 but no halo-kokoro header on include path; reverting to stub"
#    undef  ONEBIT_KOKORO_HAVE_HALO
#    define ONEBIT_KOKORO_HAVE_HALO 0
#  endif
#endif

namespace onebit::kokoro {

namespace detail {

std::expected<void, Error> validate_text(std::string_view text) noexcept {
    if (std::all_of(text.begin(), text.end(), [](char c) {
            // Whitespace per the Rust crate's str::trim semantics:
            // ASCII space, tab, LF, CR, FF, VT.
            return c == ' ' || c == '\t' || c == '\n' ||
                   c == '\r' || c == '\f' || c == '\v';
        })) {
        return std::unexpected(Error{ErrorKind::InvalidText,
                                     "text is empty or whitespace-only"});
    }
    if (text.find('\0') != std::string_view::npos) {
        return std::unexpected(Error{ErrorKind::InvalidPath,
                                     "text contains interior NUL"});
    }
    return {};
}

std::expected<void, Error> validate_voice(std::string_view voice) noexcept {
    if (voice.empty()) {
        return std::unexpected(Error{ErrorKind::InvalidVoice, "voice id empty"});
    }
    if (voice.find('\0') != std::string_view::npos) {
        return std::unexpected(Error{ErrorKind::InvalidPath,
                                     "voice contains interior NUL"});
    }
    return {};
}

std::expected<void, Error> validate_speed(float speed) noexcept {
    if (!std::isfinite(speed) || speed <= kSpeedMin || speed > kSpeedMax) {
        return std::unexpected(Error{
            ErrorKind::InvalidSpeed,
            "speed out of supported range (0, 4]",
            speed});
    }
    return {};
}

} // namespace detail

#if ONEBIT_KOKORO_HAVE_HALO
struct Engine::Impl {
    // The halo-kokoro engine type / handle name varies across upstream
    // versions. We assume the canonical "halo_kokoro::Engine" name; if
    // the upstream uses a different name, swap here. The Rust crate's
    // shim used a `KokoroCtx*` opaque pointer; the official C++ port
    // exposes a class.
    halo_kokoro::Engine engine;

    explicit Impl(const std::filesystem::path& model)
        : engine{halo_kokoro::Engine::load(model.string())} {}
};
#else
struct Engine::Impl {};
#endif

Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;
Engine::~Engine()                            = default;

bool runtime_available() noexcept {
    return ONEBIT_KOKORO_HAVE_HALO != 0;
}

bool Engine::has_runtime() const noexcept {
#if ONEBIT_KOKORO_HAVE_HALO
    return impl_ != nullptr;
#else
    return false;
#endif
}

std::expected<Engine, Error>
Engine::create(const std::filesystem::path& model) {
#if !ONEBIT_KOKORO_HAVE_HALO
    (void)model;
    return std::unexpected(Error{
        ErrorKind::UnsupportedStub,
        "build did not link halo-kokoro; set HALO_KOKORO_DIR=<path> and rebuild"});
#else
    if (!std::filesystem::exists(model)) {
        return std::unexpected(Error{
            ErrorKind::ModelLoadFailed,
            "model file does not exist: " + model.string()});
    }
    try {
        Engine e;
        e.impl_ = std::make_unique<Impl>(model);
        return e;
    } catch (const std::exception& ex) {
        return std::unexpected(Error{ErrorKind::ModelLoadFailed, ex.what()});
    } catch (...) {
        return std::unexpected(Error{ErrorKind::ModelLoadFailed,
                                     "unknown halo-kokoro load failure"});
    }
#endif
}

std::expected<SynthesisOutput, Error>
Engine::synthesize(std::string_view text,
                   std::string_view voice,
                   float            speed) {
    if (auto v = detail::validate_text(text); !v)  return std::unexpected(v.error());
    if (auto v = detail::validate_voice(voice); !v) return std::unexpected(v.error());
    if (auto v = detail::validate_speed(speed); !v) return std::unexpected(v.error());

#if !ONEBIT_KOKORO_HAVE_HALO
    return std::unexpected(Error{
        ErrorKind::UnsupportedStub,
        "synthesize: build did not link halo-kokoro"});
#else
    if (!impl_) {
        return std::unexpected(Error{ErrorKind::UnsupportedStub,
                                     "engine has no halo-kokoro impl"});
    }
    try {
        // halo-kokoro returns std::vector<int16_t> at 22050 Hz mono.
        // Names assumed; adapt to upstream when wiring lands.
        auto pcm = impl_->engine.synthesize(std::string{text},
                                            std::string{voice},
                                            speed);
        SynthesisOutput out;
        out.pcm = std::move(pcm);
        out.info.sample_rate = 22'050;
        out.info.channels    = 1;
        out.info.samples     = out.pcm.size();
        out.info.duration_ms =
            (out.pcm.size() * 1000ull) / out.info.sample_rate;
        return out;
    } catch (const std::exception& ex) {
        return std::unexpected(Error{ErrorKind::ShimError, ex.what(), -1});
    } catch (...) {
        return std::unexpected(Error{ErrorKind::ShimError,
                                     "unknown halo-kokoro error", -1});
    }
#endif
}

} // namespace onebit::kokoro
