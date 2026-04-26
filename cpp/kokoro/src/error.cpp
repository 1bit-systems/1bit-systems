// error.cpp — labels + what() for onebit::kokoro::Error.

#include "onebit/kokoro/error.hpp"

namespace onebit::kokoro {

std::string_view label(ErrorKind k) noexcept {
    switch (k) {
        case ErrorKind::UnsupportedStub: return "unsupported_stub";
        case ErrorKind::ModelLoadFailed: return "model_load_failed";
        case ErrorKind::VoiceNotFound:   return "voice_not_found";
        case ErrorKind::InvalidText:     return "invalid_text";
        case ErrorKind::InvalidVoice:    return "invalid_voice";
        case ErrorKind::InvalidSpeed:    return "invalid_speed";
        case ErrorKind::ShimError:       return "shim_error";
        case ErrorKind::InvalidPath:     return "invalid_path";
    }
    return "unknown";
}

std::string Error::what() const {
    std::string out{label(kind_)};
    if (!detail_.empty()) {
        out += ": ";
        out += detail_;
    }
    return out;
}

} // namespace onebit::kokoro
