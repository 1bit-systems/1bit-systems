// error.cpp — what()/label() implementations for onebit::onnx::Error.

#include "onebit/onnx/error.hpp"

#include <string>

namespace onebit::onnx {

std::string_view label(ErrorKind kind) noexcept {
    switch (kind) {
        case ErrorKind::MissingArtifact:       return "missing_artifact";
        case ErrorKind::NotAnArtifactDir:      return "not_an_artifact_dir";
        case ErrorKind::InvalidGenAiConfig:    return "invalid_genai_config";
        case ErrorKind::OrtRuntimeUnavailable: return "ort_runtime_unavailable";
        case ErrorKind::VitisaiUnavailable:    return "vitisai_unavailable";
        case ErrorKind::SessionInit:           return "session_init";
        case ErrorKind::TokenizerLoad:         return "tokenizer_load";
        case ErrorKind::Io:                    return "io";
    }
    return "unknown";
}

std::string Error::what() const {
    std::string out{label(kind_)};
    if (!path_.empty()) {
        out += ": ";
        out += path_.string();
    }
    if (!detail_.empty()) {
        out += ": ";
        out += detail_;
    }
    return out;
}

} // namespace onebit::onnx
