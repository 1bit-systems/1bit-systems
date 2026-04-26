// session.cpp — Session::load{,_config_only} and the pImpl hosting
// Ort::Env + Ort::Session.
//
// When ONEBIT_ONNX_HAVE_ORT == 0 the Impl struct is an empty stand-in
// and `load()` returns OrtRuntimeUnavailable. `load_config_only()` is
// always available — it only touches the filesystem.

#include "onebit/onnx/session.hpp"

#ifndef ONEBIT_ONNX_HAVE_ORT
#define ONEBIT_ONNX_HAVE_ORT 0
#endif

#if ONEBIT_ONNX_HAVE_ORT
#include <onnxruntime_cxx_api.h>
#endif

#include <fstream>
#include <utility>

namespace onebit::onnx {

#if ONEBIT_ONNX_HAVE_ORT
struct Session::Impl {
    // Ort::Env is logically a singleton — multiple instances are tolerated
    // but waste threads. We keep one per Session because the hot-path is
    // a small number of long-lived sessions, and tying the env's lifetime
    // to the session keeps RAII tidy.
    Ort::Env     env{ORT_LOGGING_LEVEL_WARNING, "onebit-onnx"};
    Ort::Session session{nullptr};
};
#else
struct Session::Impl {};
#endif

Session::Session(Session&&) noexcept            = default;
Session& Session::operator=(Session&&) noexcept = default;
Session::~Session()                             = default;

bool Session::runtime_available() noexcept {
    return ONEBIT_ONNX_HAVE_ORT != 0;
}

bool Session::has_runtime() const noexcept {
#if ONEBIT_ONNX_HAVE_ORT
    return impl_ != nullptr;
#else
    return false;
#endif
}

std::expected<Session, Error>
Session::load_config_only(const std::filesystem::path& root) {
    auto paths = ArtifactPaths::discover(root);
    if (!paths) return std::unexpected(paths.error());

    auto cfg = GenAiConfig::load(paths->genai_config);
    if (!cfg) return std::unexpected(cfg.error());

    Session s;
    s.paths_  = std::move(*paths);
    s.config_ = std::move(*cfg);
    s.lane_   = ExecutionLane::Cpu;
    s.impl_   = nullptr;            // explicit: no live ORT session
    return s;
}

#if ONEBIT_ONNX_HAVE_ORT
namespace {
// Try to register the VitisAI EP. Returns true on success. We attempt
// the AppendExecutionProvider name-based registration — concrete
// constructors vary across ORT versions, but the named lookup is stable.
[[nodiscard]] bool try_register_vitisai(Ort::SessionOptions& opts) {
    try {
        // Empty option map; AMD's loader picks defaults from the env.
        opts.AppendExecutionProvider("VitisAI", /* provider_options = */ {});
        return true;
    } catch (const Ort::Exception&) {
        return false;
    }
}
} // namespace
#endif

std::expected<Session, Error>
Session::load(const std::filesystem::path& root) {
#if !ONEBIT_ONNX_HAVE_ORT
    (void)root;
    return std::unexpected(Error{
        ErrorKind::OrtRuntimeUnavailable,
        std::string{"this build was not linked against libonnxruntime — install onnxruntime "
                    "and reconfigure CMake to enable Session::load"}});
#else
    auto base = load_config_only(root);
    if (!base) return std::unexpected(base.error());

    Session s = std::move(*base);
    auto impl = std::make_unique<Impl>();

    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    bool got_vitisai = try_register_vitisai(opts);

    try {
        impl->session = Ort::Session(impl->env,
                                     s.paths_.model.c_str(),
                                     opts);
    } catch (const Ort::Exception& e) {
        // VitisAI register may have succeeded but the actual session
        // build failed; retry on plain CPU EP before giving up.
        if (got_vitisai) {
            try {
                Ort::SessionOptions cpu_opts;
                cpu_opts.SetIntraOpNumThreads(1);
                cpu_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                impl->session = Ort::Session(impl->env,
                                             s.paths_.model.c_str(),
                                             cpu_opts);
                got_vitisai = false;
            } catch (const Ort::Exception& e2) {
                return std::unexpected(Error{ErrorKind::SessionInit, e2.what()});
            }
        } else {
            return std::unexpected(Error{ErrorKind::SessionInit, e.what()});
        }
    }

    s.lane_ = got_vitisai ? ExecutionLane::Vitisai : ExecutionLane::Cpu;
    s.impl_ = std::move(impl);
    return s;
#endif
}

} // namespace onebit::onnx
