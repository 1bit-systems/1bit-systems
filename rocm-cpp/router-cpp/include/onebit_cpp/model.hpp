// 1bit.cpp — loaded-model state.
//
// Owns the `.h1b` mmap + device-upload handles produced by
// `rocm-cpp/src/h1b_loader.cpp` (which already exposes a C ABI via
// `rcpp_bitnet_load_h1b`). Wraps the raw C struct in an RAII holder so the
// router can return early / throw without leaking GPU buffers.
//
// Rule A: no Python. Rule B: kernels stay in rocm-cpp — we just bind to
// their C entry points. Rule C: no hipBLAS touched here.

#ifndef ONEBIT_CPP_MODEL_HPP
#define ONEBIT_CPP_MODEL_HPP

#include <cstdint>
#include <string>
#include <string_view>

#include "rocm_cpp/bitnet_model.h"

namespace onebit::cpp {

// Lightweight copy of the architectural knobs the router needs. Lets us
// keep `rcpp_bitnet_model_t` private to the .cpp and keep the headers
// free of the full C struct.
struct ModelArgs {
    int hidden_size      = 0;
    int intermediate_size = 0;
    int num_layers       = 0;
    int num_heads        = 0;
    int num_kv_heads     = 0;
    int vocab_size       = 0;
    int max_seq_len      = 0;
    int tie_embeddings   = 0;

    float rope_theta    = 0.0f;
    float rms_norm_eps  = 0.0f;

    int format_version   = 0;
    unsigned int flags   = 0;
    int is_qwen3         = 0;

    // Weight-format dispatch tag, cast from `rcpp_weight_format_t` at load.
    int weight_format    = 0;
    // Model arch tag, cast from `rcpp_arch_t` at load.
    int arch             = 0;
};

// RAII owner for a loaded `.h1b`. Holds the rcpp C struct by value, plus the
// source path for logs. Move-only.
class LoadedModel {
public:
    LoadedModel() noexcept = default;
    ~LoadedModel();

    LoadedModel(const LoadedModel&)            = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;

    LoadedModel(LoadedModel&& other) noexcept;
    LoadedModel& operator=(LoadedModel&& other) noexcept;

    // Load `path` via `rcpp_bitnet_load_h1b`. Throws `std::runtime_error`
    // on loader failure; the rocm-cpp loader returns a status code we
    // translate into the string.
    void load(std::string_view path);

    // Release the model + reset internal state. Safe to call twice.
    void reset() noexcept;

    bool loaded() const noexcept { return loaded_; }
    const ModelArgs& args() const noexcept { return args_; }
    const std::string& path() const noexcept { return path_; }

    // Raw handle for the backend to feed into the rcpp_* kernel launchers.
    // Returns nullptr when `!loaded()`.
    ::rcpp_bitnet_model_t* raw() noexcept { return loaded_ ? &model_ : nullptr; }
    const ::rcpp_bitnet_model_t* raw() const noexcept { return loaded_ ? &model_ : nullptr; }

private:
    // C struct; zero-init is fine pre-load.
    ::rcpp_bitnet_model_t model_{};
    ModelArgs             args_{};
    std::string           path_{};
    bool                  loaded_ = false;
};

}  // namespace onebit::cpp

#endif  // ONEBIT_CPP_MODEL_HPP
