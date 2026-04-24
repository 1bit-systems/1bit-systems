// 1bit.cpp — loaded-model state (impl).
//
// Thin RAII wrapper over `rcpp_bitnet_load_h1b` / `rcpp_bitnet_free` from
// `rocm-cpp/include/rocm_cpp/bitnet_model.h`. The kernels live there; we
// just own the lifetime.

#include "onebit_cpp/model.hpp"

#include <stdexcept>
#include <string>
#include <string_view>

#include "rocm_cpp/bitnet_model.h"
#include "rocm_cpp/ck_gemm.h"  // rcpp_status_t

namespace onebit::cpp {

namespace {

ModelArgs to_args(const ::rcpp_bitnet_model_t& m) noexcept {
    ModelArgs a{};
    a.hidden_size       = m.hidden_size;
    a.intermediate_size = m.intermediate_size;
    a.num_layers        = m.num_layers;
    a.num_heads         = m.num_heads;
    a.num_kv_heads      = m.num_kv_heads;
    a.vocab_size        = m.vocab_size;
    a.max_seq_len       = m.max_seq_len;
    a.tie_embeddings    = m.tie_embeddings;
    a.rope_theta        = m.rope_theta;
    a.rms_norm_eps      = m.rms_norm_eps;
    a.format_version    = m.format_version;
    a.flags             = m.flags;
    a.is_qwen3          = m.is_qwen3;
    a.weight_format     = static_cast<int>(m.weight_format);
    a.arch              = static_cast<int>(m.arch);
    return a;
}

}  // namespace

LoadedModel::~LoadedModel() { reset(); }

LoadedModel::LoadedModel(LoadedModel&& other) noexcept
    : model_(other.model_),
      args_(other.args_),
      path_(std::move(other.path_)),
      loaded_(other.loaded_) {
    other.model_  = {};
    other.args_   = {};
    other.loaded_ = false;
}

LoadedModel& LoadedModel::operator=(LoadedModel&& other) noexcept {
    if (this != &other) {
        reset();
        model_  = other.model_;
        args_   = other.args_;
        path_   = std::move(other.path_);
        loaded_ = other.loaded_;
        other.model_  = {};
        other.args_   = {};
        other.loaded_ = false;
    }
    return *this;
}

void LoadedModel::reset() noexcept {
    if (loaded_) {
        ::rcpp_bitnet_free(&model_);
        loaded_ = false;
    }
    model_ = {};
    args_  = {};
    path_.clear();
}

void LoadedModel::load(std::string_view path) {
    reset();
    path_.assign(path);

    // The rcpp loader takes a C string. `std::string::c_str()` is
    // null-terminated by contract.
    const ::rcpp_status_t st =
        ::rcpp_bitnet_load_h1b(path_.c_str(), &model_);
    if (st != ::RCPP_OK) {
        // Scrub any partial state the loader might have left behind.
        ::rcpp_bitnet_free(&model_);
        model_ = {};
        throw std::runtime_error(
            "LoadedModel::load: rcpp_bitnet_load_h1b failed with status "
            + std::to_string(static_cast<int>(st)) + " on path " + path_);
    }
    args_   = to_args(model_);
    loaded_ = true;
}

}  // namespace onebit::cpp
