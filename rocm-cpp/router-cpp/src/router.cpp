// 1bit.cpp — router dispatch (impl).
//
// Scaffolding pass: constructs the backend per `RouterOptions::backend`,
// wires a default `GreedySampler`, and routes `forward_token` through
// `Backend::forward_token` → `Sampler::sample`. The `Hip` backend still
// throws `not yet wired`; the `Stub` backend does too so the smoke test
// explicitly asserts on the thrown message.

#include "onebit_cpp/router.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "onebit_cpp/backend.hpp"
#include "onebit_cpp/model.hpp"
#include "onebit_cpp/sampler.hpp"

namespace onebit::cpp {

namespace {

std::unique_ptr<Backend> make_backend(BackendKind kind) {
    switch (kind) {
        case BackendKind::Stub: return make_stub_backend();
        case BackendKind::Hip:  return make_hip_backend();
    }
    throw std::runtime_error("make_backend: unknown BackendKind");
}

}  // namespace

Router::Router() : Router(RouterOptions{}) {}

Router::Router(RouterOptions opts)
    : opts_(opts),
      backend_(make_backend(opts.backend)),
      sampler_(make_greedy()) {}

Router::~Router() = default;

Router::Router(Router&&) noexcept            = default;
Router& Router::operator=(Router&&) noexcept = default;

void Router::load_model(std::string_view h1b_path) {
    // Loader throws on failure; propagate. Attach only after a successful
    // load so a failed load leaves the router unchanged.
    LoadedModel fresh;
    fresh.load(h1b_path);
    if (!backend_) backend_ = make_backend(opts_.backend);
    backend_->attach_model(fresh);
    model_ = std::move(fresh);
}

std::int32_t Router::forward_token(std::int32_t prev_tok, std::int32_t pos) {
    if (!backend_) {
        throw std::runtime_error(
            "Router::forward_token: backend not constructed");
    }
    if (!sampler_) sampler_ = make_greedy();

    // Will throw `not yet wired` from both backends this pass. Router
    // stays correct-shape for when the kernel dispatch lands.
    const auto logits = backend_->forward_token(prev_tok, pos);
    return sampler_->sample(logits);
}

void Router::reset() {
    if (backend_) backend_->reset_kv();
}

void Router::set_sampler(std::unique_ptr<Sampler> s) noexcept {
    sampler_ = s ? std::move(s) : make_greedy();
}

const ModelArgs& Router::model_args() const {
    if (!model_.loaded()) {
        throw std::runtime_error(
            "Router::model_args: no model loaded");
    }
    return model_.args();
}

bool Router::model_loaded() const noexcept { return model_.loaded(); }

}  // namespace onebit::cpp
