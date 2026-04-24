// 1bit.cpp — router public C++ API.
//
// Phase 2 flagship port: a direct C++20 mirror of `crates/1bit-router/`
// built to kill the HTTP-overhead leak (sampler-pipe IPC + async runtime)
// measured at 50-80% of kernel tok/s.
//
// This header is the only public surface an embedder (1bit-server-cpp,
// 1bit-helm, future 1bit-cpp umbrella) needs to include.
//
// Threading model: `Router` is single-request. The eventual parallel lane
// instantiates one `Router` per worker, sharing the underlying
// `LoadedModel` + `Backend` via `std::shared_ptr`. This pass only wires
// the single-request path.
//
// Rule A (no Python) — enforced. Rule B (kernels stay in rocm-cpp) —
// enforced; we link `rocm_cpp`, do not duplicate kernels. Rule C
// (no hipBLAS) — enforced; the rcpp entry points we call are
// native-Tensile-only.

#ifndef ONEBIT_CPP_ROUTER_HPP
#define ONEBIT_CPP_ROUTER_HPP

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "onebit_cpp/backend.hpp"
#include "onebit_cpp/chat_template.hpp"
#include "onebit_cpp/model.hpp"
#include "onebit_cpp/sampler.hpp"

namespace onebit::cpp {

// Router construction options. Kept intentionally small this pass; fields
// will grow as the Rust feature parity lands (KV dtype, sampler mode, etc).
struct RouterOptions {
    // Which backend to build in `load_model`. Default `Hip`.
    BackendKind backend = BackendKind::Hip;

    // Which chat-template to apply in the server wrapper. Default `Llama3`;
    // `Short` saves ~9 prefill tokens on single-turn chat requests.
    ChatTemplate chat_template = ChatTemplate::Llama3;
};

// Router — owns model + backend + sampler. Single-request. Constructed
// on the serving thread.
class Router {
public:
    Router();
    explicit Router(RouterOptions opts);
    ~Router();

    Router(const Router&)            = delete;
    Router& operator=(const Router&) = delete;
    Router(Router&&) noexcept;
    Router& operator=(Router&&) noexcept;

    // Load a `.h1b` model and attach it to the backend. Throws
    // `std::runtime_error` on loader or attach failure. Thread-safe only
    // before any `forward_*` call lands on this router.
    void load_model(std::string_view h1b_path);

    // Single decode step. `prev_tok` is the previously-emitted token;
    // `pos` is the cache-write position. Returns the newly-sampled token
    // id by routing logits through the currently-selected sampler.
    //
    // Stubbed this pass: throws `std::runtime_error("not yet wired")` if
    // the backend is the `Hip` one (forward path not wired), returns a
    // deterministic fixed token id otherwise so the smoke test can
    // exercise the call shape without GPU hardware.
    std::int32_t forward_token(std::int32_t prev_tok, std::int32_t pos);

    // Reset the KV cache between requests. Forwards to
    // `Backend::reset_kv`. Safe to call even when no backend is attached
    // (no-op).
    void reset();

    // Swap out the sampler for subsequent `forward_token` calls. The
    // router takes ownership. Passing nullptr restores the default
    // `GreedySampler`.
    void set_sampler(std::unique_ptr<Sampler> s) noexcept;

    // Accessors. Useful for tests + the HTTP wrapper.
    const RouterOptions& options() const noexcept { return opts_; }
    const ModelArgs&     model_args() const;
    bool                 model_loaded() const noexcept;

private:
    RouterOptions              opts_{};
    LoadedModel                model_{};
    std::unique_ptr<Backend>   backend_{};
    std::unique_ptr<Sampler>   sampler_{};
};

}  // namespace onebit::cpp

#endif  // ONEBIT_CPP_ROUTER_HPP
