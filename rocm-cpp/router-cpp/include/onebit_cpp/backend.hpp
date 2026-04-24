// 1bit.cpp — backend interface (kernel dispatch contract).
//
// The router is backend-agnostic: it drives a forward token one layer at
// a time, and the `Backend` vtable turns each step into concrete kernel
// launches. The two implementations this pass scaffolds:
//
//   StubBackend — no GPU; every `forward_token` throws `not yet wired`.
//                 Lets the smoke test + CI build without a ROCm device.
//                 Also used as the default when `HALO_BACKEND=stub`.
//   HipBackend  — wraps the rocm-cpp ternary GEMV + split-KV FD attention
//                 + RMSNorm kernels. Forward-token is still stubbed this
//                 pass; the kernel wiring lands in the next commit.
//
// The Rust source of truth is `crates/1bit-router/src/backend_impl.rs`.
// The layer-by-layer sequence the C++ backend will need to mirror:
//   RMSNorm → quantize → QKV → RoPE → KV append → split-KV FD attn
//   → RMSNorm → O proj → FFN gate+up → FFN down → LM head → argmax.

#ifndef ONEBIT_CPP_BACKEND_HPP
#define ONEBIT_CPP_BACKEND_HPP

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#include "onebit_cpp/model.hpp"

namespace onebit::cpp {

enum class BackendKind {
    Stub,
    Hip,
};

// Abstract backend. All `forward_*` methods throw `std::runtime_error` on
// failure (HIP launch error, invalid state, etc). The router catches and
// translates into its own error type once the `Result`-style return plumbing
// lands; this pass keeps the exception path to stay close to the Rust
// `Result<_, BackendError>` surface without forcing a C++23 `std::expected`.
class Backend {
public:
    virtual ~Backend() = default;

    // One-shot model attach. Idempotent; calling twice is UB (caller
    // should construct a new backend instead).
    virtual void attach_model(const LoadedModel& model) = 0;

    // Prefill `tokens` into the KV cache starting at position `start_pos`.
    // Bulk entry for prompt ingestion; per-token decode uses
    // `forward_token`. Returns the logits for the LAST token in the batch
    // (the only one the router samples from at prefill end).
    //
    // Stubbed this pass.
    virtual std::span<const float>
    forward_prefill(std::span<const std::int32_t> tokens, std::int32_t start_pos) = 0;

    // One decode step. `prev_tok` is the previously-sampled token id;
    // `pos` is the KV-cache write position. Returns logits over the
    // vocabulary; the view is owned by the backend and remains valid
    // until the next `forward_*` call.
    //
    // Stubbed this pass; follow-up commit wires the rocm-cpp kernels
    // per the layer sequence listed in the module header.
    virtual std::span<const float>
    forward_token(std::int32_t prev_tok, std::int32_t pos) = 0;

    // Reset the KV cache (session boundary). The Rust side used to leak
    // stale KV state at completion N≈200 — see
    // `project_bitnet_live_bench.md`. Mirror the explicit reset here.
    virtual void reset_kv() = 0;

    virtual BackendKind kind() const noexcept = 0;
    virtual const char* name() const noexcept = 0;
};

// Stub backend — no GPU touched. Every `forward_*` throws
// `std::runtime_error("not yet wired")`. Useful for host-only CI + the
// smoke test.
std::unique_ptr<Backend> make_stub_backend();

// HIP backend — wraps rocm-cpp. `forward_*` throws `not yet wired` this
// pass; layer-by-layer wiring arrives in the next commit per the roadmap.
std::unique_ptr<Backend> make_hip_backend();

}  // namespace onebit::cpp

#endif  // ONEBIT_CPP_BACKEND_HPP
