// 1bit.cpp — backend implementations.
//
// Two backends live in this TU for now:
//
//   StubBackend — no-op. Every forward path throws `not yet wired`.
//   HipBackend  — factory wired up; forward path still throws until the
//                 next commit lands the per-layer kernel sequence.
//
// The layer schedule `HipBackend::forward_token` will implement is the
// same one `crates/1bit-router/src/backend_impl.rs` runs:
//
//   1. rcpp_embedding_lookup_fp16(prev_tok)             // token embedding
//   2. per layer:
//        rcpp_rmsnorm_fp16 (input_norm)
//        rcpp_quantize_fp16_to_i8                        // activation Q
//        rcpp_ternary_gemv_halo_f16  × Q/K/V             // or Sherry / TQ1 /
//                                                          Bonsai per format
//        rcpp_rope_fp16 (Q, pos), rcpp_rope_fp16 (K, pos)
//        KV append (host-side pointer math — no kernel)
//        rcpp_kv_cache_attn_decode_fd  (split-KV Flash Decoding)
//        rcpp_rmsnorm_fp16 (attn_sub_norm)
//        rcpp_ternary_gemv_halo_f16  × O
//        rcpp_residual_add_fp16
//        rcpp_rmsnorm_fp16 (post_attn_norm)
//        rcpp_ternary_gemv_halo_f16  × gate, × up
//        rcpp_relu2_glu_rmsnorm_fp16 (fused ReLU²·GLU + ffn_sub_norm)
//        rcpp_ternary_gemv_halo_f16  × down
//        rcpp_residual_add_fp16
//   3. rcpp_rmsnorm_fp16 (final_norm)
//   4. rcpp_fp16_gemv (LM head; tied-embedding path uses embedding matrix)
//   5. sampler.sample()                                  // host-side argmax
//
// That wiring is a separate commit — this file keeps the scaffolding
// small so the smoke test can pass on any box (no ROCm required).

#include "onebit_cpp/backend.hpp"

#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include "onebit_cpp/model.hpp"

namespace onebit::cpp {

namespace {

class StubBackend final : public Backend {
public:
    void attach_model(const LoadedModel& /*model*/) override {
        attached_ = true;
    }

    std::span<const float>
    forward_prefill(std::span<const std::int32_t> /*tokens*/,
                    std::int32_t /*start_pos*/) override {
        throw std::runtime_error(
            "StubBackend::forward_prefill: not yet wired");
    }

    std::span<const float>
    forward_token(std::int32_t /*prev_tok*/, std::int32_t /*pos*/) override {
        throw std::runtime_error(
            "StubBackend::forward_token: not yet wired");
    }

    void reset_kv() override { /* no cache, no-op */ }
    BackendKind kind() const noexcept override { return BackendKind::Stub; }
    const char* name() const noexcept override { return "stub"; }

private:
    bool attached_ = false;
};

// HIP backend — scaffold only. Holds a reference to the loaded model so we
// can grow layer-by-layer without changing the public surface. The
// `scratch_logits_` buffer will become a device-backed FP32 logits buffer
// wrapped in `unique_ptr<float[], HipDeleter>` once the forward path
// lands; today it's a std::vector to keep the smoke test host-only.
class HipBackend final : public Backend {
public:
    void attach_model(const LoadedModel& model) override {
        model_args_  = model.args();
        model_ptr_   = model.raw();
        attached_    = (model_ptr_ != nullptr);
        if (attached_) {
            scratch_logits_.assign(
                static_cast<std::size_t>(model_args_.vocab_size), 0.0f);
        }
    }

    std::span<const float>
    forward_prefill(std::span<const std::int32_t> /*tokens*/,
                    std::int32_t /*start_pos*/) override {
        throw std::runtime_error(
            "HipBackend::forward_prefill: not yet wired — scheduled for the "
            "next 1bit.cpp commit (per-layer kernel dispatch "
            "mirroring crates/1bit-router/src/backend_impl.rs)");
    }

    std::span<const float>
    forward_token(std::int32_t /*prev_tok*/, std::int32_t /*pos*/) override {
        throw std::runtime_error(
            "HipBackend::forward_token: not yet wired — scheduled for the "
            "next 1bit.cpp commit (per-layer kernel dispatch "
            "mirroring crates/1bit-router/src/backend_impl.rs)");
    }

    void reset_kv() override {
        // Kernel-level KV reset is a pointer memset — not yet plumbed.
        // Tracked alongside forward_token wiring.
    }

    BackendKind kind() const noexcept override { return BackendKind::Hip; }
    const char* name() const noexcept override { return "hip"; }

private:
    ModelArgs                    model_args_{};
    const ::rcpp_bitnet_model_t* model_ptr_ = nullptr;
    bool                         attached_ = false;
    std::vector<float>           scratch_logits_{};
};

}  // namespace

std::unique_ptr<Backend> make_stub_backend() {
    return std::make_unique<StubBackend>();
}

std::unique_ptr<Backend> make_hip_backend() {
    return std::make_unique<HipBackend>();
}

}  // namespace onebit::cpp
