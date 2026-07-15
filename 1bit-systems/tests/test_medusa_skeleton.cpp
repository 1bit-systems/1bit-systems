// test_medusa_skeleton.cpp — Medusa kernel-skeleton link + ABI smoke test.
//
// This test exists for the scope-pass (2026-04-24) where the two Medusa
// HIP entry points (rcpp_medusa_tree_attn_decode_fd, rcpp_medusa_small_m_gemv)
// are present at the ABI level but have empty kernel bodies and return
// RCPP_NOT_IMPLEMENTED.
//
// Validates:
//   1. The launchers link from librocm_cpp.so (resolved at process start).
//   2. Calling each launcher with plausible-but-pointer-zero args does not
//      crash — i.e. the host wrapper early-returns before any device-side
//      dereference, as required for the "no kernel body yet" contract.
//   3. Both launchers return RCPP_NOT_IMPLEMENTED so downstream wiring can
//      key on that sentinel and fall back to the M=1 path until the bodies
//      land.
//
// Will be REPLACED by a proper differential test (vs scalar reference)
// once the kernel bodies are authored — at that point this file moves to
// tests/test_medusa_differential.cpp and starts diffing fp16 outputs.
//
// No GPU device required. No model file. Process-exit 0 on success.
//
// No hipBLAS, no Python, no hipDeviceSynchronize.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

#include "rocm_cpp/medusa.h"

namespace {

int g_failures = 0;

void check(bool cond, std::string_view desc) {
    if (cond) {
        std::fprintf(stderr, "  PASS  %.*s\n",
                     static_cast<int>(desc.size()), desc.data());
    } else {
        std::fprintf(stderr, "  FAIL  %.*s\n",
                     static_cast<int>(desc.size()), desc.data());
        ++g_failures;
    }
}

void test_tree_attn_returns_not_implemented() {
    // All-null device pointers + plausible shape numbers. Body is empty so
    // no dereference happens; we only care that the launcher returns the
    // sentinel cleanly. tree_size=8 / num_q_heads=20 / num_kv_heads=4 /
    // head_dim=128 / seq_len=64 mirror BitNet-2B-4T runtime values.
    const rcpp_status_t s =
        rcpp_medusa_tree_attn_decode_fd(/*Q*/ nullptr,
                                        /*K*/ nullptr,
                                        /*V*/ nullptr,
                                        /*tree_mask*/ nullptr,
                                        /*out*/ nullptr,
                                        /*tree_size*/ 8,
                                        /*num_q_heads*/ 20,
                                        /*num_kv_heads*/ 4,
                                        /*head_dim*/ 128,
                                        /*seq_len*/ 64,
                                        /*scale*/ 0.08838834764831845f,
                                        /*stream*/ nullptr);
    check(s == RCPP_NOT_IMPLEMENTED,
          "rcpp_medusa_tree_attn_decode_fd returns RCPP_NOT_IMPLEMENTED");
}

void test_small_m_gemv_returns_not_implemented() {
    // BitNet-2B FFN-down shape: K=6912, N=2560. M=16 = production tree-size.
    const rcpp_status_t s =
        rcpp_medusa_small_m_gemv(/*x_i8*/ nullptr,
                                 /*x_scale*/ 1.0f / 127.0f,
                                 /*w_packed_halo*/ nullptr,
                                 /*w_row_scales*/ nullptr,
                                 /*y_fp16_out*/ nullptr,
                                 /*M*/ 16,
                                 /*N*/ 2560,
                                 /*K*/ 6912,
                                 /*stream*/ nullptr);
    check(s == RCPP_NOT_IMPLEMENTED,
          "rcpp_medusa_small_m_gemv returns RCPP_NOT_IMPLEMENTED");
}

void test_sentinel_distinct_from_other_status_codes() {
    // Belt-and-braces: ensure the sentinel does not collide with the
    // existing rcpp_status_t enumerators. If a future commit reassigns
    // RCPP_NOT_IMPLEMENTED to one of the existing values, downstream
    // code that branches on the sentinel breaks silently — fail loud here.
    check(RCPP_NOT_IMPLEMENTED != RCPP_OK,
          "RCPP_NOT_IMPLEMENTED != RCPP_OK");
    check(RCPP_NOT_IMPLEMENTED != RCPP_INVALID_ARG,
          "RCPP_NOT_IMPLEMENTED != RCPP_INVALID_ARG");
    check(RCPP_NOT_IMPLEMENTED != RCPP_UNSUPPORTED,
          "RCPP_NOT_IMPLEMENTED != RCPP_UNSUPPORTED");
    check(RCPP_NOT_IMPLEMENTED != RCPP_HIP_ERROR,
          "RCPP_NOT_IMPLEMENTED != RCPP_HIP_ERROR");
    check(RCPP_NOT_IMPLEMENTED != RCPP_INTERNAL,
          "RCPP_NOT_IMPLEMENTED != RCPP_INTERNAL");
}

}  // namespace

int main() {
    std::fprintf(stderr, "[medusa skeleton smoke test]\n");
    test_sentinel_distinct_from_other_status_codes();
    test_tree_attn_returns_not_implemented();
    test_small_m_gemv_returns_not_implemented();
    if (g_failures) {
        std::fprintf(stderr,
                     "\n[medusa skeleton] %d failure(s)\n", g_failures);
        return 1;
    }
    std::fprintf(stderr, "\n[medusa skeleton] all passed\n");
    return 0;
}
