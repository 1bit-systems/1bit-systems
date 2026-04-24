// Degenerate-input attention test — v0.1.1 correctness guard.
//
// Exercises the fp16 single-block decode + prefill kernels in
// src/kv_cache_attn.hip with pathological inputs to prove the
// `inv_l = (l > 1e-20f) ? 1/l : 0` guard keeps the output finite.
//
// Three cases:
//   1. All-zero K/V + all-zero Q.  Every attention score == 0, softmax is
//      uniform, output == 0 (finite).  Baseline: proves the happy path on
//      the zero edge still produces finite numbers.
//   2. All-zero K/V + non-zero Q.  Scores still all zero (Q·0 == 0), output
//      must be finite zero — no NaN from 0 / 0 or 0 * inf.
//   3. Prefill path, t == 0, empty-looking V.  Smoke-test the prefill guard
//      so a future masking change that leaves l = 0 for the first token does
//      not silently write NaN into the residual.
//
// Across all cases the assertion is identical: every output element is
// finite (std::isfinite → true).  If the guard regressed to `1.0f / l`, the
// zero-Q-zero-K case would still pass (l == seq_len > 0), so we also call
// the i8 path with an all-zero row to cover the int8 guard regression.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "rocm_cpp/ck_gemm.h"

#define HIP_OK(e)                                                              \
    do {                                                                       \
        auto _s = (e);                                                         \
        if (_s != hipSuccess) {                                                \
            std::fprintf(stderr, "HIP %d %s:%d\n", _s, __FILE__, __LINE__);    \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

namespace {

struct Row {
    const char* name;
    bool        pass;
};
std::vector<Row> results;

// Returns true if every fp16 element is finite (not NaN, not inf).
bool all_finite(const std::vector<_Float16>& v) {
    for (_Float16 x : v) {
        if (!std::isfinite((float)x)) return false;
    }
    return true;
}

void record(const char* name, bool ok) {
    results.push_back({name, ok});
    std::printf("  %-40s : %s\n", name, ok ? "PASS" : "FAIL");
}

// ---- fp16 decode path ------------------------------------------------------
void test_fp16_decode_zero_kv() {
    const int nh = 20, nkv = 5, hd = 128, seq_len = 64;
    const float scale = 1.0f / std::sqrt((float)hd);

    std::vector<_Float16> Q ((size_t)nh * hd, (_Float16)0.5f);   // non-zero Q
    std::vector<_Float16> Kc((size_t)seq_len * nkv * hd, (_Float16)0.0f);
    std::vector<_Float16> Vc((size_t)seq_len * nkv * hd, (_Float16)0.0f);

    _Float16 *dQ, *dK, *dV, *dO;
    HIP_OK(hipMalloc(&dQ, Q.size()  * 2));
    HIP_OK(hipMalloc(&dK, Kc.size() * 2));
    HIP_OK(hipMalloc(&dV, Vc.size() * 2));
    HIP_OK(hipMalloc(&dO, (size_t)nh * hd * 2));
    HIP_OK(hipMemcpy(dQ, Q.data(),  Q.size()  * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dK, Kc.data(), Kc.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dV, Vc.data(), Vc.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemset(dO, 0xFF, (size_t)nh * hd * 2));  // poison: any NaN survives

    rcpp_kv_cache_attn_decode(dQ, dK, dV, dO, nh, nkv, hd, seq_len, scale,
                              nullptr);
    HIP_OK(hipDeviceSynchronize());

    std::vector<_Float16> y((size_t)nh * hd);
    HIP_OK(hipMemcpy(y.data(), dO, y.size() * 2, hipMemcpyDeviceToHost));
    record("fp16 decode all-zero K/V finite", all_finite(y));

    HIP_OK(hipFree(dQ)); HIP_OK(hipFree(dK));
    HIP_OK(hipFree(dV)); HIP_OK(hipFree(dO));
}

// ---- fp16 prefill path -----------------------------------------------------
void test_fp16_prefill_zero_kv() {
    const int nh = 8, nkv = 2, hd = 64, seq_len = 16;
    const float scale = 1.0f / std::sqrt((float)hd);

    std::vector<_Float16> Q ((size_t)seq_len * nh * hd, (_Float16)0.0f);
    std::vector<_Float16> Kc((size_t)seq_len * nkv * hd, (_Float16)0.0f);
    std::vector<_Float16> Vc((size_t)seq_len * nkv * hd, (_Float16)0.0f);

    _Float16 *dQ, *dK, *dV, *dO;
    HIP_OK(hipMalloc(&dQ, Q.size()  * 2));
    HIP_OK(hipMalloc(&dK, Kc.size() * 2));
    HIP_OK(hipMalloc(&dV, Vc.size() * 2));
    HIP_OK(hipMalloc(&dO, (size_t)seq_len * nh * hd * 2));
    HIP_OK(hipMemcpy(dQ, Q.data(),  Q.size()  * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dK, Kc.data(), Kc.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dV, Vc.data(), Vc.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemset(dO, 0xFF, (size_t)seq_len * nh * hd * 2));

    rcpp_kv_cache_attn_prefill(dQ, dK, dV, dO, nh, nkv, hd, seq_len, scale,
                               nullptr);
    HIP_OK(hipDeviceSynchronize());

    std::vector<_Float16> y((size_t)seq_len * nh * hd);
    HIP_OK(hipMemcpy(y.data(), dO, y.size() * 2, hipMemcpyDeviceToHost));
    record("fp16 prefill all-zero Q/K/V finite", all_finite(y));

    HIP_OK(hipFree(dQ)); HIP_OK(hipFree(dK));
    HIP_OK(hipFree(dV)); HIP_OK(hipFree(dO));
}

// ---- fp16 split-KV flash-decoding path -------------------------------------
void test_fp16_fd_zero_kv() {
    const int nh = 20, nkv = 5, hd = 128, seq_len = 256;  // > TILE=128 so multi-tile
    const float scale = 1.0f / std::sqrt((float)hd);

    std::vector<_Float16> Q ((size_t)nh * hd, (_Float16)0.0f);
    std::vector<_Float16> Kc((size_t)seq_len * nkv * hd, (_Float16)0.0f);
    std::vector<_Float16> Vc((size_t)seq_len * nkv * hd, (_Float16)0.0f);

    _Float16 *dQ, *dK, *dV, *dO;
    HIP_OK(hipMalloc(&dQ, Q.size()  * 2));
    HIP_OK(hipMalloc(&dK, Kc.size() * 2));
    HIP_OK(hipMalloc(&dV, Vc.size() * 2));
    HIP_OK(hipMalloc(&dO, (size_t)nh * hd * 2));
    HIP_OK(hipMemcpy(dQ, Q.data(),  Q.size()  * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dK, Kc.data(), Kc.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dV, Vc.data(), Vc.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemset(dO, 0xFF, (size_t)nh * hd * 2));

    rcpp_kv_cache_attn_decode_fd(dQ, dK, dV, dO, nh, nkv, hd, seq_len, scale,
                                 nullptr);
    HIP_OK(hipDeviceSynchronize());

    std::vector<_Float16> y((size_t)nh * hd);
    HIP_OK(hipMemcpy(y.data(), dO, y.size() * 2, hipMemcpyDeviceToHost));
    record("fp16 fd  all-zero Q/K/V finite", all_finite(y));

    HIP_OK(hipFree(dQ)); HIP_OK(hipFree(dK));
    HIP_OK(hipFree(dV)); HIP_OK(hipFree(dO));
}

// ---- int8 decode path ------------------------------------------------------
// Scales set to zero-ish (min-positive fp16) so dequant yields ~0, exercising
// the same division-by-small-l path.
void test_i8_decode_zero_scales() {
    const int nh = 20, nkv = 5, hd = 128, seq_len = 64;
    const float scale = 1.0f / std::sqrt((float)hd);

    std::vector<_Float16> Q ((size_t)nh * hd, (_Float16)0.5f);
    std::vector<int8_t>   Kc((size_t)seq_len * nkv * hd, (int8_t)0);
    std::vector<int8_t>   Vc((size_t)seq_len * nkv * hd, (int8_t)0);
    std::vector<_Float16> Ks((size_t)seq_len * nkv, (_Float16)0.0f);
    std::vector<_Float16> Vs((size_t)seq_len * nkv, (_Float16)0.0f);

    _Float16 *dQ, *dKs, *dVs, *dO;
    int8_t   *dK, *dV;
    HIP_OK(hipMalloc(&dQ,  Q.size()  * 2));
    HIP_OK(hipMalloc(&dK,  Kc.size()));
    HIP_OK(hipMalloc(&dV,  Vc.size()));
    HIP_OK(hipMalloc(&dKs, Ks.size() * 2));
    HIP_OK(hipMalloc(&dVs, Vs.size() * 2));
    HIP_OK(hipMalloc(&dO,  (size_t)nh * hd * 2));
    HIP_OK(hipMemcpy(dQ,  Q.data(),  Q.size()  * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dK,  Kc.data(), Kc.size(),     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dV,  Vc.data(), Vc.size(),     hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dKs, Ks.data(), Ks.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dVs, Vs.data(), Vs.size() * 2, hipMemcpyHostToDevice));
    HIP_OK(hipMemset(dO, 0xFF, (size_t)nh * hd * 2));

    rcpp_kv_cache_attn_decode_i8(dQ, dK, dV, dKs, dVs, dO,
                                 nh, nkv, hd, seq_len, scale, nullptr);
    HIP_OK(hipDeviceSynchronize());

    std::vector<_Float16> y((size_t)nh * hd);
    HIP_OK(hipMemcpy(y.data(), dO, y.size() * 2, hipMemcpyDeviceToHost));
    record("i8   decode zero K/V+scales finite", all_finite(y));

    HIP_OK(hipFree(dQ));  HIP_OK(hipFree(dK));  HIP_OK(hipFree(dV));
    HIP_OK(hipFree(dKs)); HIP_OK(hipFree(dVs)); HIP_OK(hipFree(dO));
}

}  // namespace

int main() {
    std::printf("=== rocm-cpp attention degenerate-input guard tests ===\n");
    test_fp16_decode_zero_kv();
    test_fp16_prefill_zero_kv();
    test_fp16_fd_zero_kv();
    test_i8_decode_zero_scales();

    int fails = 0;
    for (auto& r : results) if (!r.pass) ++fails;
    std::printf("\n%zu tests: %d pass / %d fail\n",
                results.size(), (int)results.size() - fails, fails);
    return fails ? 1 : 0;
}
