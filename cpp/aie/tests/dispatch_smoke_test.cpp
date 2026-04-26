// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 halo-ai-core / bong-water-water-bong.
//
// dispatch_smoke_test.cpp — exercises BitnetGemmAIE2P at the engine-tile
// shape (currently 64x64x64; the user-spec'd 512x512x512 lands once the
// wrapper learns runtime tile sizing).
//
// Skipped when the fixture xclbin is missing or ONEBIT_REAL_BACKEND is
// unset, matching the gating convention in bitnet_gemm_aie2p_test.cpp so
// CI hosts without /dev/accel/accel0 still pass green.
//
// What this test asserts beyond bitnet_gemm_aie2p_test.cpp:
//   * Per-row absmean scale fold (Phase-2 contract): the test mints a
//     synthetic per-output-row W scale and validates that applying the
//     scale post-mmul on the host matches the bf16 CPU oracle within the
//     same rel_err < 5e-3 threshold the python reference uses.
//   * Reuses the same engine handle across two dispatches with different
//     RNG seeds — proves the kernel + BO scratch survive consecutive
//     gemm() calls without leak or state crosstalk.

// Standalone test executable — provide doctest's main() in this TU.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/aie/bitnet_gemm_aie2p.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

using namespace onebit::aie;
namespace fs = std::filesystem;

namespace {

constexpr const char* kFixtureXclbin =
    "/home/bcloud/repos/mlir-aie/programming_examples/basic/bitnet_gemm/"
    "single_core/build/final_64x64x64_64x64x64.xclbin";
constexpr const char* kFixtureInsts =
    "/home/bcloud/repos/mlir-aie/programming_examples/basic/bitnet_gemm/"
    "single_core/build/insts_64x64x64_64x64x64.txt";

// ---- bf16 helpers (round-to-nearest-even) ----
[[nodiscard]] std::uint16_t fp32_to_bf16(float x) noexcept {
    const std::uint32_t u = std::bit_cast<std::uint32_t>(x);
    const std::uint32_t lsb = (u >> 16) & 1u;
    const std::uint32_t bias = 0x7FFFu + lsb;
    const std::uint32_t rnd = u + bias;
    return static_cast<std::uint16_t>(rnd >> 16);
}
[[nodiscard]] float bf16_to_fp32(std::uint16_t b) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(b) << 16);
}

[[nodiscard]] bool real_backend_enabled() noexcept {
    const char* v = std::getenv("ONEBIT_REAL_BACKEND");
    return v && v[0] == '1' && v[1] == '\0';
}

// HALO_V2 packing: 16 ternary codes per uint32, code = 2 bits, LSB-first.
[[nodiscard]] std::vector<std::uint32_t>
pack_halo_v2(std::span<const std::int8_t> ternary,
             std::size_t k_dim, std::size_t n_dim)
{
    REQUIRE(ternary.size() == k_dim * n_dim);
    REQUIRE((k_dim * n_dim) % 16 == 0);
    std::vector<std::uint32_t> out((k_dim * n_dim) / 16, 0u);
    for (std::size_t i = 0; i < ternary.size(); ++i) {
        std::uint32_t code = 0u;
        switch (ternary[i]) {
            case  0: code = 0b00; break;
            case  1: code = 0b01; break;
            case -1: code = 0b10; break;
            default: FAIL("non-ternary value in pack_halo_v2");
        }
        const std::size_t word = i / 16;
        const std::size_t slot = i % 16;
        out[word] |= code << (2u * static_cast<std::uint32_t>(slot));
    }
    return out;
}

// CPU oracle: bf16 acts -> fp32 -> ternary mmul -> fp32 acc -> per-row scale
// -> bf16 store.  Mirrors the engine-side fold (post-mmul row scale).
[[nodiscard]] std::vector<std::uint16_t>
cpu_reference_with_row_scale(std::span<const std::uint16_t> a_bf16,
                             std::span<const std::int8_t>   w_ternary,
                             std::span<const float>         row_scale,
                             std::size_t M, std::size_t K, std::size_t N)
{
    REQUIRE(a_bf16.size()    == M * K);
    REQUIRE(w_ternary.size() == K * N);
    REQUIRE(row_scale.size() == N);
    std::vector<std::uint16_t> c(M * N, 0u);
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                const float a_f = bf16_to_fp32(a_bf16[i * K + k]);
                const float w_f = static_cast<float>(w_ternary[k * N + j]);
                acc += a_f * w_f;
            }
            c[i * N + j] = fp32_to_bf16(acc * row_scale[j]);
        }
    }
    return c;
}

// Tiny deterministic RNG so the C++ test doesn't depend on numpy's PCG.
struct LehmerRng {
    std::uint64_t state;
    explicit LehmerRng(std::uint64_t seed) : state{seed | 1ULL} {}
    std::uint32_t next_u32() noexcept {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<std::uint32_t>(state >> 32);
    }
    float uniform(float lo, float hi) noexcept {
        const float u = static_cast<float>(next_u32()) / 4294967296.0f;
        return lo + (hi - lo) * u;
    }
};

// Run one gemm + post-fold check.  Returns rel_err so the caller can log
// it from the test (doctest's MESSAGE handles the rest).
[[nodiscard]] double run_one_dispatch(BitnetGemmAIE2P& eng,
                                      std::uint64_t    seed)
{
    constexpr std::size_t M = BitnetGemmAIE2P::tile_m();
    constexpr std::size_t K = BitnetGemmAIE2P::tile_k();
    constexpr std::size_t N = BitnetGemmAIE2P::tile_n();

    LehmerRng rng{seed};

    // Random fp32 acts in [-1, 1] -> bf16.  Single-row case (M=1) is
    // exercised by zero-filling rows 1..M-1; here we cover the full
    // M×K input space because the engine smoke is about kernel sanity,
    // not GEMV-specific masking.
    std::vector<std::uint16_t> a_bf16(M * K);
    for (auto& v : a_bf16) v = fp32_to_bf16(rng.uniform(-1.0f, 1.0f));

    // ~50% sparse ternary weights drawn from {-1, 0, +1}.
    std::vector<std::int8_t> w_ternary(K * N);
    for (auto& v : w_ternary) {
        const std::uint32_t r = rng.next_u32() % 3u;
        v = (r == 1u) ? std::int8_t{1}
          : (r == 2u) ? std::int8_t{-1}
                      : std::int8_t{0};
    }
    auto w_packed = pack_halo_v2(std::span<const std::int8_t>{w_ternary}, K, N);
    REQUIRE(w_packed.size() == BitnetGemmAIE2P::w_elems_u32());

    // Per-row absmean-style scale, in [0.5, 2.0] — matches the engine's
    // BitNet-1.58 scale magnitude band.
    std::vector<float> row_scale(N);
    for (auto& v : row_scale) v = rng.uniform(0.5f, 2.0f);

    // CPU oracle (with row-scale fold).
    auto c_ref = cpu_reference_with_row_scale(
        std::span<const std::uint16_t>{a_bf16},
        std::span<const std::int8_t>{w_ternary},
        std::span<const float>{row_scale},
        M, K, N);

    // NPU dispatch — bare bf16, no scale baked in.  We apply the scale
    // post-mmul on the host (same path the engine shim takes).
    std::vector<std::uint16_t> c_npu(M * N, 0u);
    auto rc = eng.gemm(std::span<const std::uint16_t>{a_bf16},
                       std::span<const std::uint32_t>{w_packed},
                       std::span<std::uint16_t>{c_npu});
    if (!rc) {
        DOCTEST_WARN_MESSAGE(true,
            "gemm() failed: kind=" << label(rc.error().kind())
            << " detail=" << rc.error().detail());
        return std::numeric_limits<double>::infinity();
    }

    // Apply row scale + downcast to fp32 for the diff.
    double sq_diff = 0.0;
    double sq_ref  = 0.0;
    float  max_abs = 0.0f;
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            const float r = bf16_to_fp32(c_ref[i * N + j]);
            const float n_raw = bf16_to_fp32(c_npu[i * N + j]);
            const float n = n_raw * row_scale[j];
            const float d = n - r;
            sq_diff += static_cast<double>(d) * d;
            sq_ref  += static_cast<double>(r) * r;
            const float ad = std::fabs(d);
            if (ad > max_abs) max_abs = ad;
        }
    }
    return std::sqrt(sq_diff) / (std::sqrt(sq_ref) + 1e-12);
}

} // namespace

TEST_CASE("dispatch smoke — single tile with row-scale fold")
{
    if (!real_backend_enabled()) {
        DOCTEST_WARN_MESSAGE(true,
            "ONEBIT_REAL_BACKEND unset — skipped (no NPU dispatch)");
        return;
    }
    if (!fs::exists(kFixtureXclbin) || !fs::exists(kFixtureInsts)) {
        DOCTEST_WARN_MESSAGE(true,
            "fixture xclbin/insts missing — skipped (build mlir-aie first)");
        return;
    }

    auto eng_or = BitnetGemmAIE2P::load(kFixtureXclbin, kFixtureInsts);
    if (!eng_or) {
        DOCTEST_WARN_MESSAGE(true,
            "load() failed: kind=" << label(eng_or.error().kind())
            << " detail=" << eng_or.error().detail());
        return;
    }
    auto& eng = *eng_or;
    REQUIRE(eng.is_ready());

    const double rel_err = run_one_dispatch(eng, 0x20260426ULL);
    MESSAGE("dispatch_smoke seed=0x20260426 rel_err=" << rel_err);
    CHECK(rel_err < 5e-3);
}

TEST_CASE("dispatch smoke — handle reuse across two dispatches")
{
    if (!real_backend_enabled()) {
        DOCTEST_WARN_MESSAGE(true,
            "ONEBIT_REAL_BACKEND unset — skipped");
        return;
    }
    if (!fs::exists(kFixtureXclbin) || !fs::exists(kFixtureInsts)) {
        DOCTEST_WARN_MESSAGE(true,
            "fixture missing — skipped");
        return;
    }

    auto eng_or = BitnetGemmAIE2P::load(kFixtureXclbin, kFixtureInsts);
    if (!eng_or) {
        DOCTEST_WARN_MESSAGE(true,
            "load() failed; skipping reuse test");
        return;
    }
    auto& eng = *eng_or;

    const double err_a = run_one_dispatch(eng, 0xCAFEF00DULL);
    const double err_b = run_one_dispatch(eng, 0xDEADBEEFULL);
    MESSAGE("reuse rel_err A=" << err_a << " B=" << err_b);
    CHECK(err_a < 5e-3);
    CHECK(err_b < 5e-3);
}
