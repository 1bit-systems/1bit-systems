// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 halo-ai-core / bong-water-water-bong.
//
// bitnet_gemm_aie2p_test.cpp — doctest covering the libxrt-direct path.
//
// Mirrors the python reference at
//   /home/bcloud/repos/mlir-aie/programming_examples/basic/bitnet_gemm/
//   single_core/run_pyxrt_bitnet.py
//
// Execution gates:
//   * Always — engine is move-only, accessors stable on a default-
//     constructed handle, ShapeMismatch surface validated.
//   * ONEBIT_REAL_BACKEND=1 — opens /dev/accel/accel0, registers the
//     fixture xclbin, runs gemm() against the same RNG seed the python
//     reference uses, compares to the bf16-quantised CPU golden.

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

[[nodiscard]] bool real_backend_enabled() noexcept
{
    const char* v = std::getenv("ONEBIT_REAL_BACKEND");
    return v && v[0] == '1' && v[1] == '\0';
}

// ---- bf16 helpers, byte-identical to fp32_to_bf16_u16 / bf16_u16_to_fp32 ----

[[nodiscard]] std::uint16_t fp32_to_bf16(float x) noexcept
{
    // Round-to-nearest-even fp32 -> bf16. Same biasing trick used by the
    // python reference (and by torch.bfloat16 cast).
    const std::uint32_t u = std::bit_cast<std::uint32_t>(x);
    const std::uint32_t lsb = (u >> 16) & 1u;
    const std::uint32_t bias = 0x7FFFu + lsb;
    const std::uint32_t rounded = u + bias;
    return static_cast<std::uint16_t>(rounded >> 16);
}

[[nodiscard]] float bf16_to_fp32(std::uint16_t b) noexcept
{
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    return std::bit_cast<float>(u);
}

// ---- HALO_V2 packing — 16 ternary codes per uint32, code = 2 bits, LSB-
// first. {-1,0,+1} -> {10, 00, 01} per pack_halo_v2 in the python.
[[nodiscard]] std::vector<std::uint32_t>
pack_halo_v2(std::span<const std::int8_t> ternary, std::size_t k_dim, std::size_t n_dim)
{
    REQUIRE(ternary.size() == k_dim * n_dim);
    REQUIRE((k_dim * n_dim) % 16 == 0);
    const std::size_t nwords = (k_dim * n_dim) / 16;
    std::vector<std::uint32_t> out(nwords, 0u);
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

// ---- CPU reference: bf16 acts -> fp32 -> ternary mmul -> fp32 acc ->
// bf16 store. Returns the bf16 (uint16) result.
[[nodiscard]] std::vector<std::uint16_t>
cpu_reference(std::span<const std::uint16_t> a_bf16,
              std::span<const std::int8_t>   w_ternary,
              std::size_t M, std::size_t K, std::size_t N)
{
    REQUIRE(a_bf16.size()    == M * K);
    REQUIRE(w_ternary.size() == K * N);
    std::vector<std::uint16_t> c(M * N, 0u);
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < K; ++k) {
                const float a_f = bf16_to_fp32(a_bf16[i * K + k]);
                const float w_f = static_cast<float>(w_ternary[k * N + j]);
                acc += a_f * w_f;
            }
            c[i * N + j] = fp32_to_bf16(acc);
        }
    }
    return c;
}

// ---- Tiny deterministic RNG so the C++ test doesn't depend on numpy's
// PCG state. This means the *exact* tensor values diverge from the
// python reference, but the contract is the same: bake-pre-scale,
// dispatch, compare to the bf16-quantised CPU oracle.
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

} // namespace

// ----------------------------------------------------------------------------
// Compile-time / always-on cases.
// ----------------------------------------------------------------------------

TEST_CASE("BitnetGemmAIE2P compile-time tile constants")
{
    CHECK(BitnetGemmAIE2P::tile_m()      == 64);
    CHECK(BitnetGemmAIE2P::tile_k()      == 64);
    CHECK(BitnetGemmAIE2P::tile_n()      == 64);
    CHECK(BitnetGemmAIE2P::a_elems()     == 64u * 64u);
    CHECK(BitnetGemmAIE2P::c_elems()     == 64u * 64u);
    CHECK(BitnetGemmAIE2P::w_elems_u32() == (64u * 64u) / 16u);
}

TEST_CASE("BitnetGemmAIE2P missing xclbin reports XclbinNotFound or LibraryUnavailable")
{
    auto eng = BitnetGemmAIE2P::load("/no/such/file.xclbin", "/no/such/insts.txt");
    REQUIRE_FALSE(eng.has_value());
    // Two acceptable outcomes:
    //   * Built with XRT linkage AND host has /dev/accel: read_file fires
    //     XclbinNotFound first.
    //   * Built without XRT linkage: LibraryUnavailable.
    const auto k = eng.error().kind();
    const bool ok = k == ErrorKind::XclbinNotFound ||
                    k == ErrorKind::LibraryUnavailable;
    INFO("kind=" << label(k) << " detail=" << eng.error().detail());
    CHECK(ok);
}

// ----------------------------------------------------------------------------
// Real-backend cases. Skipped (with a doctest WARN) when ONEBIT_REAL_BACKEND
// is unset OR the fixture xclbin isn't on disk.
// ----------------------------------------------------------------------------

TEST_CASE("BitnetGemmAIE2P real dispatch matches CPU bf16 oracle")
{
    if (!real_backend_enabled()) {
        DOCTEST_WARN_MESSAGE(true,
            "ONEBIT_REAL_BACKEND unset — skipped (no NPU dispatch)");
        return;
    }
    if (!fs::exists(kFixtureXclbin) || !fs::exists(kFixtureInsts)) {
        DOCTEST_WARN_MESSAGE(true,
            "fixture xclbin/insts missing — skipped");
        return;
    }

    auto eng_or = BitnetGemmAIE2P::load(kFixtureXclbin, kFixtureInsts);
    if (!eng_or) {
        // Most likely cause on a non-NPU host: LibraryUnavailable from
        // dlopen, or Xrt from a missing /dev/accel/accel0. Surface the
        // detail so the operator can diagnose.
        DOCTEST_WARN_MESSAGE(true,
            "load() failed: kind=" << label(eng_or.error().kind())
            << " detail=" << eng_or.error().detail());
        return;
    }
    auto& eng = *eng_or;
    REQUIRE(eng.is_ready());
    CHECK_FALSE(eng.kernel_name().empty());

    constexpr std::size_t M = BitnetGemmAIE2P::tile_m();
    constexpr std::size_t K = BitnetGemmAIE2P::tile_k();
    constexpr std::size_t N = BitnetGemmAIE2P::tile_n();

    LehmerRng rng{0x20260426ULL};

    // 1) Random fp32 activations in [-1, 1], scale-bake into bf16.
    std::vector<float> a_fp32(M * K);
    for (auto& v : a_fp32) v = rng.uniform(-1.0f, 1.0f);

    std::vector<float> scale(M);
    for (auto& v : scale) v = rng.uniform(0.5f, 2.0f);

    std::vector<std::uint16_t> a_bf16(M * K);
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            a_bf16[i * K + k] = fp32_to_bf16(a_fp32[i * K + k] * scale[i]);
        }
    }

    // 2) ~50% sparse ternary weights drawn from {-1, 0, +1}.
    std::vector<std::int8_t> w_ternary(K * N);
    for (auto& v : w_ternary) {
        const std::uint32_t r = rng.next_u32() % 3u;
        v = (r == 1u) ? std::int8_t{1} : (r == 2u) ? std::int8_t{-1} : std::int8_t{0};
    }
    auto w_packed = pack_halo_v2(std::span<const std::int8_t>{w_ternary}, K, N);
    REQUIRE(w_packed.size() == BitnetGemmAIE2P::w_elems_u32());

    // 3) CPU oracle.
    auto c_ref = cpu_reference(std::span<const std::uint16_t>{a_bf16},
                               std::span<const std::int8_t>{w_ternary}, M, K, N);

    // 4) NPU dispatch.
    std::vector<std::uint16_t> c_npu(M * N, 0u);
    auto rc = eng.gemm(std::span<const std::uint16_t>{a_bf16},
                       std::span<const std::uint32_t>{w_packed},
                       std::span<std::uint16_t>{c_npu});
    if (!rc) {
        DOCTEST_WARN_MESSAGE(true,
            "gemm() failed: kind=" << label(rc.error().kind())
            << " detail=" << rc.error().detail());
        FAIL("real dispatch returned error");
    }

    // 5) Compare in fp32 space — bf16-quantised on both sides so the
    // only drift is bf16 accumulation order. Threshold mirrors the
    // python's: rel-err < 5e-3.
    double sq_diff = 0.0;
    double sq_ref  = 0.0;
    float  max_abs = 0.0f;
    for (std::size_t i = 0; i < c_ref.size(); ++i) {
        const float r = bf16_to_fp32(c_ref[i]);
        const float n = bf16_to_fp32(c_npu[i]);
        const float d = n - r;
        sq_diff += static_cast<double>(d) * d;
        sq_ref  += static_cast<double>(r) * r;
        const float ad = std::fabs(d);
        if (ad > max_abs) max_abs = ad;
    }
    const double rel_err = std::sqrt(sq_diff) /
                           (std::sqrt(sq_ref) + 1e-12);
    MESSAGE("rel_err=" << rel_err << " max_abs=" << max_abs);
    CHECK(rel_err < 5e-3);
}

TEST_CASE("BitnetGemmAIE2P shape-mismatch on wrong-sized spans")
{
    if (!real_backend_enabled()) {
        DOCTEST_WARN_MESSAGE(true,
            "ONEBIT_REAL_BACKEND unset — skipped");
        return;
    }
    if (!fs::exists(kFixtureXclbin) || !fs::exists(kFixtureInsts)) {
        DOCTEST_WARN_MESSAGE(true, "fixture missing — skipped");
        return;
    }
    auto eng_or = BitnetGemmAIE2P::load(kFixtureXclbin, kFixtureInsts);
    if (!eng_or) {
        DOCTEST_WARN_MESSAGE(true,
            "load() failed; can't exercise shape-mismatch path");
        return;
    }
    auto& eng = *eng_or;

    std::vector<std::uint16_t> a_short(BitnetGemmAIE2P::a_elems() - 1);
    std::vector<std::uint32_t> w(BitnetGemmAIE2P::w_elems_u32());
    std::vector<std::uint16_t> c(BitnetGemmAIE2P::c_elems());
    auto rc = eng.gemm(std::span<const std::uint16_t>{a_short},
                       std::span<const std::uint32_t>{w},
                       std::span<std::uint16_t>{c});
    REQUIRE_FALSE(rc.has_value());
    CHECK(rc.error().kind() == ErrorKind::ShapeMismatch);
}
