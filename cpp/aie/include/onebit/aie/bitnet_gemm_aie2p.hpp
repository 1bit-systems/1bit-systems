// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 halo-ai-core / bong-water-water-bong.
//
// onebit::aie::BitnetGemmAIE2P — direct-link libxrt wrapper around the
// authored AIE2P ternary-GEMM xclbin (`bitnet_gemm_bf16`, M=K=N=64).
//
// Translates `programming_examples/basic/bitnet_gemm/single_core/
// run_pyxrt_bitnet.py` into a synchronous C++23 dispatch path:
//
//     auto eng = BitnetGemmAIE2P::load(xclbin_path, insts_path);
//     eng->gemm(a_bf16, w_packed_u32, c_bf16);   // blocking
//
// Sibling to the existing dlopen-backed `XrtBackend` in this crate.
// That path is the lemonade-routed BitNet GEMV lane and is intentionally
// loose about libxrt presence (CI hosts build clean without it). This
// header is the *direct-link* lane: you must link against libxrt at
// build time and the symbols must resolve at startup. Tests are gated
// on `ONEBIT_REAL_BACKEND=1` so a CI host without /dev/accel/accel0
// reports skipped, not failed.
//
// Tile shape is fixed by the compiled xclbin (M=K=N=64 today). A future
// re-compile widens the tile; the API takes spans whose length must
// equal the compiled size and reports ShapeMismatch otherwise.
//
// Core Guidelines compliance:
//   * I.27 — pImpl, ctor/dtor declared here, defaulted in .cpp.
//   * E.27 — std::expected on every fallible path; no exceptions.
//   * F.6  — accessors marked noexcept.

#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string_view>

#include "onebit/aie/error.hpp"

namespace onebit::aie {

// Compile-time tile dims for the currently authored xclbin. If the
// authoring agent re-compiles for a different tile, bump these AND the
// path baked into the load() callsite — they must match or load() will
// reject the artifact at runtime via the BO size check.
inline constexpr std::uint32_t kBitnetGemmAIE2P_M = 64;
inline constexpr std::uint32_t kBitnetGemmAIE2P_K = 64;
inline constexpr std::uint32_t kBitnetGemmAIE2P_N = 64;

// HALO_V2 packing density: 16 ternary codes per uint32 word (2 bits
// per code: 00=0, 01=+1, 10=-1, 11=reserved). Matches pack_halo_v2 in
// run_pyxrt_bitnet.py.
inline constexpr std::uint32_t kBitnetGemmAIE2P_CodesPerU32 = 16;

class BitnetGemmAIE2P {
public:
    // Load + register the xclbin and the instruction stream. Both
    // paths must point to files produced by the IRON build for this
    // tile shape; otherwise the call returns
    //   * XclbinNotFound      — file missing / unreadable
    //   * Xrt                 — libxrt rejected the artifact
    //   * LibraryUnavailable  — no /dev/accel/accel0 (probably wrong host)
    //
    // Returns a populated engine on success. Move-only.
    [[nodiscard]] static std::expected<BitnetGemmAIE2P, Error>
    load(const std::filesystem::path& xclbin_path,
         const std::filesystem::path& insts_path);

    ~BitnetGemmAIE2P();

    BitnetGemmAIE2P(const BitnetGemmAIE2P&)            = delete;
    BitnetGemmAIE2P& operator=(const BitnetGemmAIE2P&) = delete;
    BitnetGemmAIE2P(BitnetGemmAIE2P&&) noexcept;
    BitnetGemmAIE2P& operator=(BitnetGemmAIE2P&&) noexcept;

    // Synchronous launch. Spans are validated against the compiled
    // tile shape; mismatch returns ShapeMismatch without dispatching.
    // After return, c_bf16 holds the bf16 product. Caller is
    // responsible for applying any post-mmul rescale.
    //
    // Argument order matches the kernel's positional signature
    // (A, W, C). Run kernel registers handled internally.
    [[nodiscard]] std::expected<void, Error>
    gemm(std::span<const std::uint16_t> a_bf16,
         std::span<const std::uint32_t> w_packed,
         std::span<std::uint16_t>       c_bf16);

    // Kernel name reported by the xclbin (typically "MLIR_AIE"). Stable
    // across calls; cheap.
    [[nodiscard]] std::string_view kernel_name() const noexcept;

    // True when the engine holds a valid xrt::kernel + BOs. False after
    // a moved-from access.
    [[nodiscard]] bool is_ready() const noexcept;

    // Compiled tile size — convenience for callers that want to gate on
    // it without recompiling. Constant for the lifetime of the object.
    [[nodiscard]] static constexpr std::uint32_t tile_m() noexcept { return kBitnetGemmAIE2P_M; }
    [[nodiscard]] static constexpr std::uint32_t tile_k() noexcept { return kBitnetGemmAIE2P_K; }
    [[nodiscard]] static constexpr std::uint32_t tile_n() noexcept { return kBitnetGemmAIE2P_N; }

    // Element counts a caller's spans must match.
    [[nodiscard]] static constexpr std::size_t a_elems() noexcept {
        return static_cast<std::size_t>(kBitnetGemmAIE2P_M) * kBitnetGemmAIE2P_K;
    }
    [[nodiscard]] static constexpr std::size_t w_elems_u32() noexcept {
        return (static_cast<std::size_t>(kBitnetGemmAIE2P_K) * kBitnetGemmAIE2P_N) /
               kBitnetGemmAIE2P_CodesPerU32;
    }
    [[nodiscard]] static constexpr std::size_t c_elems() noexcept {
        return static_cast<std::size_t>(kBitnetGemmAIE2P_M) * kBitnetGemmAIE2P_N;
    }

private:
    BitnetGemmAIE2P();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace onebit::aie
