// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 halo-ai-core / bong-water-water-bong.
//
// onebit::aie::BitnetGemmAIE2P — direct-link libxrt wrapper around the
// authored AIE2P ternary-GEMM xclbin. Tile shape (M=K=N) is detected at
// load() time from the xclbin's W-arg byte size; Phase-1 ships 64, the
// Phase-2 lane ships 512.
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
// Tile shape is now read from the loaded xclbin at runtime. The default
// (Phase-1) xclbin ships M=K=N=64; the Phase-2 xclbin is M=K=N=512. The
// per-instance loaded_tile_*() / loaded_*_elems() accessors report what
// the live engine is configured for. The kBitnetGemmAIE2P_{M,K,N} and
// the static tile_*() / *_elems() entry points retain the Phase-1
// defaults for source-compat with callers that depend on a compile-time
// tile (engine_npu_dispatch.cpp, dispatch_smoke_test.cpp). For runtime
// validation against the live xclbin, prefer the loaded_*() accessors.
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

// Default tile dims (Phase-1 xclbin shape). Kept as inline constexpr so
// callers that want a compile-time-known shape still have one. They do
// NOT reflect what the live engine loaded — query the per-instance
// loaded_tile_*() accessors for that.
inline constexpr std::uint32_t kBitnetGemmAIE2P_M = 64;
inline constexpr std::uint32_t kBitnetGemmAIE2P_K = 64;
inline constexpr std::uint32_t kBitnetGemmAIE2P_N = 64;

// HALO_V2 packing density: 16 ternary codes per uint32 word (2 bits
// per code: 00=0, 01=+1, 10=-1, 11=reserved). Matches pack_halo_v2 in
// run_pyxrt_bitnet.py. This invariant is set by the kernel's code-pack
// contract and is INDEPENDENT of the tile size — it stays constexpr.
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

    // Synchronous launch. Spans are validated against the LOADED tile
    // shape (loaded_a_elems() / loaded_w_elems_u32() / loaded_c_elems());
    // mismatch returns ShapeMismatch without dispatching. After return,
    // c_bf16 holds the bf16 product. Caller is responsible for applying
    // any post-mmul rescale.
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

    // Default (compile-time) tile size. Reflects kBitnetGemmAIE2P_{M,K,N}
    // — the Phase-1 64-baked shape — and is independent of which xclbin a
    // given engine has loaded. Callers that need to gate on the live tile
    // (e.g. tiled_gemv) MUST use loaded_tile_*() instead.
    [[nodiscard]] static constexpr std::uint32_t tile_m() noexcept { return kBitnetGemmAIE2P_M; }
    [[nodiscard]] static constexpr std::uint32_t tile_k() noexcept { return kBitnetGemmAIE2P_K; }
    [[nodiscard]] static constexpr std::uint32_t tile_n() noexcept { return kBitnetGemmAIE2P_N; }

    // Default-tile element counts a caller's spans must match for the
    // Phase-1 shape. As with tile_*(), these are compile-time and reflect
    // the default — NOT the loaded xclbin. Use loaded_*_elems() to get
    // the live engine's expected span lengths.
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

    // Per-instance loaded-tile accessors. Populated by load() from the
    // xclbin's W-arg byte size; stable for the lifetime of the engine.
    // Returns 0 on a moved-from / un-ready handle. These are the values
    // gemm() validates against, NOT the static defaults above.
    [[nodiscard]] std::uint32_t loaded_tile_m() const noexcept;
    [[nodiscard]] std::uint32_t loaded_tile_k() const noexcept;
    [[nodiscard]] std::uint32_t loaded_tile_n() const noexcept;

    // Per-instance element counts derived from the loaded tile dims.
    // Returns 0 on a moved-from / un-ready handle.
    [[nodiscard]] std::size_t loaded_a_elems() const noexcept;
    [[nodiscard]] std::size_t loaded_w_elems_u32() const noexcept;
    [[nodiscard]] std::size_t loaded_c_elems() const noexcept;

private:
    BitnetGemmAIE2P();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

} // namespace onebit::aie
