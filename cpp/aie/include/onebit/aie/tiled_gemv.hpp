// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 halo-ai-core / bong-water-water-bong.
//
// onebit::aie::tiled_gemv — host-side pad-and-tile dispatcher that lets
// the production halo-1bit-2b dimensions (hs=2560, is=6912, V=128256)
// run on top of the fixed-tile AIE2P xclbin authored by Phase-2
// (M=K=N=512, with a 64x64 inner mmul). The xclbin is recompile-fixed,
// so all live-layer odd shapes get padded to the next multiple of `tile`
// and dispatched as ceil(N/tile) * ceil(K/tile) calls of the leaf
// `BitnetGemmAIE2P::gemm`.
//
// Mathematically this is GEMV: A is a single bf16 row of length k_total
// and C is a single bf16 row of length n_total. The leaf is GEMM with a
// fixed M tile, so we splat the activation into the first row of the
// M-tile and read row 0 of the output. The wasted M-1 rows are the
// price of using a fixed-shape kernel; the alternative — recompiling
// the xclbin per layer — was the constraint we are designing around.
//
// K-block accumulation is fp32 to match Phase-2's in-tile fp32 running
// sum (see bitnet_gemm_iron_512.py: `f32 = str_to_dtype("f32")`). The
// leaf currently emits bf16 (host-side cast of Phase-2's fp32 result),
// so we lift each partial back to fp32, sum, then bf16-store on the
// final write. This is one extra cast pair per K-block; cheap.
//
// Core Guidelines compliance:
//   * F.6   — wrapper is a free function; no hidden state.
//   * E.27  — std::expected<void, Error> on the public surface.
//   * I.27  — config struct is trivially-copyable POD; no pImpl needed.
//   * SL.13 — std::span everywhere; no raw pointer/length pairs.

#pragma once

#include <cstdint>
#include <expected>
#include <span>

#include "onebit/aie/bitnet_gemm_aie2p.hpp"
#include "onebit/aie/error.hpp"

namespace onebit::aie {

// Pad-and-tile dispatch config. Trivially-copyable POD; passed by value
// at the call site. `tile` MUST equal the kernel's compiled tile size
// (BitnetGemmAIE2P::tile_n() == tile_k() == tile_m()); otherwise the
// call returns ShapeMismatch without dispatching.
struct TiledGemvCfg {
    int n_total;       // unpadded output dim (rows of W, output of GEMV)
    int k_total;       // unpadded input  dim (cols of W, length of A)
    int tile = 512;    // must match the loaded xclbin's compiled tile
};

// Pad-and-tile GEMV. `a_bf16` is a row-major bf16 vector of length
// cfg.k_total. `w_packed` is HALO_V2-packed weights for the
// (cfg.n_total, cfg.k_total) matrix in row-major code order: each row
// has cfg.k_total ternary codes packed 16-per-uint32 (LSB-first), so
// the buffer length is (cfg.n_total * cfg.k_total) / 16 uint32s. Both
// dims of the unpadded (n_total, k_total) shape MUST be multiples of
// 16 — anything else can't be HALO_V2 packed without straddling a u32
// boundary, which the slicer doesn't try to handle. (Strix-Halo live
// shapes 2560, 640, 6912, 128256 all satisfy this.)
//
// Per-row scales are folded into A by the caller before this call —
// same contract as the Phase-2 host runner's A_baked_bf16. We do not
// rescale here.
//
// Output `c_bf16` is filled with the cfg.n_total bf16 dot-products.
// On failure the buffer is left in a partially-written state; callers
// should not consume c_bf16 unless the call returns success.
[[nodiscard]] std::expected<void, Error>
tiled_gemv(BitnetGemmAIE2P&               kernel,
           std::span<const std::uint16_t> a_bf16,
           std::span<const std::uint32_t> w_packed,
           std::span<std::uint16_t>       c_bf16,
           const TiledGemvCfg&            cfg);

} // namespace onebit::aie
