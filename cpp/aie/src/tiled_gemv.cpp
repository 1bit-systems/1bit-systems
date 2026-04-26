// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 halo-ai-core / bong-water-water-bong.
//
// tiled_gemv.cpp — implementation of onebit::aie::tiled_gemv.
//
// Layout assumption (locked at API): w_packed is row-major HALO_V2 over
// (n_total, k_total) ternary codes, packed 16-per-uint32 LSB-first.
// Each weight row holds k_total/16 u32 words; total length is
// n_total*k_total/16. The wrapper slices a (tile, tile) block at
// (n_block, k_block) into a contiguous (tile*tile/16) staging buffer
// before each leaf call. Zero-padding rows or partial-K columns is
// done with all-zero u32 words (= ternary code 00 = +0, no contribution
// to the dot product; safe for both the multiplicand AND the accumulator
// because Phase-2's in-tile zero_bitnet_f32_512 zeroes the C tile once
// per outer block).
//
// W layout transpose: the wrapper's input is documented as `[N, K]`
// row-major (matches engine_npu_dispatch.cpp's `[N*K/16]` host mirror),
// but the leaf's gemm() contract is `[K, N]` row-major (matches the
// IRON kernel's pW indexing in bitnet_gemm.cc / bitnet_gemm_512.cc and
// the leaf test in bitnet_gemm_aie2p_test.cpp). We therefore decode
// each ternary code from the source `[N, K]` block and re-pack into a
// `[K, N]` staging buffer per (n_block, k_block) call. The `[N, K] →
// [K, N]` transpose is done at code granularity (2-bit codes) so the
// HALO_V2 word packing rule is preserved end to end.
//
// Why this layout vs. the Phase-2 pretiled microtile order:
//   - The wrapper is a layout-stable abstraction over arbitrary K_total,
//     N_total. The microtile pretile is a per-tile transform that
//     belongs either inside the leaf or in a sibling helper called once
//     per (n_block, k_block) before kernel.gemm(). Keeping the wrapper
//     row-major-clean lets us unit-test it on a CPU stub backend without
//     dragging in pretile machinery, and lets the leaf take ownership
//     of any future re-tiling without breaking the API here.
//
// Math:
//   c[n] = sum_k a[k] * sign(W[n,k])    (W in {-1, 0, +1})
//
// Tiling:
//   for n_block in [0, ceil(N/tile)):
//       c_acc[tile] = 0.0f                                     (fp32)
//       for k_block in [0, ceil(K/tile)):
//           a_pad[tile_m * tile_k] = 0  (row 0 = a slice, else zero)
//           w_pad[tile_n * tile_k / 16] = 0
//           copy weight block (n_block, k_block) into w_pad
//           kernel.gemm(a_pad, w_pad, c_pad)                   (bf16-out)
//           # accumulate row 0 of c_pad into c_acc
//           for j in [0, tile): c_acc[j] += bf16_to_fp32(c_pad[j])
//       for j in [0, min(tile, n_total - n_block*tile)):
//           c_bf16[n_block*tile + j] = fp32_to_bf16(c_acc[j])
//
// We DO NOT zero c_pad before each kernel call: the kernel's first
// inner-tile op is `zero_bitnet_f32_512` which zeroes the C tile in
// the AIE-side accumulator. Host-side, c_pad is overwritten on the
// `bo_c.read` after `run.wait()`; the prior contents don't matter.

#include "onebit/aie/tiled_gemv.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace onebit::aie {

namespace {

// Round-to-nearest-even fp32 -> bf16. Identical bit pattern to
// torch.bfloat16 cast and to fp32_to_bf16_u16 in the Phase-2 reference.
[[nodiscard]] constexpr std::uint16_t fp32_to_bf16(float x) noexcept {
    const std::uint32_t u   = std::bit_cast<std::uint32_t>(x);
    const std::uint32_t lsb = (u >> 16) & 1u;
    const std::uint32_t bias = 0x7FFFu + lsb;
    const std::uint32_t rounded = u + bias;
    return static_cast<std::uint16_t>(rounded >> 16);
}

// bf16 -> fp32 via low-bit zero-extension. Bit-exact inverse of bf16
// quantisation (the round step is one-way; this just unpacks).
[[nodiscard]] constexpr float bf16_to_fp32(std::uint16_t b) noexcept {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    return std::bit_cast<float>(u);
}

constexpr std::uint32_t kCodesPerU32 = kBitnetGemmAIE2P_CodesPerU32;  // 16

} // namespace

std::expected<void, Error>
tiled_gemv(BitnetGemmAIE2P&               kernel,
           std::span<const std::uint16_t> a_bf16,
           std::span<const std::uint32_t> w_packed,
           std::span<std::uint16_t>       c_bf16,
           const TiledGemvCfg&            cfg)
{
    if (!kernel.is_ready()) {
        return std::unexpected(Error{ErrorKind::NotYetWired,
            "tiled_gemv called on un-ready kernel"});
    }

    // Per-leaf tile dims, read off the loaded xclbin (Phase-1 = 64,
    // Phase-2 = 512). The wrapper only sees the outer kernel tile;
    // the inner 64x64 mmul lives below.
    const std::size_t tile_m = kernel.loaded_tile_m();
    const std::size_t tile_k = kernel.loaded_tile_k();
    const std::size_t tile_n = kernel.loaded_tile_n();
    if (tile_m == 0 || tile_k == 0 || tile_n == 0) {
        return std::unexpected(Error{ErrorKind::NotYetWired,
            "tiled_gemv: kernel reports zero tile dims (un-ready)"});
    }

    // cfg.tile == 0 -> auto-detect (use the kernel's loaded tile). Any
    // positive value must match tile_k AND tile_n; mismatch is rejected
    // before dispatch so we don't silently produce a wrong-shape buffer.
    if (cfg.tile != 0) {
        if (cfg.tile < 0 ||
            static_cast<std::size_t>(cfg.tile) != tile_k ||
            static_cast<std::size_t>(cfg.tile) != tile_n) {
            return std::unexpected(Error{ErrorKind::ShapeMismatch,
                "cfg.tile (" + std::to_string(cfg.tile) +
                ") does not match leaf tile_k (" + std::to_string(tile_k) +
                ") / tile_n (" + std::to_string(tile_n) + ")"});
        }
    }

    if (cfg.n_total <= 0 || cfg.k_total <= 0) {
        return std::unexpected(Error{ErrorKind::ShapeMismatch,
            "n_total and k_total must be positive"});
    }

    const std::size_t n_total = static_cast<std::size_t>(cfg.n_total);
    const std::size_t k_total = static_cast<std::size_t>(cfg.k_total);

    // HALO_V2 packing requires both dims to be multiples of 16 codes
    // so that no row crosses a u32 boundary.  Live-shape constraints
    // (2560, 640, 6912, 128256) all satisfy this; we reject anything
    // else early to keep the slicer simple.
    if ((k_total % kCodesPerU32) != 0) {
        return std::unexpected(Error{ErrorKind::ShapeMismatch,
            "k_total must be a multiple of 16 (HALO_V2 word boundary)"});
    }

    if (a_bf16.size() != k_total) {
        return std::unexpected(Error{ErrorKind::ShapeMismatch,
            "a_bf16.size() != k_total"});
    }
    if (c_bf16.size() != n_total) {
        return std::unexpected(Error{ErrorKind::ShapeMismatch,
            "c_bf16.size() != n_total"});
    }
    const std::size_t k_words_total = (n_total * k_total) / kCodesPerU32;
    if (w_packed.size() != k_words_total) {
        return std::unexpected(Error{ErrorKind::ShapeMismatch,
            "w_packed.size() != n_total*k_total/16 (got " +
            std::to_string(w_packed.size()) + ", expected " +
            std::to_string(k_words_total) + ")"});
    }

    const std::size_t k_blocks = (k_total + tile_k - 1) / tile_k;
    const std::size_t n_blocks = (n_total + tile_n - 1) / tile_n;

    // Per-row stride (in u32 words) of the source w_packed buffer
    // (which is `[N, K]` row-major). The dest tile w_pad is `[K, N]`
    // row-major; its per-K-row stride is tile_n / 16, computed at the
    // staging site below.
    const std::size_t src_row_stride_u32 = k_total / kCodesPerU32;

    // Sanity-check the leaf's expected sizes match what we'll feed it.
    if (kernel.loaded_a_elems()     != tile_m * tile_k ||
        kernel.loaded_c_elems()     != tile_m * tile_n ||
        kernel.loaded_w_elems_u32() != (tile_k * tile_n) / kCodesPerU32) {
        return std::unexpected(Error{ErrorKind::ShapeMismatch,
            "leaf loaded_a_elems/c_elems/w_elems_u32 inconsistent with tile dims"});
    }

    // Staging buffers reused across all (n_block, k_block) calls. We
    // zero-init once and only overwrite the live-data slots; the rest
    // stays zero, which is the canonical "skip this row/column" value
    // for both the A-tile (row 0 carries the activation, rows 1..M-1
    // are dead weight) and the W-tile (HALO_V2 code 00 = +0).
    std::vector<std::uint16_t> a_pad(kernel.loaded_a_elems(), 0u);
    std::vector<std::uint32_t> w_pad(kernel.loaded_w_elems_u32(), 0u);
    std::vector<std::uint16_t> c_pad(kernel.loaded_c_elems(), 0u);

    // fp32 K-block accumulator, one per N-block (size = tile_n).
    std::vector<float> c_acc(tile_n, 0.0f);

    for (std::size_t nb = 0; nb < n_blocks; ++nb) {
        std::fill(c_acc.begin(), c_acc.end(), 0.0f);

        const std::size_t n_off  = nb * tile_n;
        const std::size_t n_live = std::min(tile_n, n_total - n_off);

        for (std::size_t kb = 0; kb < k_blocks; ++kb) {
            const std::size_t k_off  = kb * tile_k;
            const std::size_t k_live = std::min(tile_k, k_total - k_off);

            // ---- A staging: zero-fill row 0 past k_live, copy live a slice.
            // Rows 1..M-1 stay zero from the initial fill — they're
            // never overwritten (the dead-rows-of-the-M-tile cost).
            std::memset(a_pad.data(), 0,
                        kernel.loaded_a_elems() * sizeof(std::uint16_t));
            std::memcpy(a_pad.data(),
                        a_bf16.data() + k_off,
                        k_live * sizeof(std::uint16_t));

            // ---- W staging: zero-fill, then transpose [N,K] -> [K,N]
            // at 2-bit-code granularity for the live (n_live × k_live)
            // sub-block.
            //
            // Source layout : w_packed is HALO_V2 over [n_total, k_total]
            //                 row-major; row r=(n_off+r), col c=(k_off+c)
            //                 lives in u32 word
            //                   src_row_stride_u32 * r + (k_off+c)/16
            //                 at slot ((k_off+c) % 16).
            // Dest   layout : w_pad is HALO_V2 over [tile_k, tile_n]
            //                 row-major; row k, col n lives in u32 word
            //                   dst_row_stride_n_u32 * k + n/16
            //                 at slot (n % 16). dst_row_stride_n_u32 is
            //                 tile_n / 16 (NOT tile_k / 16 — the W tile
            //                 the leaf consumes is K-major-rows, N-cols).
            //
            // Code-level transpose is the only safe way: byte-stride
            // memcpy would require src and dst to share a stride, which
            // they don't for this transpose.
            std::memset(w_pad.data(), 0,
                        kernel.loaded_w_elems_u32() * sizeof(std::uint32_t));

            const std::size_t dst_row_stride_n_u32 = tile_n / kCodesPerU32;

            for (std::size_t r = 0; r < n_live; ++r) {
                const std::uint32_t* src_row =
                    w_packed.data() + (n_off + r) * src_row_stride_u32;
                for (std::size_t c = 0; c < k_live; ++c) {
                    const std::size_t src_idx       = k_off + c;
                    const std::size_t src_word      = src_idx / kCodesPerU32;
                    const std::size_t src_slot      = src_idx % kCodesPerU32;
                    const std::uint32_t code =
                        (src_row[src_word] >> (2u * src_slot)) & 0x3u;

                    // Dest: row=k (=c), col=n (=r) inside the (tile_k,
                    // tile_n) staging tile.
                    const std::size_t dst_word =
                        c * dst_row_stride_n_u32 + (r / kCodesPerU32);
                    const std::size_t dst_slot = r % kCodesPerU32;
                    w_pad[dst_word] |=
                        static_cast<std::uint32_t>(code & 0x3u)
                        << (2u * dst_slot);
                }
            }

            // ---- Dispatch.
            auto rc = kernel.gemm(
                std::span<const std::uint16_t>{a_pad},
                std::span<const std::uint32_t>{w_pad},
                std::span<std::uint16_t>{c_pad});
            if (!rc) return std::unexpected(rc.error());

            // ---- Accumulate row 0 of c_pad (the only live row) into
            // c_acc. Rows 1..M-1 are by-product of the GEMM-as-GEMV
            // mapping and are dropped here; they correspond to dot
            // products of zero activation rows with the weight, which
            // happen to also be zero, but we don't need that fact.
            for (std::size_t j = 0; j < tile_n; ++j) {
                c_acc[j] += bf16_to_fp32(c_pad[j]);
            }
        }

        // ---- Final cast + store of the unpadded leading slice.
        for (std::size_t j = 0; j < n_live; ++j) {
            c_bf16[n_off + j] = fp32_to_bf16(c_acc[j]);
        }
    }

    return {};
}

} // namespace onebit::aie
