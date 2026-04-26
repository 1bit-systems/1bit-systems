// h1b-sherry — header-only requantizer kernels.
//
// Faithful C++23 port of `tools/h1b-sherry/src/{tq1_unpack,sherry_pack,
// convert}.rs`. Layout references:
//   * `rocm-cpp/include/rocm_cpp/sherry.h` (kernel-side LUT + spec)
//   * `tools/h1b-sherry/src/sherry_pack.rs` (5-bit-per-group pack)
//   * `tools/h1b-sherry/src/tq1_unpack.rs` (base-3, 5 ternaries/byte)
//
// Header-only so the test TU and the CLI TU both inline the same code paths
// without a separate static library — keeps the compile clean and avoids the
// "is it the same packer?" question.

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace onebit::tools::h1b_sherry {

// ----- TQ1 v4 (base-3, 5 digits/byte) ---------------------------------------

// Logical cols rounded up to a multiple of 20 (TQ1's macro-group size).
[[nodiscard]] constexpr std::size_t tq1_row_bytes(std::size_t cols) noexcept
{
    const std::size_t cols_padded = ((cols + 19) / 20) * 20;
    return cols_padded / 5;
}

// Decode one TQ1 v4 row of `cols` logical weights. `packed` must be
// `tq1_row_bytes(cols)` bytes; `out` must be `cols` int8s.
//
// Each byte: byte = d0 + d1*3 + d2*9 + d3*27 + d4*81. d_i ∈ {0,1,2} maps
// to ternary {-1, 0, +1}. byte ≥ 243 is undefined-in-spec; we treat as
// all-zero (matches the kernel's zero-fill LUT fallback).
inline void unpack_tq1_row(std::span<const std::uint8_t> packed,
                           std::span<std::int8_t>        out,
                           std::size_t                   cols) noexcept
{
    assert(packed.size() == tq1_row_bytes(cols));
    assert(out.size() == cols);

    const std::size_t cols_padded = ((cols + 19) / 20) * 20;
    std::size_t k = 0;
    for (std::uint8_t byte : packed) {
        std::array<std::int8_t, 5> digits{};
        if (byte < 243) {
            std::uint32_t b = byte;
            for (auto& slot : digits) {
                slot = static_cast<std::int8_t>(static_cast<std::int32_t>(b % 3) - 1);
                b /= 3;
            }
        }
        for (std::int8_t d : digits) {
            if (k < cols) out[k] = d;
            ++k;
            if (k >= cols_padded) return;
        }
    }
}

// ----- Sherry 1.25-bit pack (5-bit-per-group, 3:4 sparse) -------------------

// Number of Sherry-packed bytes for a row of `cols` ternary weights.
// Mirrors `H1bWeightFormat::SherryV3.row_bytes`; cols % 32 == 0 required.
[[nodiscard]] constexpr std::size_t sherry_row_bytes(std::size_t cols) noexcept
{
    // assert cols % 32 == 0 — caller enforces.
    return cols * 5 / 32;
}

// Pick the "zero position" lane within a group of 4 ternary weights.
// Smallest absolute magnitude wins (zeros beat ±1); ties broken by
// lowest index. Matches `pick_zero_pos` in the Rust port.
[[nodiscard]] inline std::uint8_t
pick_zero_pos(const std::array<std::int8_t, 4>& g) noexcept
{
    std::uint8_t best = 0;
    auto absmag = [](std::int8_t v) -> std::uint8_t {
        return static_cast<std::uint8_t>(v < 0 ? -v : v);
    };
    std::uint8_t best_mag = absmag(g[0]);
    for (std::size_t i = 1; i < 4; ++i) {
        const std::uint8_t mag = absmag(g[i]);
        if (mag < best_mag) {
            best     = static_cast<std::uint8_t>(i);
            best_mag = mag;
        }
    }
    return best;
}

// Encode one group's 5-bit code: code = (zero_pos << 3) | signs_field.
//   signs_field bit i = 1 if the i-th non-zero-pos lane's weight is +1
//   or 0 (treated as +1 fallback to match the bench packer); 0 if -1.
[[nodiscard]] inline std::uint8_t
encode_group(const std::array<std::int8_t, 4>& g,
             std::uint8_t                      zero_pos) noexcept
{
    std::uint8_t signs    = 0;
    std::uint8_t sign_idx = 0;
    for (std::size_t p = 0; p < 4; ++p) {
        if (static_cast<std::uint8_t>(p) == zero_pos) continue;
        const std::int8_t v = g[p];
        const std::uint8_t bit = (v == 1 || v == 0) ? 1u : 0u;
        signs |= static_cast<std::uint8_t>(bit << sign_idx);
        ++sign_idx;
    }
    return static_cast<std::uint8_t>(
        (zero_pos << 3) | (signs & 0b111u));
}

struct PackRowStats {
    std::uint32_t forced_zero_flips = 0;
};

// Pack one row of `cols` ternary weights into Sherry's 5-bit-per-group
// layout. cols must be a multiple of 4; for byte-aligned rows cols % 32 == 0.
inline PackRowStats pack_sherry_row(std::span<const std::int8_t> ternary,
                                    std::span<std::uint8_t>      packed,
                                    std::size_t                  cols) noexcept
{
    assert(ternary.size() == cols);
    assert(packed.size() == sherry_row_bytes(cols));
    assert(cols % 4 == 0);

    std::fill(packed.begin(), packed.end(), std::uint8_t{0});
    PackRowStats stats{};
    const std::size_t groups = cols / 4;
    for (std::size_t g = 0; g < groups; ++g) {
        const std::size_t base = g * 4;
        std::array<std::int8_t, 4> grp{
            ternary[base + 0], ternary[base + 1],
            ternary[base + 2], ternary[base + 3]};
        const std::uint8_t zp = pick_zero_pos(grp);
        if (grp[zp] != 0) {
            // We're flipping a ±1 lane to zero — lossy.
            ++stats.forced_zero_flips;
        }
        const std::uint8_t code = encode_group(grp, zp);
        const std::size_t bit_pos = 5 * g;
        const std::size_t byte_idx = bit_pos >> 3;
        const std::uint32_t shift = static_cast<std::uint32_t>(bit_pos & 7u);
        // First byte gets the low (8-shift) bits of the code (or all 5 if
        // shift+5 <= 8). Sentinel: if shift==4, low fits in byte 0 with 1
        // bit straddling byte 1.
        if (shift < 8) {
            packed[byte_idx] = static_cast<std::uint8_t>(
                packed[byte_idx] | static_cast<std::uint8_t>(code << shift));
        }
        if (shift + 5 > 8) {
            packed[byte_idx + 1] = static_cast<std::uint8_t>(
                packed[byte_idx + 1] | static_cast<std::uint8_t>(code >> (8 - shift)));
        }
    }
    return stats;
}

// Decode one group from a packed Sherry row. Returns (zero_pos, signs).
[[nodiscard]] inline std::pair<std::uint8_t, std::uint8_t>
decode_group(std::span<const std::uint8_t> packed, std::size_t g) noexcept
{
    const std::size_t bit_pos  = 5 * g;
    const std::size_t byte_idx = bit_pos >> 3;
    const std::uint32_t shift  = static_cast<std::uint32_t>(bit_pos & 7u);
    const std::uint32_t lo_take = (8u - shift) > 5u ? 5u : (8u - shift);
    const std::uint8_t lo = static_cast<std::uint8_t>(
        (packed[byte_idx] >> shift) & ((1u << lo_take) - 1u));
    std::uint8_t code;
    if (shift + 5 > 8) {
        const std::uint32_t hi_bits = (shift + 5) - 8;
        const std::uint8_t hi = static_cast<std::uint8_t>(
            packed[byte_idx + 1] & ((1u << hi_bits) - 1u));
        code = static_cast<std::uint8_t>(lo | (hi << (8 - shift)));
    } else {
        code = static_cast<std::uint8_t>(lo & 0b11111u);
    }
    return {static_cast<std::uint8_t>((code >> 3) & 0b11u),
            static_cast<std::uint8_t>(code & 0b111u)};
}

// Decode a full Sherry row back to ternary {-1, 0, +1}.
inline void unpack_sherry_row(std::span<const std::uint8_t> packed,
                              std::span<std::int8_t>        out,
                              std::size_t                   cols) noexcept
{
    assert(packed.size() == sherry_row_bytes(cols));
    assert(out.size() == cols);
    const std::size_t groups = cols / 4;
    for (std::size_t g = 0; g < groups; ++g) {
        auto [zero_pos, signs] = decode_group(packed, g);
        std::uint8_t sign_idx = 0;
        for (std::uint8_t p = 0; p < 4; ++p) {
            const std::size_t dst = g * 4 + p;
            if (p == zero_pos) {
                out[dst] = 0;
            } else {
                const std::uint8_t bit = static_cast<std::uint8_t>((signs >> sign_idx) & 1u);
                out[dst] = static_cast<std::int8_t>(bit ? 1 : -1);
                ++sign_idx;
            }
        }
    }
}

// ----- error type -----------------------------------------------------------

struct ConvertError {
    std::string what;
};

// ----- aggregate stats ------------------------------------------------------

struct ConvertStats {
    std::uint64_t groups_total       = 0;
    std::uint64_t forced_zero_flips  = 0;
    std::uint64_t rows_total         = 0;
    std::uint32_t layers_processed   = 0;
    bool          hadamard_preserved = false;

    [[nodiscard]] double flip_fraction() const noexcept
    {
        return groups_total ? static_cast<double>(forced_zero_flips)
                                / static_cast<double>(groups_total)
                            : 0.0;
    }
};

} // namespace onebit::tools::h1b_sherry
