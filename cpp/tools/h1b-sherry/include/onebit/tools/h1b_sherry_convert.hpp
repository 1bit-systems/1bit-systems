// SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
// Sherry — see LICENSE-SHERRY.md and SHERRY-FILES.txt at the repo root.
// Commercial use requires a separate license.
//
// File-level converter: .h1b (TQ1 v4) → .h1b (Sherry v3, fp16 flag set).
//
// Self-contained: parses + writes the Rust onebit-core streaming layout
// (magic + i32 version + 9*i32 cfg + (v>=2: 2*f32) + embedding + final_norm
// + per-layer norms + 7 ternary tensors). The C++ onebit::core::h1b::File
// parser today expects a layer-offset-table format that doesn't match this
// streaming layout, so we keep the read/write inline in this tool.

#pragma once

#include "onebit/tools/h1b_sherry.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace onebit::tools::h1b_sherry {

// On-disk flags (cfg[8]).
inline constexpr std::int32_t H1B_FLAG_HADAMARD_ROTATED = 0x1;
inline constexpr std::int32_t H1B_FLAG_SHERRY_FP16      = 0x2;

inline constexpr std::array<std::uint8_t, 4> H1B_MAGIC = {'H', '1', 'B', 0};

// Read whole file into a vector. Used in lieu of mmap for simplicity —
// these files are at most a few GB and the converter is offline.
[[nodiscard]] inline std::expected<std::vector<std::uint8_t>, ConvertError>
read_all(const std::filesystem::path& p)
{
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return std::unexpected(ConvertError{"cannot open " + p.string()});
    const std::streamsize n = f.tellg();
    if (n < 0) return std::unexpected(ConvertError{"tellg failed: " + p.string()});
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(n));
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(buf.data()), n);
    if (!f) return std::unexpected(ConvertError{"read failed: " + p.string()});
    return buf;
}

template <typename T>
[[nodiscard]] inline T read_le(const std::uint8_t* p) noexcept
{
    T v{};
    std::memcpy(&v, p, sizeof(T));
    return v;
}

// Write helpers for streaming output.
struct ByteWriter {
    std::vector<std::uint8_t> buf;

    template <typename T>
    void put(const T& v) {
        static_assert(std::is_trivially_copyable_v<T>,
                      "ByteWriter::put requires a trivially-copyable T");
        std::array<std::uint8_t, sizeof(T)> bytes{};
        std::memcpy(bytes.data(), &v, sizeof(T));
        buf.insert(buf.end(), bytes.begin(), bytes.end());
    }
    void put_bytes(std::span<const std::uint8_t> s) {
        buf.insert(buf.end(), s.begin(), s.end());
    }
};

// Convert in_path → out_path. Returns ConvertStats on success.
//
// CLI binary contract (matches the Rust tool, bit-for-bit):
//   * input must be v4 (TQ1) — v1/v2/v3 inputs are rejected.
//   * output is v3 with H1B_FLAG_SHERRY_FP16 set in cfg[8].
//   * H1B_FLAG_HADAMARD_ROTATED is preserved if present on input.
[[nodiscard]] inline std::expected<ConvertStats, ConvertError>
convert_file(const std::filesystem::path& in_path,
             const std::filesystem::path& out_path)
{
    auto buf_or = read_all(in_path);
    if (!buf_or) return std::unexpected(buf_or.error());
    const std::vector<std::uint8_t>& buf = *buf_or;

    if (buf.size() < 4 + 4 + 9 * 4) {
        return std::unexpected(ConvertError{"input too small"});
    }
    std::array<std::uint8_t, 4> magic{buf[0], buf[1], buf[2], buf[3]};
    if (magic != H1B_MAGIC) {
        return std::unexpected(ConvertError{"bad magic (want H1B\\0)"});
    }
    std::size_t off = 4;
    const std::int32_t version = read_le<std::int32_t>(buf.data() + off);
    off += 4;
    if (version != 4) {
        return std::unexpected(ConvertError{
            "h1b-sherry expects a v4 (TQ1) input file; got version "
            + std::to_string(version)});
    }
    if (off + 9 * 4 > buf.size()) {
        return std::unexpected(ConvertError{"short read: cfg"});
    }
    std::int32_t cfg[9];
    std::memcpy(cfg, buf.data() + off, sizeof(cfg));
    off += sizeof(cfg);
    const std::int32_t hs    = cfg[0];
    const std::int32_t is_   = cfg[1];
    const std::int32_t L     = cfg[2];
    const std::int32_t nh    = cfg[3];
    const std::int32_t nkv   = cfg[4];
    const std::int32_t vocab = cfg[5];
    if (hs <= 0 || is_ <= 0 || L <= 0 || nh <= 0 || nkv <= 0 || vocab <= 0) {
        return std::unexpected(ConvertError{"invalid cfg dims"});
    }
    if (nh == 0 || hs % nh != 0) {
        return std::unexpected(ConvertError{"hidden_size not divisible by num_heads"});
    }
    const std::int32_t hd = hs / nh;
    const std::int32_t reserved_in = cfg[8];

    // v4 always carries the v2-extras tail (rope+eps).
    if (off + 8 > buf.size()) {
        return std::unexpected(ConvertError{"short read: rope/eps"});
    }
    const float rope_theta = read_le<float>(buf.data() + off); off += 4;
    const float rms_eps    = read_le<float>(buf.data() + off); off += 4;

    // ── Build output config: v3, set SHERRY_FP16, preserve HADAMARD bit ──
    const std::int32_t preserved =
        reserved_in & H1B_FLAG_HADAMARD_ROTATED;
    std::int32_t cfg_out[9];
    std::memcpy(cfg_out, cfg, sizeof(cfg_out));
    cfg_out[8] = H1B_FLAG_SHERRY_FP16 | preserved;

    ByteWriter w;
    w.buf.reserve(buf.size());
    w.put_bytes(std::span<const std::uint8_t>{H1B_MAGIC.data(), 4});
    const std::int32_t v3 = 3;
    w.put(v3);
    for (std::int32_t v : cfg_out) w.put(v);
    w.put(rope_theta);
    w.put(rms_eps);

    // ── Embedding + final_norm: pass through as fp32 ──
    const std::size_t emb_bytes  = static_cast<std::size_t>(vocab) * hs * 4u;
    const std::size_t fnorm_bytes = static_cast<std::size_t>(hs) * 4u;
    if (off + emb_bytes + fnorm_bytes > buf.size()) {
        return std::unexpected(ConvertError{"short read: embedding/final_norm"});
    }
    w.put_bytes(std::span<const std::uint8_t>{buf.data() + off, emb_bytes});
    off += emb_bytes;
    w.put_bytes(std::span<const std::uint8_t>{buf.data() + off, fnorm_bytes});
    off += fnorm_bytes;

    // ── Per-layer norm block + 7 ternary tensors ──
    // Norm block layout (matches onebit-core::h1b::serialize):
    //   [input_norm hs][post_attn_norm hs][attn_sub_norm hs × 4]
    //   [trunc_ffn_sub hs × 2][ffn_sub_norm is]
    const std::size_t norm_block_bytes =
        static_cast<std::size_t>(hs) * 4u * (1u + 1u + 4u + 2u)
        + static_cast<std::size_t>(is_) * 4u;

    struct TensorSpec { const char* name; std::int32_t rows; std::int32_t cols; };
    auto layer_specs = [&](std::int32_t hs_, std::int32_t is_2,
                           std::int32_t nh_, std::int32_t nkv_,
                           std::int32_t hd_) -> std::array<TensorSpec, 7> {
        return {
            TensorSpec{"q",    nh_  * hd_, hs_},
            TensorSpec{"k",    nkv_ * hd_, hs_},
            TensorSpec{"v",    nkv_ * hd_, hs_},
            TensorSpec{"o",    hs_,        nh_ * hd_},
            TensorSpec{"gate", is_2,       hs_},
            TensorSpec{"up",   is_2,       hs_},
            TensorSpec{"down", hs_,        is_2},
        };
    };

    ConvertStats stats{};
    stats.layers_processed   = static_cast<std::uint32_t>(L);
    stats.hadamard_preserved = preserved != 0;

    // Sherry alignment.
    auto require32 = [&](const char* name, std::int32_t k)
        -> std::expected<void, ConvertError> {
        if (k % 32 != 0) {
            return std::unexpected(ConvertError{
                std::string{"tensor "} + name + ": cols=" + std::to_string(k)
                + " not divisible by 32 (Sherry requirement)"});
        }
        return {};
    };
    for (auto sp : layer_specs(hs, is_, nh, nkv, hd)) {
        if (auto r = require32(sp.name, sp.cols); !r) return std::unexpected(r.error());
    }

    std::vector<std::int8_t> ternary_buf;
    std::vector<std::uint8_t> sherry_dst;
    for (std::int32_t li = 0; li < L; ++li) {
        if (off + norm_block_bytes > buf.size()) {
            return std::unexpected(ConvertError{
                "short read: layer " + std::to_string(li) + " norms"});
        }
        w.put_bytes(std::span<const std::uint8_t>{buf.data() + off, norm_block_bytes});
        off += norm_block_bytes;

        for (auto sp : layer_specs(hs, is_, nh, nkv, hd)) {
            const std::size_t row_in_bytes  = tq1_row_bytes(static_cast<std::size_t>(sp.cols));
            const std::size_t row_out_bytes = sherry_row_bytes(static_cast<std::size_t>(sp.cols));
            const std::size_t packed_in     = static_cast<std::size_t>(sp.rows) * row_in_bytes;
            const std::size_t scales_bytes  = static_cast<std::size_t>(sp.rows) * 4u;
            const std::size_t packed_out    = static_cast<std::size_t>(sp.rows) * row_out_bytes;

            if (off + packed_in + scales_bytes > buf.size()) {
                return std::unexpected(ConvertError{
                    std::string{"short read: layer "} + std::to_string(li) + " tensor "
                    + sp.name});
            }
            const std::uint8_t* src    = buf.data() + off;
            const std::uint8_t* scales = src + packed_in;

            ternary_buf.assign(static_cast<std::size_t>(sp.cols), 0);
            sherry_dst.assign(packed_out, 0);
            for (std::int32_t r = 0; r < sp.rows; ++r) {
                const std::size_t in_start  = static_cast<std::size_t>(r) * row_in_bytes;
                const std::size_t out_start = static_cast<std::size_t>(r) * row_out_bytes;
                unpack_tq1_row(
                    std::span<const std::uint8_t>{src + in_start, row_in_bytes},
                    std::span<std::int8_t>{ternary_buf.data(), ternary_buf.size()},
                    static_cast<std::size_t>(sp.cols));
                auto rs = pack_sherry_row(
                    std::span<const std::int8_t>{ternary_buf.data(), ternary_buf.size()},
                    std::span<std::uint8_t>{sherry_dst.data() + out_start, row_out_bytes},
                    static_cast<std::size_t>(sp.cols));
                stats.groups_total      += static_cast<std::uint64_t>(sp.cols / 4);
                stats.forced_zero_flips += rs.forced_zero_flips;
                ++stats.rows_total;
            }
            // Emit packed (Sherry) + per-row scales (verbatim).
            w.put_bytes(std::span<const std::uint8_t>{sherry_dst.data(), packed_out});
            w.put_bytes(std::span<const std::uint8_t>{scales, scales_bytes});
            off += packed_in + scales_bytes;
        }
    }

    // Trailing bytes (untied LM head, etc.) — pass through.
    if (off < buf.size()) {
        w.put_bytes(std::span<const std::uint8_t>{
            buf.data() + off, buf.size() - off});
    }

    {
        std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
        if (!f) return std::unexpected(ConvertError{"cannot open output " + out_path.string()});
        f.write(reinterpret_cast<const char*>(w.buf.data()),
                static_cast<std::streamsize>(w.buf.size()));
        if (!f) return std::unexpected(ConvertError{"short write " + out_path.string()});
    }
    return stats;
}

} // namespace onebit::tools::h1b_sherry
