// h1b-sherry test suite (doctest, header-only target).
//
// Coverage (≥ 4 test cases):
//   1. pick_zero_pos / encode_group / decode_group basic invariants
//   2. pack_sherry_row -> unpack_sherry_row round-trip on a 3:4-sparse row
//   3. tq1_unpack basic decode + edge cases (oversize byte, padding cap)
//   4. convert_file end-to-end on a synthetic v4 .h1b (hadamard preserved,
//      flag composition, flip fraction within budget)
//   5. convert_file rejects v3 input
//   6. pack_sherry_row repairs multi-zero groups (Sherry regression bug #2):
//      groups with 2/3/4 zeros must round-trip non-zero values exactly,
//      with extras flipped to +1 and counted in `forced_zero_flips`.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/tools/h1b_sherry.hpp"
#include "onebit/tools/h1b_sherry_convert.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <system_error>
#include <vector>

using namespace onebit::tools::h1b_sherry;

namespace {

// ----- Helpers ---------------------------------------------------------------

std::filesystem::path tmp_dir()
{
    std::error_code ec;
    auto base = std::filesystem::temp_directory_path(ec) / "onebit_h1b_sherry_test";
    std::filesystem::create_directories(base, ec);
    return base;
}

// Build a structurally valid v4 (TQ1) .h1b in memory. Tiny dims, all
// `cols % 32 == 0`. Per-tensor ternary ground truth: every group of 4
// has one zero (already 3:4-sparse) except group 0 of every row in the
// "noisy" tensor (up).
//
// Returns (raw bytes, ground truth ternary per tensor).
struct V4Fixture {
    std::vector<std::uint8_t>            bytes;
    std::array<std::vector<std::int8_t>, 7> ternaries;
    std::int32_t hs    = 32;
    std::int32_t is_   = 64;
    std::int32_t L     = 1;
    std::int32_t nh    = 2;
    std::int32_t nkv   = 1;
    std::int32_t vocab = 4;
    std::int32_t hd    = 16;
};

std::array<std::array<std::int32_t, 2>, 7> shapes_of(const V4Fixture& fx)
{
    return {{
        {fx.nh * fx.hd, fx.hs},   // q
        {fx.nkv * fx.hd, fx.hs},  // k
        {fx.nkv * fx.hd, fx.hs},  // v
        {fx.hs, fx.nh * fx.hd},   // o
        {fx.is_, fx.hs},          // gate
        {fx.is_, fx.hs},          // up
        {fx.hs, fx.is_}           // down
    }};
}

std::vector<std::uint8_t>
pack_tq1_v4(std::span<const std::int8_t> ternary, std::int32_t rows, std::int32_t cols)
{
    const std::size_t row_bytes = tq1_row_bytes(static_cast<std::size_t>(cols));
    std::vector<std::uint8_t> out(static_cast<std::size_t>(rows) * row_bytes, 0);
    for (std::int32_t r = 0; r < rows; ++r) {
        for (std::size_t i = 0; i < row_bytes; ++i) {
            const std::size_t base = i * 5;
            std::uint32_t byte = 0;
            std::uint32_t mul  = 1;
            for (std::size_t d = 0; d < 5; ++d) {
                const std::size_t k = base + d;
                std::uint32_t digit = 1; // pad → ternary 0
                if (k < static_cast<std::size_t>(cols)) {
                    digit = static_cast<std::uint32_t>(
                        ternary[static_cast<std::size_t>(r) * cols + k] + 1);
                }
                byte += digit * mul;
                mul *= 3;
            }
            out[static_cast<std::size_t>(r) * row_bytes + i] = static_cast<std::uint8_t>(byte);
        }
    }
    return out;
}

V4Fixture build_v4_fixture(std::int32_t reserved)
{
    V4Fixture fx;
    auto shapes = shapes_of(fx);

    // Build per-tensor ternary ground truth.
    for (std::size_t ti = 0; ti < 7; ++ti) {
        const std::int32_t rows = shapes[ti][0];
        const std::int32_t cols = shapes[ti][1];
        std::vector<std::int8_t> t(static_cast<std::size_t>(rows) * cols, 0);
        for (std::int32_t r = 0; r < rows; ++r) {
            for (std::int32_t g = 0; g < cols / 4; ++g) {
                const std::size_t base = static_cast<std::size_t>(r) * cols + g * 4;
                if (g == 0) {
                    // Noisy first group: all ±1, no zeros.
                    t[base + 0] = 1;
                    t[base + 1] = -1;
                    t[base + 2] = 1;
                    t[base + 3] = -1;
                } else {
                    const std::int32_t zp = (r + g) % 4;
                    for (std::int32_t p = 0; p < 4; ++p) {
                        if (p == zp)              t[base + p] = 0;
                        else if (((g + p) & 1) == 0) t[base + p] = 1;
                        else                      t[base + p] = -1;
                    }
                }
            }
        }
        fx.ternaries[ti] = std::move(t);
    }

    // Build the bytestream.
    ByteWriter w;
    w.put_bytes(std::span<const std::uint8_t>{H1B_MAGIC.data(), 4});
    const std::int32_t version = 4;
    w.put(version);
    const std::int32_t cfg[9] = {
        fx.hs, fx.is_, fx.L, fx.nh, fx.nkv, fx.vocab,
        /*max_seq_len=*/32, /*tie_embeddings=*/1, reserved};
    for (auto v : cfg) w.put(v);
    w.put<float>(500'000.0f);  // rope_theta
    w.put<float>(1e-5f);       // rms_norm_eps

    // Embedding (vocab × hs) + final_norm (hs) — fp32 placeholders.
    std::vector<std::uint8_t> embedding(static_cast<std::size_t>(fx.vocab * fx.hs * 4), 0xCC);
    std::vector<std::uint8_t> final_norm(static_cast<std::size_t>(fx.hs * 4), 0xDD);
    w.put_bytes(embedding);
    w.put_bytes(final_norm);

    // Per-layer: norm block + 7 ternary tensors + per-row scales.
    for (std::int32_t li = 0; li < fx.L; ++li) {
        // Norm block: hs*4 * (1+1+4+2) + is*4
        std::vector<std::uint8_t> norms(
            static_cast<std::size_t>(fx.hs) * 4 * (1 + 1 + 4 + 2)
            + static_cast<std::size_t>(fx.is_) * 4,
            0xEE);
        w.put_bytes(norms);
        for (std::size_t ti = 0; ti < 7; ++ti) {
            const std::int32_t rows = shapes[ti][0];
            const std::int32_t cols = shapes[ti][1];
            auto packed = pack_tq1_v4(fx.ternaries[ti], rows, cols);
            w.put_bytes(packed);
            std::vector<std::uint8_t> scales(static_cast<std::size_t>(rows) * 4, 0x7F);
            w.put_bytes(scales);
        }
    }
    fx.bytes = std::move(w.buf);
    return fx;
}

void write_file(const std::filesystem::path& p, std::span<const std::uint8_t> b)
{
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    REQUIRE(f.is_open());
    f.write(reinterpret_cast<const char*>(b.data()),
            static_cast<std::streamsize>(b.size()));
    REQUIRE(static_cast<bool>(f));
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& p)
{
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    REQUIRE(f.is_open());
    const std::streamsize n = f.tellg();
    REQUIRE(n >= 0);
    std::vector<std::uint8_t> b(static_cast<std::size_t>(n));
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(b.data()), n);
    return b;
}

} // anonymous

// ============================================================================

TEST_CASE("h1b-sherry: pick_zero_pos / encode_group basics")
{
    CHECK(pick_zero_pos({1, -1, 0, 1}) == 2);
    CHECK(pick_zero_pos({0, -1, 0, 1}) == 0);   // ties → lowest index
    CHECK(pick_zero_pos({1, -1, 1, -1}) == 0);  // all ±1 → lowest

    // Group [+1, 0, -1, +1], zero_pos=1 → surviving lanes [+1, -1, +1]
    // sign bits LSB-first: [1, 0, 1] = 0b101 = 5 ; code = (1<<3)|5 = 13.
    CHECK(encode_group({1, 0, -1, 1}, 1) == 0b01'101u);
}

TEST_CASE("h1b-sherry: pack/unpack round-trip on 3:4-sparse row")
{
    // 32 cols = 8 groups; place exactly one zero per group; vary signs.
    std::vector<std::int8_t> ternary(32, 0);
    for (std::size_t g = 0; g < 8; ++g) {
        const std::size_t base = g * 4;
        ternary[base + (g % 4)] = 0;
        const std::int8_t nz[3] = {1, -1, 1};
        std::size_t idx = 0;
        for (std::size_t p = 0; p < 4; ++p) {
            if (p != g % 4) {
                ternary[base + p] = nz[idx % 3];
                ++idx;
            }
        }
    }

    std::vector<std::uint8_t> packed(sherry_row_bytes(32), 0);
    auto stats = pack_sherry_row(
        std::span<const std::int8_t>{ternary.data(), ternary.size()},
        std::span<std::uint8_t>{packed.data(), packed.size()},
        32);
    CHECK(stats.forced_zero_flips == 0u);

    std::vector<std::int8_t> unpacked(32, 99);
    unpack_sherry_row(
        std::span<const std::uint8_t>{packed.data(), packed.size()},
        std::span<std::int8_t>{unpacked.data(), unpacked.size()},
        32);
    CHECK(unpacked == ternary);
}

TEST_CASE("h1b-sherry: tq1_unpack base-3 + edge cases")
{
    // byte = 2 + 1*9 + 2*27 = 65 → digits [2,0,1,2,0] → ternary [+1,-1,0,+1,-1]
    std::array<std::uint8_t, 4> row{65, 0, 0, 0};
    std::array<std::int8_t, 20> out{};
    unpack_tq1_row(row, out, 20);
    CHECK(out[0] == 1);
    CHECK(out[1] == -1);
    CHECK(out[2] == 0);
    CHECK(out[3] == 1);
    CHECK(out[4] == -1);
    // Three trailing zero bytes → 5 × 0 each → ternary -1 each.
    for (std::size_t i = 5; i < 20; ++i) CHECK(out[i] == -1);

    // Oversize byte (>=243) → kernel LUT zero-fill.
    std::array<std::uint8_t, 4> bad{250, 0, 0, 0};
    std::array<std::int8_t, 20> outb{};
    unpack_tq1_row(bad, outb, 20);
    for (std::size_t i = 0; i < 5; ++i)  CHECK(outb[i] == 0);
    for (std::size_t i = 5; i < 20; ++i) CHECK(outb[i] == -1);

    // Padding-respect: cols=18 with 4 packed bytes still works.
    std::array<std::uint8_t, 4> r0{0, 0, 0, 0};
    std::array<std::int8_t, 18> out18{};
    unpack_tq1_row(r0, out18, 18);
    for (auto v : out18) CHECK(v == -1);
}

TEST_CASE("h1b-sherry: convert_file end-to-end + flag composition")
{
    auto base = tmp_dir();
    // Run twice: once with HADAMARD bit set, once without — flag must
    // propagate while SHERRY_FP16 is unconditionally set.
    for (std::int32_t in_flag : {0, H1B_FLAG_HADAMARD_ROTATED}) {
        auto fx = build_v4_fixture(in_flag);
        const auto in_path  = base / ("src_" + std::to_string(in_flag) + ".h1b");
        const auto out_path = base / ("dst_" + std::to_string(in_flag) + ".h1b");
        write_file(in_path, fx.bytes);

        auto r = convert_file(in_path, out_path);
        REQUIRE(r.has_value());
        const ConvertStats& s = *r;

        // Noisy "up" tensor contributes (is_*hs/4) = 64*32/4 = 512 forced
        // flips (1 per row in g=0). Other 6 tensors contribute zero.
        // Across the layer the total fraction stays ≤ 12%.
        CHECK(s.flip_fraction() <= 0.12);
        CHECK((in_flag != 0) == s.hadamard_preserved);

        // Re-parse output: version field at byte 4, reserved at byte 8 + 8*4 = 40.
        auto buf = read_file(out_path);
        REQUIRE(buf.size() >= 4 + 4 + 9 * 4 + 8);
        CHECK(buf[0] == 'H');
        CHECK(buf[1] == '1');
        CHECK(buf[2] == 'B');
        CHECK(buf[3] == 0u);
        const std::int32_t version = read_le<std::int32_t>(buf.data() + 4);
        CHECK(version == 3);
        const std::int32_t reserved =
            read_le<std::int32_t>(buf.data() + 4 + 4 + 8 * 4);
        CHECK((reserved & H1B_FLAG_SHERRY_FP16) != 0);
        CHECK(((reserved & H1B_FLAG_HADAMARD_ROTATED) != 0) == (in_flag != 0));
    }
}

TEST_CASE("h1b-sherry: pack_sherry_row repairs multi-zero groups")
{
    // Sherry regression bug #2 (2026-04-26): groups with >=2 zeros used to
    // be silently corrupted (extras decoded as +1) WITHOUT being counted
    // in `forced_zero_flips`.  Post-fix: extras are deterministically
    // flipped to +1 *and* counted, and every NON-zero lane survives the
    // round-trip exactly.

    auto round_trip = [](const std::vector<std::int8_t>& in)
        -> std::pair<std::vector<std::int8_t>, std::uint32_t> {
        std::vector<std::uint8_t> packed(sherry_row_bytes(in.size()), 0);
        auto stats = pack_sherry_row(
            std::span<const std::int8_t>{in.data(), in.size()},
            std::span<std::uint8_t>{packed.data(), packed.size()},
            in.size());
        std::vector<std::int8_t> out(in.size(), 99);
        unpack_sherry_row(
            std::span<const std::uint8_t>{packed.data(), packed.size()},
            std::span<std::int8_t>{out.data(), out.size()},
            in.size());
        return {std::move(out), stats.forced_zero_flips};
    };

    // ── 8 groups, 2 zeros each ──
    // Pattern: {0, 0, +1, -1}.  pick_zero_pos picks lane 0 (lowest-index
    // zero); lane 1 is the extra zero → forced to +1.  Lanes 2, 3 survive
    // exactly.  Expected: 1 flip per group × 8 = 8 total.
    {
        std::vector<std::int8_t> in(32, 0);
        for (std::size_t g = 0; g < 8; ++g) {
            const std::size_t b = g * 4;
            in[b + 0] = 0; in[b + 1] = 0; in[b + 2] = 1; in[b + 3] = -1;
        }
        auto [out, flips] = round_trip(in);
        CHECK(flips == 8u);
        for (std::size_t g = 0; g < 8; ++g) {
            const std::size_t b = g * 4;
            CHECK(out[b + 0] == 0);   // chosen zero_pos (preserved)
            CHECK(out[b + 1] == 1);   // extra zero → +1
            CHECK(out[b + 2] == 1);   // survives exactly
            CHECK(out[b + 3] == -1);  // survives exactly
        }
    }

    // ── 8 groups, 3 zeros each ──
    // Pattern: {0, 0, 0, -1}.  pick_zero_pos = 0; lanes 1, 2 forced to +1;
    // lane 3 survives.  Expected: 2 flips per group × 8 = 16.
    {
        std::vector<std::int8_t> in(32, 0);
        for (std::size_t g = 0; g < 8; ++g) {
            const std::size_t b = g * 4;
            in[b + 0] = 0; in[b + 1] = 0; in[b + 2] = 0; in[b + 3] = -1;
        }
        auto [out, flips] = round_trip(in);
        CHECK(flips == 16u);
        for (std::size_t g = 0; g < 8; ++g) {
            const std::size_t b = g * 4;
            CHECK(out[b + 0] == 0);
            CHECK(out[b + 1] == 1);
            CHECK(out[b + 2] == 1);
            CHECK(out[b + 3] == -1);
        }
    }

    // ── 8 groups, 4 zeros each (degenerate all-zero row) ──
    // pick_zero_pos = 0; lanes 1, 2, 3 forced to +1.  Expected: 3 flips
    // per group × 8 = 24.  This is the worst case and confirms encoder
    // never emits a non-zp zero.
    {
        std::vector<std::int8_t> in(32, 0);
        auto [out, flips] = round_trip(in);
        CHECK(flips == 24u);
        for (std::size_t g = 0; g < 8; ++g) {
            const std::size_t b = g * 4;
            CHECK(out[b + 0] == 0);
            CHECK(out[b + 1] == 1);
            CHECK(out[b + 2] == 1);
            CHECK(out[b + 3] == 1);
        }
    }

    // ── Mixed-density row: every group has at least one zero, so the
    // chosen zero_pos always lands on a 0 lane and every NON-zero input
    // lane survives the round-trip exactly.  Confirms zero-count handling
    // is per-group and that signs are stable across density variation.
    {
        std::vector<std::int8_t> in = {
            // g0: 1 zero  (lane 1)       → 0 flips
             1,  0, -1,  1,
            // g1: 2 zeros (lanes 1,2)    → 1 flip  (lane 2 → +1)
            -1,  0,  0,  1,
            // g2: 1 zero  (lane 2)       → 0 flips
             1, -1,  0,  1,
            // g3: 3 zeros (lanes 0,2,3)  → 2 flips (lanes 2,3 → +1)
             0, -1,  0,  0,
            // g4: 1 zero  (lane 3)       → 0 flips
             1, -1,  1,  0,
            // g5: 4 zeros (degenerate)   → 3 flips (lanes 1,2,3 → +1)
             0,  0,  0,  0,
            // g6: 1 zero  (lane 0)       → 0 flips
             0,  1, -1,  1,
            // g7: 2 zeros (lanes 0,3)    → 1 flip  (lane 3 → +1)
             0,  1, -1,  0,
        };
        REQUIRE(in.size() == 32u);
        auto [out, flips] = round_trip(in);
        CHECK(flips == (0u + 1u + 0u + 2u + 0u + 3u + 0u + 1u));
        // Bit-exact survival: every NON-zero input lane round-trips
        // unchanged.  (Holds because every group above has >=1 zero, so
        // pick_zero_pos lands on an input zero and never clobbers a ±1.)
        for (std::size_t i = 0; i < in.size(); ++i) {
            if (in[i] != 0) CHECK(out[i] == in[i]);
        }
        // Decoder never emits a 0 outside the chosen zero_pos.  Each group
        // has exactly one 0 in the output.
        for (std::size_t g = 0; g < 8; ++g) {
            int zero_count = 0;
            for (std::size_t p = 0; p < 4; ++p) {
                if (out[g * 4 + p] == 0) ++zero_count;
            }
            CHECK(zero_count == 1);
        }
    }
}

TEST_CASE("h1b-sherry: convert_file rejects non-v4 input")
{
    auto base = tmp_dir();
    // Build a v4 fixture, then patch the version field down to 3 to fake
    // an already-Sherry input.
    auto fx = build_v4_fixture(0);
    fx.bytes[4] = 3;
    fx.bytes[5] = 0;
    fx.bytes[6] = 0;
    fx.bytes[7] = 0;
    const auto in_path  = base / "src_v3.h1b";
    const auto out_path = base / "dst_v3.h1b";
    write_file(in_path, fx.bytes);
    auto r = convert_file(in_path, out_path);
    CHECK_FALSE(r.has_value());
}
