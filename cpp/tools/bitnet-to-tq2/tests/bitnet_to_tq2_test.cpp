// bitnet-to-tq2 test suite (doctest, header-only target).
//
// ≥ 4 cases:
//   1. quantize/dequantize round-trip preserves sign + zero positions
//   2. all-zero scale path emits decoder-zero codes (0b01)
//   3. per-tensor scale duplicated across blocks; per-block varies
//   4. header flag composition: H1B_FLAG_BONSAI_TQ2 set, version=4, magic ok
//   5. zero-row (all weights zero) produces zero scale and decoder-zero codes
//   6. odd shape rejection (cols not multiple of TQ2_GROUP_SIZE)

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/tools/bitnet_to_tq2.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <span>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

using namespace onebit::tools::bitnet_to_tq2;

namespace {

std::filesystem::path tmp_dir(const std::string& tag)
{
    std::error_code ec;
    auto base = std::filesystem::temp_directory_path(ec)
              / ("onebit_bitnet_to_tq2_" + tag);
    std::filesystem::remove_all(base, ec);
    std::filesystem::create_directories(base, ec);
    return base;
}

std::uint16_t f32_to_bf16(float v) noexcept
{
    std::uint32_t bits;
    std::memcpy(&bits, &v, 4);
    // Round-to-nearest-even for the lower 16 bits.
    const std::uint32_t lsb     = (bits >> 16) & 1u;
    const std::uint32_t rounder = 0x7FFFu + lsb;
    bits += rounder;
    return static_cast<std::uint16_t>(bits >> 16);
}

void append_bf16(std::vector<std::uint8_t>& dst, float v)
{
    const std::uint16_t b = f32_to_bf16(v);
    dst.push_back(static_cast<std::uint8_t>(b & 0xFFu));
    dst.push_back(static_cast<std::uint8_t>((b >> 8) & 0xFFu));
}

// Build a single-tensor-set safetensors v0 file. Tensors are bf16.
std::vector<std::uint8_t> build_safetensors(
    const std::vector<std::tuple<std::string,
                                 std::vector<std::size_t>,
                                 std::vector<std::uint8_t>>>& items)
{
    nlohmann::json hdr = nlohmann::json::object();
    std::size_t cursor = 0;
    for (const auto& [name, shape, bytes] : items) {
        nlohmann::json e = nlohmann::json::object();
        e["dtype"] = "BF16";
        e["shape"] = nlohmann::json::array();
        for (auto d : shape) e["shape"].push_back(d);
        e["data_offsets"] = nlohmann::json::array({cursor, cursor + bytes.size()});
        hdr[name] = std::move(e);
        cursor += bytes.size();
    }
    std::string hdr_str = hdr.dump();
    // Pad header to 8-byte alignment (safetensors recommends but tolerant).
    while ((hdr_str.size() % 8) != 0) hdr_str.push_back(' ');

    std::vector<std::uint8_t> out(8 + hdr_str.size() + cursor, 0);
    const std::uint64_t hdr_len = hdr_str.size();
    std::memcpy(out.data(), &hdr_len, 8);
    std::memcpy(out.data() + 8, hdr_str.data(), hdr_str.size());
    std::size_t off = 8 + hdr_str.size();
    for (const auto& [name, shape, bytes] : items) {
        std::memcpy(out.data() + off, bytes.data(), bytes.size());
        off += bytes.size();
    }
    return out;
}

std::vector<std::uint8_t> ternary_bf16(std::size_t rows, std::size_t cols, std::uint32_t seed)
{
    std::vector<std::uint8_t> out;
    out.reserve(rows * cols * 2);
    std::uint32_t s = seed;
    for (std::size_t i = 0; i < rows * cols; ++i) {
        s = s * 1664525u + 1013904223u;
        const float v = (s % 3 == 0) ? 0.3f : (s % 3 == 1) ? -0.3f : 0.0f;
        append_bf16(out, v);
    }
    return out;
}

std::vector<std::uint8_t> uniform_bf16(std::size_t n, float v)
{
    std::vector<std::uint8_t> out;
    out.reserve(n * 2);
    for (std::size_t i = 0; i < n; ++i) append_bf16(out, v);
    return out;
}

void write_file(const std::filesystem::path& p, std::span<const std::uint8_t> b)
{
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    REQUIRE(f.is_open());
    f.write(reinterpret_cast<const char*>(b.data()),
            static_cast<std::streamsize>(b.size()));
    REQUIRE(static_cast<bool>(f));
}

// One-layer fixture: hidden=hs, intermediate=is_, heads=nh, kv_heads=nkv,
// vocab=32, ctx=64. cols of every ternary tensor must be a multiple of 128.
struct Fixture {
    std::filesystem::path input_dir;
    std::size_t           hs       = 256;
    std::size_t           is_      = 512;
    std::size_t           nh       = 4;
    std::size_t           nkv      = 1;
    std::size_t           vocab    = 32;
    std::size_t           head_dim = 64;
};

Fixture make_fixture(const std::string& tag, std::size_t hs = 256, std::size_t is_ = 512,
                     std::size_t nh = 4, std::size_t nkv = 1)
{
    Fixture fx;
    fx.input_dir = tmp_dir(tag);
    fx.hs        = hs;
    fx.is_       = is_;
    fx.nh        = nh;
    fx.nkv       = nkv;
    fx.head_dim  = hs / nh;

    nlohmann::json cfg = {
        {"hidden_size",             fx.hs},
        {"intermediate_size",       fx.is_},
        {"num_hidden_layers",       1},
        {"num_attention_heads",     fx.nh},
        {"num_key_value_heads",     fx.nkv},
        {"vocab_size",              fx.vocab},
        {"max_position_embeddings", 64},
        {"tie_word_embeddings",     true},
        {"rope_theta",              10000.0},
        {"rms_norm_eps",            1e-5}
    };
    {
        std::ofstream f(fx.input_dir / "config.json");
        f << cfg.dump();
    }

    const std::size_t q_rows  = fx.nh  * fx.head_dim;
    const std::size_t kv_rows = fx.nkv * fx.head_dim;

    auto items = std::vector<std::tuple<std::string,
                                        std::vector<std::size_t>,
                                        std::vector<std::uint8_t>>>{
        {"model.embed_tokens.weight",                                  {fx.vocab, fx.hs}, uniform_bf16(fx.vocab * fx.hs, 0.0f)},
        {"model.norm.weight",                                          {fx.hs},           uniform_bf16(fx.hs, 1.0f)},
        {"model.layers.0.input_layernorm.weight",                      {fx.hs},           uniform_bf16(fx.hs, 1.0f)},
        {"model.layers.0.post_attention_layernorm.weight",             {fx.hs},           uniform_bf16(fx.hs, 1.0f)},
        {"model.layers.0.self_attn.attn_sub_norm.weight",              {fx.hs},           uniform_bf16(fx.hs, 1.0f)},
        {"model.layers.0.mlp.ffn_sub_norm.weight",                     {fx.is_},          uniform_bf16(fx.is_, 1.0f)},
        {"model.layers.0.self_attn.q_proj.weight",                     {q_rows, fx.hs},   ternary_bf16(q_rows, fx.hs, 1)},
        {"model.layers.0.self_attn.k_proj.weight",                     {kv_rows, fx.hs},  ternary_bf16(kv_rows, fx.hs, 2)},
        {"model.layers.0.self_attn.v_proj.weight",                     {kv_rows, fx.hs},  ternary_bf16(kv_rows, fx.hs, 3)},
        {"model.layers.0.self_attn.o_proj.weight",                     {fx.hs, q_rows},   ternary_bf16(fx.hs, q_rows, 4)},
        {"model.layers.0.mlp.gate_proj.weight",                        {fx.is_, fx.hs},   ternary_bf16(fx.is_, fx.hs, 5)},
        {"model.layers.0.mlp.up_proj.weight",                          {fx.is_, fx.hs},   ternary_bf16(fx.is_, fx.hs, 6)},
        {"model.layers.0.mlp.down_proj.weight",                        {fx.hs, fx.is_},   ternary_bf16(fx.hs, fx.is_, 7)},
    };
    auto bytes = build_safetensors(items);
    write_file(fx.input_dir / "model.safetensors", bytes);
    return fx;
}

} // anonymous

// ============================================================================

TEST_CASE("bitnet-to-tq2: quantize_group round-trip preserves signs + scale")
{
    std::array<float, TQ2_GROUP_SIZE> w{};
    for (std::size_t i = 0; i < TQ2_GROUP_SIZE; ++i) {
        w[i] = (i % 3 == 0) ? 0.3f : (i % 3 == 1) ? -0.3f : 0.0f;
    }
    auto block = quantize_group_tq2(w);
    auto out   = dequantize_group_tq2(block);

    // d should approximate the absmean (~0.2). Read back from block bytes.
    const std::uint16_t bits = static_cast<std::uint16_t>(
        block[0] | (block[1] << 8));
    const float d = f16_bits_to_f32(bits);
    double absmean_expected = 0.0;
    for (float x : w) absmean_expected += x < 0 ? -x : x;
    absmean_expected /= TQ2_GROUP_SIZE;
    CHECK(std::abs(d - static_cast<float>(absmean_expected)) < 1e-3f);

    // Sign preservation.
    for (std::size_t i = 0; i < TQ2_GROUP_SIZE; ++i) {
        if (w[i] > 0.0f)      CHECK(out[i] > 0.0f);
        else if (w[i] < 0.0f) CHECK(out[i] < 0.0f);
        else                  CHECK(out[i] == 0.0f);
    }
}

TEST_CASE("bitnet-to-tq2: all-zero group emits decoder-zero codes")
{
    std::array<float, TQ2_GROUP_SIZE> w{};
    auto block = quantize_group_tq2(w);
    // d=0 path: codes all 0b01 (= 0x55 per byte).
    for (std::size_t i = 2; i < TQ2_BLOCK_BYTES; ++i) {
        CHECK(block[i] == 0x55);
    }
    auto out = dequantize_group_tq2(block);
    for (float v : out) CHECK(v == 0.0f);
}

TEST_CASE("bitnet-to-tq2: header flag composition + ternary payload non-empty")
{
    auto fx = make_fixture("flag");
    const auto out_path = fx.input_dir / "out.h1b";
    auto r = convert(fx.input_dir, out_path);
    REQUIRE(r.has_value());
    const auto& s = *r;
    CHECK(s.h1b_reserved_flags == H1B_FLAG_BONSAI_TQ2);
    auto rf = read_reserved_flags(out_path);
    REQUIRE(rf.has_value());
    CHECK(*rf == H1B_FLAG_BONSAI_TQ2);
    CHECK(s.packed_ternary_bytes > 0);

    // Read header bytes to verify magic + version.
    std::ifstream f(out_path, std::ios::binary);
    std::array<std::uint8_t, 8> head{};
    f.read(reinterpret_cast<char*>(head.data()), head.size());
    CHECK(head[0] == 'H');
    CHECK(head[1] == '1');
    CHECK(head[2] == 'B');
    CHECK(head[3] == 0u);
    std::int32_t ver = 0;
    std::memcpy(&ver, head.data() + 4, 4);
    CHECK(ver == 4);
}

TEST_CASE("bitnet-to-tq2: per-tensor scale duplicated across blocks; per-block varies")
{
    auto fx = make_fixture("scale");
    const auto pt = fx.input_dir / "pt.h1b";
    const auto pb = fx.input_dir / "pb.h1b";
    auto rpt = convert_with_mode(fx.input_dir, pt, ScaleMode::PerTensor);
    auto rpb = convert_with_mode(fx.input_dir, pb, ScaleMode::PerBlock);
    REQUIRE(rpt.has_value());
    REQUIRE(rpb.has_value());
    CHECK(std::filesystem::file_size(pt) == std::filesystem::file_size(pb));
    CHECK(rpt->packed_ternary_bytes == rpb->packed_ternary_bytes);

    // Header layout: magic+version+9*i32+2*f32 = 8 + 36 + 8 = 52.
    const std::size_t header   = 52;
    const std::size_t embed    = fx.vocab * fx.hs * 4;
    const std::size_t fnorm    = fx.hs * 4;
    const std::size_t norms    =
          fx.hs * 4              // input_norm
        + fx.hs * 4              // post_attn_norm
        + fx.hs * 4 * 4          // attn_sub_norm × 4
        + fx.hs * 4 * 2          // trunc ffn × 2
        + fx.is_ * 4;            // ffn_sub_norm
    const std::size_t ternary_start = header + embed + fnorm + norms;

    // q_proj: rows = nh * head_dim = fx.hs (since nh*head_dim==hs), cols = hs.
    const std::size_t q_rows           = fx.nh * fx.head_dim;
    const std::size_t n_blocks_per_row = fx.hs / TQ2_GROUP_SIZE;
    const std::size_t q_blocks         = q_rows * n_blocks_per_row;

    auto read_all_bytes = [](const std::filesystem::path& p) {
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        const std::streamsize n = f.tellg();
        std::vector<std::uint8_t> b(static_cast<std::size_t>(n));
        f.seekg(0, std::ios::beg);
        f.read(reinterpret_cast<char*>(b.data()), n);
        return b;
    };
    auto pt_bytes = read_all_bytes(pt);
    auto pb_bytes = read_all_bytes(pb);

    std::vector<std::uint16_t> pt_d, pb_d;
    pt_d.reserve(q_blocks);
    pb_d.reserve(q_blocks);
    for (std::size_t b = 0; b < q_blocks; ++b) {
        const std::size_t off = ternary_start + b * TQ2_BLOCK_BYTES;
        pt_d.push_back(static_cast<std::uint16_t>(
            pt_bytes[off] | (pt_bytes[off + 1] << 8)));
        pb_d.push_back(static_cast<std::uint16_t>(
            pb_bytes[off] | (pb_bytes[off + 1] << 8)));
    }
    // Per-tensor: every d byte-identical.
    const std::uint16_t first = pt_d[0];
    for (auto d : pt_d) CHECK(d == first);
    // First scale must be in the expected range (~0.2 from ternary × 0.3).
    const float d_f32 = f16_bits_to_f32(first);
    CHECK(d_f32 > 0.15f);
    CHECK(d_f32 < 0.25f);
    // Per-block: at least one block differs (synthetic seeded data is varied).
    bool any_diff = false;
    for (auto d : pb_d) if (d != pb_d[0]) { any_diff = true; break; }
    CHECK(any_diff);
}

TEST_CASE("bitnet-to-tq2: misaligned cols rejected")
{
    // Build a fixture but keep a tensor whose cols isn't a multiple of 128.
    // We force this by using hs=120 (fail at q_proj) — needs hs % nh == 0
    // so use nh=4, hs=120, head_dim=30, q_rows=120.
    auto base = tmp_dir("misaligned");
    nlohmann::json cfg = {
        {"hidden_size",             120},
        {"intermediate_size",       128},
        {"num_hidden_layers",       1},
        {"num_attention_heads",     4},
        {"num_key_value_heads",     1},
        {"vocab_size",              16},
        {"max_position_embeddings", 32},
        {"tie_word_embeddings",     true},
        {"rope_theta",              10000.0},
        {"rms_norm_eps",            1e-5}
    };
    {
        std::ofstream f(base / "config.json");
        f << cfg.dump();
    }
    const std::size_t hs       = 120;
    const std::size_t is_      = 128;
    const std::size_t nh       = 4;
    const std::size_t nkv      = 1;
    const std::size_t head_dim = hs / nh;  // 30
    const std::size_t q_rows   = nh * head_dim;
    const std::size_t kv_rows  = nkv * head_dim;
    auto items = std::vector<std::tuple<std::string,
                                        std::vector<std::size_t>,
                                        std::vector<std::uint8_t>>>{
        {"model.embed_tokens.weight",                       {16, hs},   uniform_bf16(16 * hs, 0.0f)},
        {"model.norm.weight",                               {hs},       uniform_bf16(hs, 1.0f)},
        {"model.layers.0.input_layernorm.weight",           {hs},       uniform_bf16(hs, 1.0f)},
        {"model.layers.0.post_attention_layernorm.weight",  {hs},       uniform_bf16(hs, 1.0f)},
        {"model.layers.0.self_attn.attn_sub_norm.weight",   {hs},       uniform_bf16(hs, 1.0f)},
        {"model.layers.0.mlp.ffn_sub_norm.weight",          {is_},      uniform_bf16(is_, 1.0f)},
        {"model.layers.0.self_attn.q_proj.weight",          {q_rows, hs},  ternary_bf16(q_rows, hs, 1)},
        {"model.layers.0.self_attn.k_proj.weight",          {kv_rows, hs}, ternary_bf16(kv_rows, hs, 2)},
        {"model.layers.0.self_attn.v_proj.weight",          {kv_rows, hs}, ternary_bf16(kv_rows, hs, 3)},
        {"model.layers.0.self_attn.o_proj.weight",          {hs, q_rows},  ternary_bf16(hs, q_rows, 4)},
        {"model.layers.0.mlp.gate_proj.weight",             {is_, hs},     ternary_bf16(is_, hs, 5)},
        {"model.layers.0.mlp.up_proj.weight",               {is_, hs},     ternary_bf16(is_, hs, 6)},
        {"model.layers.0.mlp.down_proj.weight",             {hs, is_},     ternary_bf16(hs, is_, 7)},
    };
    auto bytes = build_safetensors(items);
    write_file(base / "model.safetensors", bytes);

    auto r = convert(base, base / "out.h1b");
    CHECK_FALSE(r.has_value());
}
