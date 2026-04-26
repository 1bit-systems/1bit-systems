// gguf-to-h1b test suite (doctest, header-only target).
//
// ≥ 4 cases:
//   1. Bonsai dtype tag round-trip + block byte sizes
//   2. is_gguf_magic + read_gguf_version on a synth tiny GGUF
//   3. .htok serializer round-trip (Rust HtokFile format) — header field
//      positions match
//   4. convert_file rejects a non-Bonsai GGUF (architecture != qwen3)
//   5. build_htok_blob skips merges with unresolvable sides

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/tools/gguf_to_h1b.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace onebit::tools::gguf_to_h1b;

namespace {

std::filesystem::path tmp_dir(const std::string& tag)
{
    std::error_code ec;
    auto base = std::filesystem::temp_directory_path(ec)
              / ("onebit_gguf_to_h1b_" + tag);
    std::filesystem::remove_all(base, ec);
    std::filesystem::create_directories(base, ec);
    return base;
}

// Hand-written tiny GGUF v3 builder. Only covers the fields we exercise.
class GgufBuilder {
public:
    void put_u32(std::uint32_t v) { put_le(v); }
    void put_u64(std::uint64_t v) { put_le(v); }
    void put_f32(float v)         { put_le(v); }
    void put_str(std::string_view s)
    {
        put_u64(s.size());
        b_.insert(b_.end(), s.begin(), s.end());
    }
    void put_bytes(std::span<const std::uint8_t> s)
    {
        b_.insert(b_.end(), s.begin(), s.end());
    }

    void kv_string(std::string_view key, std::string_view val)
    {
        put_str(key);
        put_u32(8); // String type tag
        put_str(val);
        ++kv_count_;
    }
    void kv_u32(std::string_view key, std::uint32_t val)
    {
        put_str(key);
        put_u32(4); // UInt32 type tag
        put_u32(val);
        ++kv_count_;
    }
    void kv_f32(std::string_view key, float val)
    {
        put_str(key);
        put_u32(6); // Float32 type tag
        put_f32(val);
        ++kv_count_;
    }
    void kv_string_array(std::string_view key,
                         const std::vector<std::string>& items)
    {
        put_str(key);
        put_u32(9); // Array type tag
        put_u32(8); // Element type: String
        put_u64(items.size());
        for (const auto& s : items) put_str(s);
        ++kv_count_;
    }
    void tensor_info(std::string_view name,
                     const std::vector<std::uint64_t>& dims,
                     std::uint32_t dtype, std::uint64_t offset)
    {
        put_str(name);
        put_u32(static_cast<std::uint32_t>(dims.size()));
        for (auto d : dims) put_u64(d);
        put_u32(dtype);
        put_u64(offset);
        ++tensor_count_;
    }

    std::vector<std::uint8_t> finish_no_data(std::uint64_t alignment = 32)
    {
        std::vector<std::uint8_t> out;
        out.insert(out.end(), {'G', 'G', 'U', 'F'});
        auto append_le = [&](auto x) {
            std::array<std::uint8_t, sizeof(x)> tmp{};
            std::memcpy(tmp.data(), &x, sizeof(x));
            out.insert(out.end(), tmp.begin(), tmp.end());
        };
        append_le(std::uint32_t{3});
        append_le(static_cast<std::uint64_t>(tensor_count_));
        append_le(static_cast<std::uint64_t>(kv_count_));
        out.insert(out.end(), b_.begin(), b_.end());
        while ((out.size() % alignment) != 0u) out.push_back(0u);
        return out;
    }

private:
    std::vector<std::uint8_t> b_;
    std::uint64_t             kv_count_     = 0;
    std::uint64_t             tensor_count_ = 0;

    template <typename T>
    void put_le(T v)
    {
        std::array<std::uint8_t, sizeof(T)> bytes{};
        std::memcpy(bytes.data(), &v, sizeof(T));
        b_.insert(b_.end(), bytes.begin(), bytes.end());
    }
};

void write_file(const std::filesystem::path& p, std::span<const std::uint8_t> b)
{
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    REQUIRE(f.is_open());
    f.write(reinterpret_cast<const char*>(b.data()),
            static_cast<std::streamsize>(b.size()));
    REQUIRE(static_cast<bool>(f));
}

} // anonymous

// ============================================================================

TEST_CASE("gguf-to-h1b: Bonsai dtype constants")
{
    auto q = bonsai_from_u32(41);
    REQUIRE(q.has_value());
    CHECK(*q == BonsaiDtype::Q1G128);
    CHECK(bonsai_block_bytes(*q) == 18);
    CHECK(bonsai_h1b_flag(*q) == H1B_FLAG_BONSAI_Q1);

    auto t = bonsai_from_u32(42);
    REQUIRE(t.has_value());
    CHECK(*t == BonsaiDtype::TQ2G128);
    CHECK(bonsai_block_bytes(*t) == 34);
    CHECK(bonsai_h1b_flag(*t) == H1B_FLAG_BONSAI_TQ2);

    CHECK_FALSE(bonsai_from_u32(35).has_value()); // canonical ggml TQ2_0
    CHECK_FALSE(bonsai_from_u32(0).has_value());
}

TEST_CASE("gguf-to-h1b: is_gguf_magic + read_gguf_version on synth GGUF")
{
    auto base = tmp_dir("magic");
    GgufBuilder b;
    b.kv_string("general.architecture", "tiny");
    auto bytes = b.finish_no_data();
    auto p = base / "tiny.gguf";
    write_file(p, bytes);

    auto m = is_gguf_magic(p);
    REQUIRE(m.has_value());
    CHECK(*m);

    auto v = read_gguf_version(p);
    REQUIRE(v.has_value());
    CHECK(*v == 3u);

    // A non-GGUF file.
    auto bad = base / "not.gguf";
    std::array<std::uint8_t, 8> ggml{'G', 'G', 'M', 'L', 0, 0, 0, 0};
    write_file(bad, std::span<const std::uint8_t>{ggml.data(), ggml.size()});
    auto m2 = is_gguf_magic(bad);
    REQUIRE(m2.has_value());
    CHECK_FALSE(*m2);
}

TEST_CASE("gguf-to-h1b: serialize_htok matches Rust HtokFile layout")
{
    // Round-trip: write 5 pieces + 1 merge, then re-parse the byte layout
    // by hand and check field offsets match the documented Rust format.
    std::vector<std::vector<std::uint8_t>> pieces = {
        {'<', 'b', 'o', 's', '>'},
        {'<', 'e', 'o', 's', '>'},
        {'a'},
        {'b'},
        {'a', 'b'},
    };
    std::vector<MergeTriple> merges = {{2, 3, 4}};
    auto blob = serialize_htok(0, 1, pieces, merges);
    REQUIRE(blob.size() >= 20u);

    // Header (Rust format): 'HTOK' + u32 vocab + u32 nmerges + u32 bos + u32 eos.
    CHECK(blob[0] == 'H');
    CHECK(blob[1] == 'T');
    CHECK(blob[2] == 'O');
    CHECK(blob[3] == 'K');
    std::uint32_t vocab = 0, nm = 0, bos = 0, eos = 0;
    std::memcpy(&vocab, blob.data() + 4,  4);
    std::memcpy(&nm,    blob.data() + 8,  4);
    std::memcpy(&bos,   blob.data() + 12, 4);
    std::memcpy(&eos,   blob.data() + 16, 4);
    CHECK(vocab == 5u);
    CHECK(nm    == 1u);
    CHECK(bos   == 0u);
    CHECK(eos   == 1u);

    // Then 5 pieces × (u16 len + bytes). First piece "<bos>" = 5 bytes.
    std::size_t off = 20;
    for (const auto& p : pieces) {
        std::uint16_t len = 0;
        std::memcpy(&len, blob.data() + off, 2);
        CHECK(len == p.size());
        off += 2;
        for (std::size_t i = 0; i < p.size(); ++i) {
            CHECK(blob[off + i] == p[i]);
        }
        off += p.size();
    }
    // Then 1 merge × (u32 a, u32 b, u32 merged).
    std::uint32_t a = 0, b = 0, merged = 0;
    std::memcpy(&a,      blob.data() + off,      4);
    std::memcpy(&b,      blob.data() + off + 4,  4);
    std::memcpy(&merged, blob.data() + off + 8,  4);
    CHECK(a      == 2u);
    CHECK(b      == 3u);
    CHECK(merged == 4u);
}

TEST_CASE("gguf-to-h1b: build_htok_blob skips merges with unresolvable sides")
{
    BonsaiHeader hdr;
    hdr.architecture = "qwen3";
    hdr.tokens       = {"<bos>", "<eos>", "a", "b", "ab"};
    hdr.merges       = {
        "a b",     // resolvable (a=2, b=3, merged=4)
        "a x",     // unresolvable (no `x` in vocab)
        "x y",     // unresolvable (neither side)
    };
    hdr.bos_token_id = 0;
    hdr.eos_token_id = 1;

    auto built = build_htok_blob(hdr);
    REQUIRE(built.has_value());
    const auto& [stats, blob] = *built;
    CHECK(stats.vocab_size     == 5u);
    CHECK(stats.num_merges     == 1u);
    CHECK(stats.dropped_merges == 2u);
    CHECK(stats.bos_id         == 0);
    CHECK(stats.eos_id         == 1);
    // Sanity: blob is well-formed.
    CHECK(blob.size() == stats.output_bytes);
}

TEST_CASE("gguf-to-h1b: convert_file rejects non-qwen3 architecture")
{
    auto base = tmp_dir("arch");
    GgufBuilder b;
    b.kv_string("general.architecture", "llama");
    auto bytes = b.finish_no_data();
    auto p = base / "wrong.gguf";
    write_file(p, bytes);

    auto r = convert_file(p, base / "out.h1b");
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().what.find("not a Bonsai") != std::string::npos);
}

TEST_CASE("gguf-to-h1b: convert_file rejects GGUF with no Bonsai tensors")
{
    auto base = tmp_dir("notensors");
    GgufBuilder b;
    b.kv_string("general.architecture", "qwen3");
    b.kv_u32("qwen3.block_count", 0); // 0 layers → loop body never enters
    b.kv_u32("qwen3.embedding_length", 128);
    b.kv_u32("qwen3.feed_forward_length", 256);
    b.kv_u32("qwen3.attention.head_count", 4);
    b.kv_u32("qwen3.attention.head_count_kv", 1);
    b.kv_u32("qwen3.context_length", 64);
    auto bytes = b.finish_no_data();
    auto p = base / "empty.gguf";
    write_file(p, bytes);

    auto r = convert_file(p, base / "out.h1b");
    REQUIRE_FALSE(r.has_value());
    // 0-layer fixture errors at "no ternary tensors found".
    CHECK(r.error().what.find("no ternary tensors") != std::string::npos);
}
