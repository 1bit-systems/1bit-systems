#include <doctest/doctest.h>

#include "onebit/core/htok.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

using onebit::core::htok::File;
using onebit::core::htok::MAGIC;

namespace {

template <typename T>
void push_le(std::vector<std::uint8_t>& v, T x)
{
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        v.push_back(static_cast<std::uint8_t>((static_cast<std::uint64_t>(x) >> (i * 8)) & 0xff));
    }
}

// Build a minimal valid .htok with 4 byte-pieces and 1 merge.
std::vector<std::uint8_t> build_min_htok()
{
    std::vector<std::uint8_t> v;
    v.insert(v.end(), MAGIC.begin(), MAGIC.end());
    push_le<std::int32_t>(v, 1);   // bos
    push_le<std::int32_t>(v, 2);   // eos
    push_le<std::int32_t>(v, 0);   // pad
    push_le<std::int32_t>(v, 4);   // n_pieces
    push_le<std::int32_t>(v, 1);   // n_merges
    // 8 bytes of reserved padding to reach offset 32 (24 bytes of header
    // fields above + 8 = 32).
    for (int i = 0; i < 8; ++i) v.push_back(0);
    // Pieces: 'a', 'b', 'c', "ab"
    auto add_piece = [&](const char* s) {
        const auto len = static_cast<std::uint32_t>(std::strlen(s));
        push_le<std::uint32_t>(v, len);
        v.insert(v.end(), s, s + len);
    };
    add_piece("a"); add_piece("b"); add_piece("c"); add_piece("ab");
    // Merge: a + b → ab (rank 0)
    push_le<std::uint32_t>(v, 1);
    push_le<std::uint32_t>(v, 1);
    v.push_back('a'); v.push_back('b');
    return v;
}

} // namespace

TEST_CASE("htok: parse minimal file")
{
    auto bytes = build_min_htok();
    auto r = File::parse(bytes);
    REQUIRE(r.has_value());
    CHECK(r->bos_id() == 1);
    CHECK(r->eos_id() == 2);
    CHECK(r->pad_id() == 0);
    CHECK(r->vocab()  == 4u);
    CHECK(r->merges().size() == 1u);
}

TEST_CASE("htok: encode greedy merges 'a' + 'b' → 'ab'")
{
    auto bytes = build_min_htok();
    auto f = File::parse(bytes);
    REQUIRE(f.has_value());
    std::vector<std::uint8_t> input = {'a', 'b', 'c'};
    auto ids = f->encode(input);
    REQUIRE(ids.has_value());
    REQUIRE(ids->size() == 2u);
    CHECK(ids->at(0) == 3);  // "ab" piece id
    CHECK(ids->at(1) == 2);  // "c"
}

TEST_CASE("htok: decode round-trips encode")
{
    auto bytes = build_min_htok();
    auto f = File::parse(bytes);
    REQUIRE(f.has_value());
    std::vector<std::uint8_t> input = {'a', 'b', 'c'};
    auto ids = f->encode(input);
    REQUIRE(ids.has_value());
    auto out = f->decode(*ids);
    CHECK(out == input);
}

TEST_CASE("htok: bad magic rejected")
{
    std::vector<std::uint8_t> v(64, 0);
    auto r = File::parse(v);
    CHECK_FALSE(r.has_value());
}

TEST_CASE("htok: truncated rejected")
{
    std::vector<std::uint8_t> v;
    v.insert(v.end(), MAGIC.begin(), MAGIC.end());
    auto r = File::parse(v);
    CHECK_FALSE(r.has_value());
}
