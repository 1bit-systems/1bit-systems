#include <doctest/doctest.h>

#include "onebit/ingest/sha256.hpp"

#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using onebit::ingest::detail::sha256;
using onebit::ingest::detail::sha256_hex;
using onebit::ingest::detail::Sha256;

TEST_CASE("known vector — empty string")
{
    auto h = sha256_hex(std::span<const std::uint8_t>{});
    CHECK(h ==
          "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

TEST_CASE("known vector — 'abc'")
{
    const char*               s = "abc";
    std::span<const std::uint8_t> sp{reinterpret_cast<const std::uint8_t*>(s), 3};
    auto                      h = sha256_hex(sp);
    CHECK(h ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST_CASE("incremental update equals one-shot")
{
    const std::string data = "the quick brown fox jumps over the lazy dog";
    Sha256            a;
    a.update(data);
    auto out_a = a.finalize();

    Sha256 b;
    b.update(std::string_view{data}.substr(0, 4));
    b.update(std::string_view{data}.substr(4, 11));
    b.update(std::string_view{data}.substr(15));
    auto out_b = b.finalize();

    CHECK(std::memcmp(out_a.data(), out_b.data(), 32) == 0);
}

TEST_CASE("multi-block input (>64 bytes)")
{
    std::vector<std::uint8_t> v(200, 0xA5);
    auto                      a = sha256(v);
    Sha256                    h;
    h.update(std::span<const std::uint8_t>{v.data(), 100});
    h.update(std::span<const std::uint8_t>{v.data() + 100, 100});
    auto b = h.finalize();
    CHECK(std::memcmp(a.data(), b.data(), 32) == 0);
}

TEST_CASE("hex output is lowercase 64 chars")
{
    auto h = sha256_hex(std::span<const std::uint8_t>{});
    CHECK(h.size() == 64);
    for (char c : h) {
        CHECK(((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')));
    }
}
