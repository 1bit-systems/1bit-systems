#include "onebit/ingest/sha256.hpp"

#include <bit>
#include <cstring>

namespace onebit::ingest::detail {

namespace {

constexpr std::array<std::uint32_t, 64> K{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

[[nodiscard]] constexpr std::uint32_t big_sigma0(std::uint32_t x) noexcept
{
    return std::rotr(x, 2) ^ std::rotr(x, 13) ^ std::rotr(x, 22);
}
[[nodiscard]] constexpr std::uint32_t big_sigma1(std::uint32_t x) noexcept
{
    return std::rotr(x, 6) ^ std::rotr(x, 11) ^ std::rotr(x, 25);
}
[[nodiscard]] constexpr std::uint32_t small_sigma0(std::uint32_t x) noexcept
{
    return std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3);
}
[[nodiscard]] constexpr std::uint32_t small_sigma1(std::uint32_t x) noexcept
{
    return std::rotr(x, 17) ^ std::rotr(x, 19) ^ (x >> 10);
}
[[nodiscard]] constexpr std::uint32_t ch(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept
{
    return (x & y) ^ (~x & z);
}
[[nodiscard]] constexpr std::uint32_t maj(std::uint32_t x, std::uint32_t y, std::uint32_t z) noexcept
{
    return (x & y) ^ (x & z) ^ (y & z);
}

[[nodiscard]] std::uint32_t load_be32(const std::uint8_t* p) noexcept
{
    return (static_cast<std::uint32_t>(p[0]) << 24) |
           (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) |
           static_cast<std::uint32_t>(p[3]);
}

void store_be32(std::uint8_t* p, std::uint32_t v) noexcept
{
    p[0] = static_cast<std::uint8_t>(v >> 24);
    p[1] = static_cast<std::uint8_t>(v >> 16);
    p[2] = static_cast<std::uint8_t>(v >> 8);
    p[3] = static_cast<std::uint8_t>(v);
}

} // namespace

Sha256::Sha256()
    : h_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}
{
}

void Sha256::compress(const std::uint8_t* block) noexcept
{
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16; ++i) {
        w[i] = load_be32(block + i * 4);
    }
    for (std::size_t i = 16; i < 64; ++i) {
        w[i] = small_sigma1(w[i - 2]) + w[i - 7] +
               small_sigma0(w[i - 15]) + w[i - 16];
    }

    std::uint32_t a = h_[0];
    std::uint32_t b = h_[1];
    std::uint32_t c = h_[2];
    std::uint32_t d = h_[3];
    std::uint32_t e = h_[4];
    std::uint32_t f = h_[5];
    std::uint32_t g = h_[6];
    std::uint32_t hh = h_[7];

    for (std::size_t i = 0; i < 64; ++i) {
        const std::uint32_t t1 = hh + big_sigma1(e) + ch(e, f, g) + K[i] + w[i];
        const std::uint32_t t2 = big_sigma0(a) + maj(a, b, c);
        hh = g;
        g  = f;
        f  = e;
        e  = d + t1;
        d  = c;
        c  = b;
        b  = a;
        a  = t1 + t2;
    }

    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += hh;
}

void Sha256::update(std::span<const std::uint8_t> data) noexcept
{
    bits_ += static_cast<std::uint64_t>(data.size()) * 8U;
    std::size_t pos = 0;
    if (buf_len_ > 0) {
        const std::size_t want = 64 - buf_len_;
        const std::size_t take = (data.size() < want) ? data.size() : want;
        std::memcpy(buf_.data() + buf_len_, data.data(), take);
        buf_len_ += take;
        pos       = take;
        if (buf_len_ == 64) {
            compress(buf_.data());
            buf_len_ = 0;
        }
    }
    while (pos + 64 <= data.size()) {
        compress(data.data() + pos);
        pos += 64;
    }
    if (pos < data.size()) {
        std::memcpy(buf_.data(), data.data() + pos, data.size() - pos);
        buf_len_ = data.size() - pos;
    }
}

void Sha256::update(std::string_view sv) noexcept
{
    update(std::span<const std::uint8_t>{
        reinterpret_cast<const std::uint8_t*>(sv.data()), sv.size()});
}

std::array<std::uint8_t, 32> Sha256::finalize() noexcept
{
    const std::uint64_t total_bits = bits_;
    buf_[buf_len_++]               = 0x80;
    if (buf_len_ > 56) {
        while (buf_len_ < 64) {
            buf_[buf_len_++] = 0;
        }
        compress(buf_.data());
        buf_len_ = 0;
    }
    while (buf_len_ < 56) {
        buf_[buf_len_++] = 0;
    }
    for (int i = 7; i >= 0; --i) {
        buf_[buf_len_++] = static_cast<std::uint8_t>(total_bits >> (i * 8));
    }
    compress(buf_.data());

    std::array<std::uint8_t, 32> out{};
    for (std::size_t i = 0; i < 8; ++i) {
        store_be32(out.data() + i * 4, h_[i]);
    }
    return out;
}

std::array<std::uint8_t, 32> sha256(std::span<const std::uint8_t> data) noexcept
{
    Sha256 h;
    h.update(data);
    return h.finalize();
}

std::string to_hex(std::span<const std::uint8_t> data)
{
    static constexpr char hex[] = "0123456789abcdef";
    std::string           out;
    out.reserve(data.size() * 2);
    for (auto b : data) {
        out.push_back(hex[(b >> 4) & 0xF]);
        out.push_back(hex[b & 0xF]);
    }
    return out;
}

std::string sha256_hex(std::span<const std::uint8_t> data)
{
    return to_hex(sha256(data));
}

} // namespace onebit::ingest::detail
