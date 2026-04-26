#include "onebit/cli/update.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <string>

// Minimal pure-C++ SHA-256. The CLI hashes update artifacts (~50-500 MB)
// once on download — performance is not critical, simplicity is. We
// deliberately avoid pulling in OpenSSL just for one hash.

namespace onebit::cli {

namespace {

class Sha256 {
public:
    Sha256() noexcept { reset(); }

    void reset() noexcept
    {
        state_ = INIT_STATE;
        bit_len_ = 0;
        buf_len_ = 0;
    }

    void update(const std::uint8_t* data, std::size_t n) noexcept
    {
        bit_len_ += static_cast<std::uint64_t>(n) * 8U;
        while (n > 0) {
            const std::size_t take =
                std::min<std::size_t>(BLOCK - buf_len_, n);
            std::memcpy(buf_.data() + buf_len_, data, take);
            buf_len_ += take;
            data    += take;
            n       -= take;
            if (buf_len_ == BLOCK) {
                process_block(buf_.data());
                buf_len_ = 0;
            }
        }
    }

    [[nodiscard]] std::array<std::uint8_t, 32> finalize() noexcept
    {
        const std::uint64_t bit_len_be = bit_len_;

        // Pad: 0x80 then zeros to 56 (mod 64), then 8-byte big-endian length.
        buf_[buf_len_++] = 0x80U;
        if (buf_len_ > 56) {
            while (buf_len_ < BLOCK) buf_[buf_len_++] = 0;
            process_block(buf_.data());
            buf_len_ = 0;
        }
        while (buf_len_ < 56) buf_[buf_len_++] = 0;

        for (int i = 7; i >= 0; --i) {
            buf_[buf_len_++] = static_cast<std::uint8_t>((bit_len_be >> (i * 8)) & 0xFFU);
        }
        process_block(buf_.data());

        std::array<std::uint8_t, 32> digest{};
        for (std::size_t i = 0; i < 8; ++i) {
            digest[i * 4 + 0] = static_cast<std::uint8_t>((state_[i] >> 24) & 0xFFU);
            digest[i * 4 + 1] = static_cast<std::uint8_t>((state_[i] >> 16) & 0xFFU);
            digest[i * 4 + 2] = static_cast<std::uint8_t>((state_[i] >> 8) & 0xFFU);
            digest[i * 4 + 3] = static_cast<std::uint8_t>(state_[i] & 0xFFU);
        }
        reset();
        return digest;
    }

private:
    static constexpr std::size_t BLOCK = 64;

    static constexpr std::array<std::uint32_t, 64> K = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U,
    };

    static constexpr std::array<std::uint32_t, 8> INIT_STATE = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };

    void process_block(const std::uint8_t* p) noexcept
    {
        std::array<std::uint32_t, 64> w{};
        for (std::size_t i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(p[i * 4 + 0]) << 24) |
                   (static_cast<std::uint32_t>(p[i * 4 + 1]) << 16) |
                   (static_cast<std::uint32_t>(p[i * 4 + 2]) << 8)  |
                   (static_cast<std::uint32_t>(p[i * 4 + 3]));
        }
        for (std::size_t i = 16; i < 64; ++i) {
            const std::uint32_t s0 = std::rotr(w[i - 15], 7) ^ std::rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = std::rotr(w[i - 2], 17) ^ std::rotr(w[i - 2],  19) ^ (w[i - 2]  >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (std::size_t i = 0; i < 64; ++i) {
            const std::uint32_t S1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = h + S1 + ch + K[i] + w[i];
            const std::uint32_t S0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const std::uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = S0 + mj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::array<std::uint8_t, BLOCK> buf_{};
    std::size_t                  buf_len_ = 0;
    std::uint64_t                bit_len_ = 0;
};

[[nodiscard]] std::string to_hex(const std::array<std::uint8_t, 32>& digest)
{
    static constexpr char H[] = "0123456789abcdef";
    std::string out(64, '\0');
    for (std::size_t i = 0; i < 32; ++i) {
        out[i * 2 + 0] = H[(digest[i] >> 4) & 0x0F];
        out[i * 2 + 1] = H[digest[i] & 0x0F];
    }
    return out;
}

[[nodiscard]] bool eq_ignore_case(std::string_view a, std::string_view b) noexcept
{
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        char ca = a[i], cb = b[i];
        if (ca >= 'A' && ca <= 'Z') ca = static_cast<char>(ca + 32);
        if (cb >= 'A' && cb <= 'Z') cb = static_cast<char>(cb + 32);
        if (ca != cb) return false;
    }
    return true;
}

}  // namespace

std::expected<std::string, Error> sha256_file(const std::filesystem::path& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return std::unexpected(Error::io("cannot open " + path.string()));
    }
    Sha256 h;
    std::array<std::uint8_t, 64 * 1024> buf{};
    while (f) {
        f.read(reinterpret_cast<char*>(buf.data()),
               static_cast<std::streamsize>(buf.size()));
        const auto n = f.gcount();
        if (n > 0) {
            h.update(buf.data(), static_cast<std::size_t>(n));
        }
    }
    if (f.bad()) {
        return std::unexpected(Error::io("read error on " + path.string()));
    }
    return to_hex(h.finalize());
}

std::expected<void, Error>
verify_sha256(const std::filesystem::path& path, std::string_view expect_hex)
{
    auto got = sha256_file(path);
    if (!got) return std::unexpected(got.error());
    // Trim whitespace on the expected pin to match Rust behavior.
    std::string_view expected = expect_hex;
    while (!expected.empty() && (expected.front() == ' ' || expected.front() == '\t' ||
                                  expected.front() == '\n' || expected.front() == '\r')) {
        expected.remove_prefix(1);
    }
    while (!expected.empty() && (expected.back() == ' ' || expected.back() == '\t' ||
                                  expected.back() == '\n' || expected.back() == '\r')) {
        expected.remove_suffix(1);
    }
    if (!eq_ignore_case(*got, expected)) {
        return std::unexpected(Error::hash(
            "sha256 mismatch for " + path.string() +
            ": got " + *got + ", expected " + std::string(expected)));
    }
    return {};
}

}  // namespace onebit::cli
