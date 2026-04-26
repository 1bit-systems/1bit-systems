#pragma once

// Minimal SHA-256 implementation used by .1bl pack/validate. Self-contained
// — no OpenSSL link, no third-party. Mirrors what the Rust crate gets from
// the `sha2` crate.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace onebit::ingest::detail {

class Sha256 {
public:
    Sha256();
    void                              update(std::span<const std::uint8_t> data) noexcept;
    void                              update(std::string_view sv) noexcept;
    [[nodiscard]] std::array<std::uint8_t, 32> finalize() noexcept;

private:
    std::array<std::uint32_t, 8> h_{};
    std::array<std::uint8_t, 64> buf_{};
    std::size_t                  buf_len_{0};
    std::uint64_t                bits_{0};

    void compress(const std::uint8_t* block) noexcept;
};

[[nodiscard]] std::array<std::uint8_t, 32>
                          sha256(std::span<const std::uint8_t> data) noexcept;
[[nodiscard]] std::string sha256_hex(std::span<const std::uint8_t> data);
[[nodiscard]] std::string to_hex(std::span<const std::uint8_t> data);

} // namespace onebit::ingest::detail
