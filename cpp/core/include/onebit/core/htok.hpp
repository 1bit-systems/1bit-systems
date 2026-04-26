#pragma once

#include "onebit/core/error.hpp"
#include "onebit/core/types.hpp"

#include <array>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace onebit::core::htok {

// Magic = "HTOK"
inline constexpr std::array<std::uint8_t, 4> MAGIC = {'H', 'T', 'O', 'K'};

struct Merge {
    std::vector<std::uint8_t> a;
    std::vector<std::uint8_t> b;
    std::int32_t              rank = 0;
};

class File {
public:
    [[nodiscard]] static std::expected<File, HaloError>
    open(const std::filesystem::path& path);

    [[nodiscard]] static std::expected<File, HaloError>
    parse(std::span<const std::uint8_t> bytes);

    [[nodiscard]] std::int32_t bos_id()   const noexcept { return bos_id_; }
    [[nodiscard]] std::int32_t eos_id()   const noexcept { return eos_id_; }
    [[nodiscard]] std::int32_t pad_id()   const noexcept { return pad_id_; }
    [[nodiscard]] std::size_t  vocab()    const noexcept { return pieces_.size(); }

    [[nodiscard]] const std::vector<std::vector<std::uint8_t>>& pieces() const noexcept { return pieces_; }
    [[nodiscard]] const std::vector<Merge>&                     merges() const noexcept { return merges_; }

    // BPE encode raw UTF-8 bytes → token ids. Greedy left-to-right by
    // ranked merges; matches the Rust crate.
    [[nodiscard]] std::expected<std::vector<TokenId>, HaloError>
    encode(std::span<const std::uint8_t> bytes) const;

    // Token id → raw UTF-8 bytes (joined from pieces).
    [[nodiscard]] std::vector<std::uint8_t>
    decode(std::span<const TokenId> ids) const;

private:
    std::int32_t bos_id_ = -1;
    std::int32_t eos_id_ = -1;
    std::int32_t pad_id_ = -1;
    std::vector<std::vector<std::uint8_t>>                 pieces_{};
    std::vector<Merge>                                     merges_{};
    std::unordered_map<std::string, std::int32_t>          piece_to_id_{};
};

} // namespace onebit::core::htok
