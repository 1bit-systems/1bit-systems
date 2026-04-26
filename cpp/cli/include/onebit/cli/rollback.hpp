#pragma once

// `1bit rollback` — snapper-backed snapshot rollback.

#include "onebit/cli/error.hpp"

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::cli {

inline constexpr std::string_view HALO_PREINSTALL_LABEL = ".halo-preinstall";

struct SnapperEntry {
    std::uint32_t number;
    std::string   description;
};

class Snapper {
public:
    Snapper() = default;
    Snapper(const Snapper&) = delete;
    Snapper& operator=(const Snapper&) = delete;
    Snapper(Snapper&&) noexcept = default;
    Snapper& operator=(Snapper&&) noexcept = default;
    virtual ~Snapper() = default;

    [[nodiscard]] virtual bool                                          available() = 0;
    [[nodiscard]] virtual std::expected<std::vector<SnapperEntry>, Error> list() = 0;
    [[nodiscard]] virtual std::expected<void, Error>                     rollback(std::uint32_t n) = 0;
};

[[nodiscard]] std::vector<SnapperEntry> parse_snapper_list(std::string_view stdout_text);

[[nodiscard]] std::optional<std::uint32_t>
pick_latest_preinstall(const std::vector<SnapperEntry>& entries);

[[nodiscard]] std::expected<void, Error>
run_with_snapper(Snapper& snapper,
                 std::optional<std::uint32_t> snapshot,
                 bool yes);

}  // namespace onebit::cli
