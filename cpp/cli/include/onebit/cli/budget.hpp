#pragma once

// `1bit budget` — GTT + RAM budget audit on the strixhalo iGPU.

#include "onebit/cli/error.hpp"

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::cli {

struct ServiceRss {
    std::string   name;
    std::uint64_t rss_kib;
};

struct BudgetSnapshot {
    std::uint64_t gtt_total      = 0;
    std::uint64_t gtt_used       = 0;
    std::uint64_t vram_total     = 0;
    std::uint64_t vram_used      = 0;
    std::uint64_t mem_total      = 0;   ///< bytes
    std::uint64_t mem_available  = 0;   ///< bytes
    std::vector<ServiceRss> services;

    [[nodiscard]] std::uint64_t gtt_free() const noexcept
    {
        return gtt_total > gtt_used ? gtt_total - gtt_used : 0;
    }

    [[nodiscard]] std::uint64_t budget_for_next_model() const noexcept;

    [[nodiscard]] std::string render() const;
};

inline constexpr std::uint64_t COMPOSITOR_RESERVE_BYTES = 4ULL * 1024 * 1024 * 1024;

// Pure parser used by the budget renderer + tested directly.
[[nodiscard]] std::optional<std::uint64_t>
parse_meminfo_kib(std::string_view meminfo, std::string_view key) noexcept;

[[nodiscard]] std::string fmt_bytes(std::uint64_t n);

[[nodiscard]] bool looks_like_halo_service(std::string_view comm) noexcept;

}  // namespace onebit::cli
