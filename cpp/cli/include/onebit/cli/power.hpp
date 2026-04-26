#pragma once

// `1bit power [profile]` — Ryzen APU power-profile CLI.

#include "onebit/cli/error.hpp"

#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::cli {

enum class PowerProfile : std::uint8_t { Inference, Chat, Idle };

struct PowerEnvelope {
    std::uint32_t stapm_mw;
    std::uint32_t fast_mw;
    std::uint32_t slow_mw;
    std::uint32_t tctl_c;
};

[[nodiscard]] PowerEnvelope envelope_of(PowerProfile p) noexcept;
[[nodiscard]] std::string_view name_of(PowerProfile p) noexcept;
[[nodiscard]] std::string_view description_of(PowerProfile p) noexcept;
[[nodiscard]] std::vector<std::string> ryzenadj_argv(PowerProfile p);

[[nodiscard]] std::expected<PowerProfile, Error>
parse_profile(std::string_view raw);

[[nodiscard]] std::vector<PowerProfile> list_profiles();

// Pure helper so tests can drive `ryzenadj --info`-shaped input through.
[[nodiscard]] std::string summarize_info(std::string_view info);

}  // namespace onebit::cli
