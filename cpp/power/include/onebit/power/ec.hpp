#pragma once

// Embedded-controller backend for the Sixunited AXB35-02 board.
//
// Drives fans + APU power-mode + CPU temp via the `ec_su_axb35` kernel
// module's sysfs surface at /sys/class/ec_su_axb35/. The sysfs root is
// injectable so tests can point it at a tempdir mimicking the layout.
//
// Writes need root (sysfs files are 0644, root-owned); the CLI documents
// that `fan` and `board` subcommands need sudo.

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "onebit/power/result.hpp"

namespace onebit::power {

inline constexpr std::string_view EC_SYSFS_ROOT = "/sys/class/ec_su_axb35";

enum class CurveDir { Rampup, Rampdown };

struct FanSnapshot {
    std::uint8_t            id{0};
    std::uint32_t           rpm{0};
    std::string             mode;
    std::uint8_t            level{0};
    std::vector<std::uint8_t> rampup;
    std::vector<std::uint8_t> rampdown;
};

struct EcSnapshot {
    std::optional<std::int32_t> temp_c;
    std::optional<std::string>  power_mode;
    std::vector<FanSnapshot>    fans;
};

class EcBackend {
public:
    EcBackend() : root_(EC_SYSFS_ROOT) {}
    explicit EcBackend(std::filesystem::path root) : root_(std::move(root)) {}

    [[nodiscard]] bool available() const noexcept;

    [[nodiscard]] Result<std::int32_t> temp_c() const;
    [[nodiscard]] Result<std::string>  power_mode() const;

    [[nodiscard]] Status set_power_mode(std::string_view mode) const;

    [[nodiscard]] Result<FanSnapshot> fan(std::uint8_t id) const;
    [[nodiscard]] Status set_fan_mode(std::uint8_t id, std::string_view mode) const;
    [[nodiscard]] Status set_fan_level(std::uint8_t id, std::uint8_t level) const;
    [[nodiscard]] Status set_fan_curve(std::uint8_t id,
                                       CurveDir direction,
                                       const std::array<std::uint8_t, 5>& curve) const;

    [[nodiscard]] EcSnapshot snapshot() const;

    [[nodiscard]] const std::filesystem::path& root() const noexcept { return root_; }

private:
    std::filesystem::path root_;
};

// Helpers exposed for unit tests.
[[nodiscard]] Status validate_fan_id(std::uint8_t id);
[[nodiscard]] std::vector<std::uint8_t> parse_curve_csv(std::string_view s);

} // namespace onebit::power
