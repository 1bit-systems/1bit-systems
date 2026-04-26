#pragma once

// One-line JSON metrics sampler for `1bit-power log`.
//
// Reads `/sys/class/hwmon` entries (temp_input, power1_average) and the
// EC sysfs surface, emits a single nlohmann::json line. Anything missing
// just becomes `null` so the systemd timer keeps logging on every box.

#include <cstdint>
#include <optional>
#include <string>

#include "onebit/power/ec.hpp"
#include "onebit/power/result.hpp"

namespace onebit::power {

struct Sample {
    std::uint64_t                   ts_unix{0};
    std::string                     host;
    std::optional<float>            tctl_c;
    std::optional<float>            edge_c;
    std::optional<float>            pkg_power_w;
    std::optional<std::int32_t>     ec_temp_c;
    std::optional<std::string>      ec_power_mode;
    std::optional<std::uint32_t>    ec_fan1_rpm;
    std::optional<std::uint32_t>    ec_fan2_rpm;
    std::optional<std::uint32_t>    ec_fan3_rpm;
};

// Materialise the current sample. Never fails — missing sources become
// std::nullopt. The Result wrapper is kept for symmetry but always ok().
[[nodiscard]] Result<Sample> collect_sample(const EcBackend& ec);

// Render Sample as compact one-line JSON (matches Rust `serde_json::to_string`).
[[nodiscard]] std::string sample_to_json(const Sample& s);

// Helpers — exposed for tests.
[[nodiscard]] std::string read_hostname();
[[nodiscard]] std::optional<float> read_hwmon(std::string_view want, std::string_view file);

} // namespace onebit::power
