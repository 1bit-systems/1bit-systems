// 1bit-helm — /_live/stats SSE shape + parser.
//
// Mirrors crates/1bit-helm/src/telemetry.rs. The on-the-wire SSE
// reader runs in a background QThread (see telemetry_worker.hpp); the
// pure JSON-to-LiveStats parser lives here for headless tests.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::helm {

struct ServiceDot {
    std::string name;
    bool        active{false};
};

struct LiveStats {
    std::string             loaded_model;
    float                   tok_s_decode{0.0F};
    float                   gpu_temp_c{0.0F};
    std::uint8_t            gpu_util_pct{0};
    bool                    npu_up{false};
    float                   shadow_burn_exact_pct{0.0F};
    std::vector<ServiceDot> services;
    bool                    stale{false};
};

// Parse the JSON payload of one SSE `data:` line. nullopt on
// malformed input.
[[nodiscard]] std::optional<LiveStats> parse_stats(std::string_view data);

// Pull the JSON payload out of a raw SSE line. Returns nullopt for
// non-`data:` lines, blanks, comments, and the `[DONE]` sentinel.
[[nodiscard]] std::optional<std::string>
extract_landing_payload(std::string_view line);

} // namespace onebit::helm
