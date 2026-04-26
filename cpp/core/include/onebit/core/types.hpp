#pragma once

#include <cstdint>

namespace onebit::core {

// Token id. Matches the C++ side's int32_t; stable across the FFI boundary
// to rocm-cpp.
using TokenId = std::int32_t;

// Minimum .h1b format version this lib parses.
inline constexpr std::int32_t MIN_SUPPORTED_VERSION = 1;

// Maximum .h1b format version this lib parses.
//
//   v1 — original halo-1bit export (2 bpw, no rope/eps in header).
//   v2 — same payload as v1 but with rope_theta + rms_norm_eps in header.
//   v3 — Sherry 1.25 bpw ternary packing (cols % 32 == 0).
//   v4 — TQ1 base-3 packing, 1.6 bpw, lossless (cols padded to mult. of 20).
inline constexpr std::int32_t MAX_SUPPORTED_VERSION = 4;

// Defaults for v1 files that don't carry explicit values. Match the
// fallbacks in rocm-cpp/src/h1b_loader.cpp.
inline constexpr float DEFAULT_ROPE_THETA   = 500'000.0f;
inline constexpr float DEFAULT_RMS_NORM_EPS = 1e-5f;

} // namespace onebit::core
