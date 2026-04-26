#pragma once

#include <string_view>

namespace onebit::cli {

// Compile-time version pin. Matches the Rust crate's `Cargo.toml`
// version field and the `releases.json` feed pin. Bump in lockstep.
inline constexpr std::string_view ONEBIT_CLI_VERSION = "0.1.0";

}  // namespace onebit::cli
