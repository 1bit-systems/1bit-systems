#pragma once

// XDG / HOME helpers. Centralized so the rest of the CLI never reads
// $HOME / $XDG_* directly.

#include <filesystem>
#include <string>

namespace onebit::cli {

[[nodiscard]] std::filesystem::path home_dir();
[[nodiscard]] std::filesystem::path xdg_config_home();
[[nodiscard]] std::filesystem::path xdg_data_home();

[[nodiscard]] std::string env_or(std::string_view key, std::string_view fallback);

}  // namespace onebit::cli
