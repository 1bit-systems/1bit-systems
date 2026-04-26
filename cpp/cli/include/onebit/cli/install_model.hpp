#pragma once

// `1bit install <model>` — fetch GGUF via `hf` CLI, sha256-verify,
// atomic symlink to ~/.local/share/1bit/models/<id>/, restart owning units.

#include "onebit/cli/error.hpp"
#include "onebit/cli/package.hpp"

#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::cli {

[[nodiscard]] std::filesystem::path models_root();
[[nodiscard]] std::filesystem::path model_dir(std::string_view id);

// Driver. `engine_units` is the systemd units to restart after the GGUF
// lands; resolved by the caller from `model.requires` -> component.units.
[[nodiscard]] std::expected<void, Error>
install_model(const Model& spec, const std::vector<std::string>& engine_units);

}  // namespace onebit::cli
