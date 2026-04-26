#include "onebit/cli/version.hpp"

// Header-only constant. .cpp present so the lib has a consistent
// per-header TU even when nothing else uses the symbol.

namespace onebit::cli {

[[maybe_unused]] static constexpr auto _kVersion = ONEBIT_CLI_VERSION;

}  // namespace onebit::cli
