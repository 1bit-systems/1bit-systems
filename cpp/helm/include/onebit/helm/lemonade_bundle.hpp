#pragma once

// Embeddable Lemonade integration. When 1bit-helm is built with
// `-DHELM_BUNDLE_LEMONADE=ON`, the install drops the upstream
// lemonade-embeddable tarball at `share/1bit-helm/lemonade/`. At runtime
// helm probes for a live OpenAI-compat server on :8180; if absent and the
// bundle exists, it spawns the bundled `lemond` as a child process.

#include <filesystem>
#include <optional>
#include <string>

namespace onebit::helm::lemonade_bundle {

// Default port (matches our halo deployment; upstream default is 13305).
inline constexpr int DEFAULT_PORT = 8180;

// Resolve the bundled `lemond` binary path relative to the helm executable.
// Returns std::nullopt when:
//   * the build was not configured with HELM_BUNDLE_LEMONADE=ON, OR
//   * the bundle directory exists but `bin/lemond` is missing.
[[nodiscard]] std::optional<std::filesystem::path>
bundled_lemond_path();

// True when a server on `host:port` answers GET /v1/models with HTTP 2xx
// within `timeout_ms`. Used by helm before deciding to spawn the bundle.
[[nodiscard]] bool
probe_openai_endpoint(const std::string& host,
                      int                port,
                      int                timeout_ms = 250);

} // namespace onebit::helm::lemonade_bundle
