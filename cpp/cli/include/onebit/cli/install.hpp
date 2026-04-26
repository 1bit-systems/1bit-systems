#pragma once

// `1bit install <component>` — dependency-resolved component installer.
//
// The pure layer is `resolve_install_order`: pure dependency-graph walk,
// fully testable without touching the host. The host-touching side
// (subprocess spawn / file copy / systemctl) is wrapped behind a
// `HostExecutor` interface so unit tests can use a `FakeExecutor` that
// records calls.

#include "onebit/cli/error.hpp"
#include "onebit/cli/install_tracker.hpp"
#include "onebit/cli/package.hpp"

#include <expected>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace onebit::cli {

// Topologically-sorted install order. Returns names (not Component
// pointers) so callers re-look-up via `manifest.components.at(name)` on
// each step.
[[nodiscard]] std::expected<std::vector<std::string>, Error>
resolve_install_order(const Manifest& m, std::string_view target);

// Abstract host executor — every side-effecting step funnels through here.
class HostExecutor {
public:
    HostExecutor() = default;
    HostExecutor(const HostExecutor&) = delete;
    HostExecutor& operator=(const HostExecutor&) = delete;
    HostExecutor(HostExecutor&&) noexcept = default;
    HostExecutor& operator=(HostExecutor&&) noexcept = default;
    virtual ~HostExecutor() = default;

    // argv, run with `cwd` = workspace root. Return Ok if exit==0.
    virtual std::expected<void, Error>
    run_argv(const std::filesystem::path& cwd,
             const std::vector<std::string>& argv) = 0;

    // `systemctl --user enable --now <unit>`
    virtual std::expected<void, Error>
    systemctl_enable_now(std::string_view unit) = 0;

    // `systemctl --user restart <unit>`
    virtual std::expected<void, Error>
    systemctl_restart(std::string_view unit) = 0;

    // Render `src` → `dest`, applying placeholder substitutions. If the
    // destination path is absolute the implementation is allowed to
    // shell out to `sudo tee`; relative dests use ordinary fs::write.
    // Returns true if a new file was written; false on "skip (exists)".
    virtual std::expected<bool, Error>
    copy_tracked_file(const std::filesystem::path& src,
                      const std::filesystem::path& dest,
                      const std::map<std::string, std::string>& subs) = 0;

    // 2xx healthcheck against url. Returns true on 2xx, false otherwise.
    // Empty url is a no-op (returns true).
    [[nodiscard]] virtual bool healthcheck(std::string_view url) = 0;
};

// Real implementation that shells out via popen / fork + execvp / fs.
[[nodiscard]] std::unique_ptr<HostExecutor> make_real_executor();

// `1bit install --list` rendered as text rows.
[[nodiscard]] std::vector<std::string>
render_install_list(const Manifest& m);

// Drive a non-OOBE `1bit install <component>`.
struct InstallContext {
    std::filesystem::path workspace_root;   ///< `cwd` for build steps
    std::filesystem::path config_root;      ///< $XDG_CONFIG_HOME equivalent
};

[[nodiscard]] std::expected<void, Error>
run_install(HostExecutor& host,
            const Manifest& m,
            std::string_view component,
            InstallTracker& tracker,
            const InstallContext& ctx);

// OOBE flow surface — see preflight.hpp + oobe_error.hpp for the gate /
// error types. `run_oobe_full` is the testable core.
struct OobeDefaults {
    std::string component   = "core";
    bool        skip_build  = false;
    bool        yes         = false;
    bool        doctor_skip = false;
};

class DoctorProbe {
public:
    DoctorProbe() = default;
    DoctorProbe(const DoctorProbe&) = delete;
    DoctorProbe& operator=(const DoctorProbe&) = delete;
    DoctorProbe(DoctorProbe&&) noexcept = default;
    DoctorProbe& operator=(DoctorProbe&&) noexcept = default;
    virtual ~DoctorProbe() = default;
    /// Returns (warn_count, fail_count).
    [[nodiscard]] virtual std::pair<std::uint32_t, std::uint32_t> run() = 0;
};

class SystemProbe;  // forward — defined in preflight.hpp

// Run the full OOBE pipeline (preflight → install → doctor) with
// injectable probes. Returns Ok only when every gate passes.
[[nodiscard]] std::expected<void, Error>
run_oobe_full(HostExecutor& host,
              SystemProbe& probe,
              DoctorProbe& doctor,
              const Manifest& m,
              const OobeDefaults& defaults,
              const InstallContext& ctx);

}  // namespace onebit::cli
