#pragma once

// RyzenAdj backend abstraction.
//
// Two implementations:
//
//   * ShelloutBackend — invokes `/usr/bin/ryzenadj`. Mirrors the Rust
//     crate's default behaviour bit-for-bit so existing scripts/tests
//     keep working.
//
//   * LibBackend — dlopen()s `libryzenadj.so` and calls the C API
//     directly via a small struct of function pointers. Falls back
//     gracefully (returns NotAvailable) when the .so or required
//     symbols are missing.
//
// Selection is up to the caller; the CLI defaults to ShelloutBackend
// to preserve Rust parity, with `--backend lib` flipping to LibBackend.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "onebit/power/profile.hpp"
#include "onebit/power/result.hpp"

namespace onebit::power {

// Canonical RyzenAdj knob names — matches `--<flag>=<value>` CLI surface.
inline constexpr std::array<std::string_view, 8> KNOB_NAMES{
    "stapm-limit",        "fast-limit",        "slow-limit",
    "tctl-temp",          "vrm-current",       "vrmmax-current",
    "vrmsoc-current",     "vrmsocmax-current",
};

[[nodiscard]] bool is_known_knob(std::string_view key) noexcept;

class PowerBackend {
public:
    virtual ~PowerBackend() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual Status apply_profile(const Profile& p) = 0;
    [[nodiscard]] virtual Status set_one(std::string_view key, std::uint32_t value) = 0;
};

// ---------------------------------------------------------------------
// ShelloutBackend — fork/exec of /usr/bin/ryzenadj. Mirrors Rust default.
// ---------------------------------------------------------------------

class ShelloutBackend final : public PowerBackend {
public:
    explicit ShelloutBackend(bool dry_run, std::string path = "/usr/bin/ryzenadj")
        : dry_run_(dry_run), path_(std::move(path)) {}

    [[nodiscard]] std::string_view name() const noexcept override { return "shellout"; }
    [[nodiscard]] Status apply_profile(const Profile& p) override;
    [[nodiscard]] Status set_one(std::string_view key, std::uint32_t value) override;

    // Used by tests + dry-run path to inspect the args we would have spawned.
    [[nodiscard]] static std::vector<std::string> profile_to_args(const Profile& p);

    [[nodiscard]] bool dry_run() const noexcept { return dry_run_; }
    [[nodiscard]] std::string_view exec_path() const noexcept { return path_; }

private:
    [[nodiscard]] Status run(const std::vector<std::string>& args) const;

    bool        dry_run_;
    std::string path_;
};

// ---------------------------------------------------------------------
// LibBackend — dlopen(libryzenadj.so) + dlsym() the C API.
// ---------------------------------------------------------------------
//
// The handle is opaque (FlyGoat/RyzenAdj typedefs `ryzen_access` to
// `struct ryzen_access *`). We never dereference it on the C++ side.
// Function-pointer table mirrors the upstream public surface we need.
//
// Symbol stability: these names have been stable since at least 0.11.0
// (2021) and are part of FlyGoat's documented C ABI.

namespace detail {

using ryzen_access = void*;     // opaque C handle
using init_fn      = ryzen_access (*)();
using cleanup_fn   = void (*)(ryzen_access);
using set_u32_fn   = int  (*)(ryzen_access, std::uint32_t);

struct LibSymbols {
    init_fn    init_ryzenadj    = nullptr;
    cleanup_fn cleanup_ryzenadj = nullptr;

    set_u32_fn set_stapm_limit       = nullptr;
    set_u32_fn set_fast_limit        = nullptr;
    set_u32_fn set_slow_limit        = nullptr;
    set_u32_fn set_tctl_temp         = nullptr;
    set_u32_fn set_vrm_current       = nullptr;
    set_u32_fn set_vrmmax_current    = nullptr;
    set_u32_fn set_vrmsoc_current    = nullptr;
    set_u32_fn set_vrmsocmax_current = nullptr;

    [[nodiscard]] bool valid() const noexcept
    {
        return init_ryzenadj && cleanup_ryzenadj
            && set_stapm_limit && set_fast_limit && set_slow_limit
            && set_tctl_temp
            && set_vrm_current && set_vrmmax_current
            && set_vrmsoc_current && set_vrmsocmax_current;
    }
};

// Closes the dlopen handle on destruction. void* keeps the public
// header free of <dlfcn.h>.
struct DlHandleDeleter {
    void operator()(void* h) const noexcept;
};

using DlHandle = std::unique_ptr<void, DlHandleDeleter>;

} // namespace detail

class LibBackend : public PowerBackend {
public:
    // dlopen libryzenadj.so. Returns NotAvailable if the .so isn't present
    // or any required symbol is missing — never aborts the process.
    static Result<std::unique_ptr<LibBackend>> open(bool dry_run,
                                                    std::string_view soname = "libryzenadj.so");

    ~LibBackend() override;

    LibBackend(const LibBackend&)            = delete;
    LibBackend& operator=(const LibBackend&) = delete;
    LibBackend(LibBackend&&)                 = delete;
    LibBackend& operator=(LibBackend&&)      = delete;

    [[nodiscard]] std::string_view name() const noexcept override { return "libryzenadj"; }
    [[nodiscard]] Status apply_profile(const Profile& p) override;
    [[nodiscard]] Status set_one(std::string_view key, std::uint32_t value) override;

    [[nodiscard]] bool dry_run() const noexcept { return dry_run_; }

protected:
    LibBackend(detail::DlHandle h,
               detail::LibSymbols syms,
               detail::ryzen_access access,
               bool dry_run)
        : dl_(std::move(h)), sym_(syms), access_(access), dry_run_(dry_run) {}

private:

    [[nodiscard]] Status invoke(detail::set_u32_fn fn,
                                std::string_view label,
                                std::uint32_t value) const;

    detail::DlHandle      dl_;
    detail::LibSymbols    sym_{};
    detail::ryzen_access  access_{nullptr};
    bool                  dry_run_{false};
};

} // namespace onebit::power
