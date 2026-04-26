#include "onebit/cli/oobe_error.hpp"

#include <fmt/core.h>

namespace onebit::cli {

OobeError OobeError::kernel_too_new(std::string_view current)
{
    return {
        "Kernel is too new for Strix Halo OOBE.",
        fmt::format("Linux 6.18-lts is the recommended baseline. Detected: {}.",
                    current),
        "Boot snapper snapshot #6 or install linux-lts and reboot.",
        "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#kernel-too-new",
        "Next: `1bit rollback` to pick a tested snapshot, then reboot.",
    };
}

OobeError OobeError::rocm_missing()
{
    return {
        "ROCm 7.x userspace not detected.",
        "rocminfo reachable on $PATH with a gfx1151 or gfx1201 agent.",
        "sudo pacman -S rocm-hip-runtime rocminfo",
        "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#rocm-missing",
        "Next: install ROCm via install.sh or pacman, then re-run `1bit install --oobe`.",
    };
}

OobeError OobeError::disk_too_small(std::uint64_t free_gb)
{
    return {
        "Free disk is below the 10 GB OOBE floor.",
        fmt::format("≥ 10 GB free on the install target. Detected: {} GB.",
                    free_gb),
        "df -h / ; remove old caches under ~/.cargo/registry and ~/.halo/logs",
        "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#disk-too-small",
        "Next: free up space, then re-run `1bit install --oobe`.",
    };
}

OobeError OobeError::ram_too_small(std::uint64_t have_gb, std::uint64_t floor_gb)
{
    return {
        "RAM is below the OOBE minimum.",
        fmt::format("≥ {} GB RAM for halo-v2 at Q4_K_M. Detected: {} GB.",
                    floor_gb, have_gb),
        "Close other tenants or choose a smaller default model (Qwen3-4B Q4).",
        "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#ram-too-small",
        "Next: pick a smaller model with `1bit install --oobe core` on a Q4 profile.",
    };
}

OobeError OobeError::doctor_failed(std::uint32_t fail_count)
{
    return {
        "`1bit doctor` reported one or more failing probes after install.",
        fmt::format("All `1bit doctor` probes green (or WARN). Detected: {} FAIL.",
                    fail_count),
        "1bit doctor",
        "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#doctor-failed",
        "Next: run `1bit doctor` to see which rows are red and follow each row's fix.",
    };
}

OobeError OobeError::snapper_absent()
{
    return {
        "`snapper` is not installed or not on $PATH.",
        "snapper ≥ 0.10 available on $PATH (btrfs + snapper on CachyOS).",
        "sudo pacman -S snapper && sudo snapper -c root create-config /",
        "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#snapper-absent",
        "Next: install snapper, or pass a manual snapshot number via `1bit rollback <N>`.",
    };
}

OobeError OobeError::no_rollback_candidate()
{
    return {
        "No `.halo-preinstall` snapper snapshot found.",
        "At least one snapshot labelled `.halo-preinstall` in `snapper list`.",
        "sudo snapper -c root list | grep .halo-preinstall",
        "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#no-rollback-candidate",
        "Next: pass a snapshot number explicitly, e.g. `1bit rollback 6`.",
    };
}

OobeError OobeError::install_step_failed(std::string_view step)
{
    return {
        "An install step failed; best-effort rollback was attempted.",
        fmt::format("Clean completion of step `{}`.", step),
        "Re-run `1bit install --oobe` after addressing the cause printed above.",
        "https://github.com/bong-water-water-bong/1bit-systems/wiki/Troubleshooting#install-step-failed",
        "Next: read the `left state:` line above, then re-run `1bit install --oobe`.",
    };
}

std::ostream& operator<<(std::ostream& out, const OobeError& e)
{
    out << "  what     : " << e.what     << '\n';
    out << "  expected : " << e.expected << '\n';
    out << "  fix      : " << e.repro    << '\n';
    out << "  wiki     : " << e.wiki_link << '\n';
    out << "  next     : " << e.next_step;
    return out;
}

}  // namespace onebit::cli
