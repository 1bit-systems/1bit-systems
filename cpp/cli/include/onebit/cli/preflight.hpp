#pragma once

// OOBE anchor #1 — pre-flight gates. SystemProbe abstracts every host
// fact the gates want, so unit tests inject a FakeProbe with canned
// answers and the real CLI uses RealProbe (uname / rocminfo / statvfs /
// /proc/meminfo / systemctl).

#include "onebit/cli/oobe_error.hpp"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace onebit::cli {

class SystemProbe {
public:
    SystemProbe() = default;
    SystemProbe(const SystemProbe&) = delete;
    SystemProbe& operator=(const SystemProbe&) = delete;
    SystemProbe(SystemProbe&&) noexcept = default;
    SystemProbe& operator=(SystemProbe&&) noexcept = default;
    virtual ~SystemProbe() = default;

    [[nodiscard]] virtual std::string  kernel_release()  = 0;
    [[nodiscard]] virtual bool         rocminfo_ok()     = 0;
    [[nodiscard]] virtual bool         systemd_user_ok() = 0;
    [[nodiscard]] virtual std::uint64_t disk_free_gb()   = 0;
    [[nodiscard]] virtual std::uint64_t ram_total_gb()   = 0;
};

class RealProbe : public SystemProbe {
public:
    [[nodiscard]] std::string  kernel_release()  override;
    [[nodiscard]] bool         rocminfo_ok()     override;
    [[nodiscard]] bool         systemd_user_ok() override;
    [[nodiscard]] std::uint64_t disk_free_gb()   override;
    [[nodiscard]] std::uint64_t ram_total_gb()   override;
};

struct PreflightPass { std::string note; };
struct PreflightSkip { std::string note; };
struct PreflightFail { OobeError   err;  };

using PreflightOutcome = std::variant<PreflightPass,
                                      PreflightSkip,
                                      PreflightFail>;

[[nodiscard]] inline bool is_green(const PreflightOutcome& o) noexcept
{
    return !std::holds_alternative<PreflightFail>(o);
}

struct GateResult {
    const char*       name;
    PreflightOutcome  outcome;
};

[[nodiscard]] PreflightOutcome gate_kernel(SystemProbe& probe);
[[nodiscard]] PreflightOutcome gate_rocm  (SystemProbe& probe);
[[nodiscard]] PreflightOutcome gate_disk  (SystemProbe& probe);
[[nodiscard]] PreflightOutcome gate_ram   (SystemProbe& probe);

[[nodiscard]] std::vector<GateResult> run_all(SystemProbe& probe);

[[nodiscard]] constexpr std::uint64_t default_ram_floor_gb() noexcept { return 64; }

}  // namespace onebit::cli
