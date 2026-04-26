#pragma once

// `1bit doctor` — health probes. Sysfs-touching probes take a `root` path
// so unit tests can pass a tempdir; production callers pass "/".

#include <cstdint>
#include <filesystem>
#include <string>

namespace onebit::cli {

enum class DoctorOutcome : std::uint8_t { Ok = 0, Warn = 1, Fail = 2 };

[[nodiscard]] constexpr const char* outcome_glyph(DoctorOutcome o) noexcept
{
    switch (o) {
        case DoctorOutcome::Ok:   return "OK";
        case DoctorOutcome::Warn: return "WARN";
        case DoctorOutcome::Fail: return "FAIL";
    }
    return "?";
}

struct ProbeResult {
    DoctorOutcome outcome;
    std::string   detail;
};

[[nodiscard]] ProbeResult npu_probe    (const std::filesystem::path& root);
[[nodiscard]] ProbeResult xe2_probe    (const std::filesystem::path& root);
[[nodiscard]] ProbeResult gfx1201_probe(const std::filesystem::path& root);

// `1bit doctor` wired to the host. Prints to stdout, returns the process
// exit code (0 green, 1 warn, 2 fail).
[[nodiscard]] int run_doctor();

// Silent (warn, fail) tally for the OOBE anchor #7 hook.
[[nodiscard]] std::pair<std::uint32_t, std::uint32_t> tally_for_oobe();

}  // namespace onebit::cli
