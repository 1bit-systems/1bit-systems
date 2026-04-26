#pragma once

// Five-line user-facing error block. Every constructor populates all five
// fields (`what`, `expected`, `repro`, `wiki_link`, `next_step`) so the
// renderer never silently drops a row.

#include <ostream>
#include <string>
#include <string_view>

namespace onebit::cli {

struct OobeError {
    std::string what;       ///< one-sentence summary of the failure
    std::string expected;   ///< what the installer wanted
    std::string repro;      ///< exact fix command
    std::string wiki_link;  ///< https URL with #anchor
    std::string next_step;  ///< "Next: …" pointer

    [[nodiscard]] static OobeError kernel_too_new(std::string_view current);
    [[nodiscard]] static OobeError rocm_missing();
    [[nodiscard]] static OobeError disk_too_small(std::uint64_t free_gb);
    [[nodiscard]] static OobeError ram_too_small(std::uint64_t have_gb,
                                                 std::uint64_t floor_gb);
    [[nodiscard]] static OobeError doctor_failed(std::uint32_t fail_count);
    [[nodiscard]] static OobeError snapper_absent();
    [[nodiscard]] static OobeError no_rollback_candidate();
    [[nodiscard]] static OobeError install_step_failed(std::string_view step);
};

// Renders the fixed five-line block to `out`. No trailing newline.
std::ostream& operator<<(std::ostream& out, const OobeError& e);

}  // namespace onebit::cli
