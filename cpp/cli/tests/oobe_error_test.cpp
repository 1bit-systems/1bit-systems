#include <doctest/doctest.h>

#include "onebit/cli/oobe_error.hpp"

#include <sstream>
#include <vector>

using onebit::cli::OobeError;

TEST_CASE("every constructor populates all five fields")
{
    const std::vector<OobeError> errs = {
        OobeError::kernel_too_new("7.1.0"),
        OobeError::rocm_missing(),
        OobeError::disk_too_small(3),
        OobeError::ram_too_small(32, 64),
        OobeError::doctor_failed(2),
        OobeError::snapper_absent(),
        OobeError::no_rollback_candidate(),
        OobeError::install_step_failed("cargo build"),
    };
    for (const auto& e : errs) {
        CHECK_FALSE(e.what.empty());
        CHECK_FALSE(e.expected.empty());
        CHECK_FALSE(e.repro.empty());
        CHECK(e.wiki_link.starts_with("https://"));
        CHECK(e.wiki_link.find('#') != std::string::npos);
        CHECK_FALSE(e.next_step.empty());
        CHECK(e.next_step.starts_with("Next:"));
    }
}

TEST_CASE("Display block emits exactly five lines in order")
{
    std::ostringstream os;
    os << OobeError::kernel_too_new("7.1.0");
    const std::string out = os.str();

    std::vector<std::string> lines;
    std::size_t pos = 0;
    while (pos < out.size()) {
        const auto eol = out.find('\n', pos);
        if (eol == std::string::npos) {
            lines.push_back(out.substr(pos));
            break;
        }
        lines.push_back(out.substr(pos, eol - pos));
        pos = eol + 1;
    }
    REQUIRE(lines.size() == 5);
    CHECK(lines[0].find("what")     != std::string::npos);
    CHECK(lines[1].find("expected") != std::string::npos);
    CHECK(lines[2].find("fix")      != std::string::npos);
    CHECK(lines[3].find("wiki")     != std::string::npos);
    CHECK(lines[4].find("next")     != std::string::npos);
}
