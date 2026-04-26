#include <doctest/doctest.h>

#include "onebit/cli/power.hpp"

using namespace onebit::cli;

TEST_CASE("parse_profile — base names")
{
    CHECK(parse_profile("inference").value() == PowerProfile::Inference);
    CHECK(parse_profile("chat").value()      == PowerProfile::Chat);
    CHECK(parse_profile("idle").value()      == PowerProfile::Idle);
}

TEST_CASE("parse_profile — case insensitive + aliases")
{
    CHECK(parse_profile("INFERENCE").value() == PowerProfile::Inference);
    CHECK(parse_profile("Decode").value()    == PowerProfile::Inference);
    CHECK(parse_profile("balanced").value()  == PowerProfile::Chat);
    CHECK(parse_profile("silent").value()    == PowerProfile::Idle);
}

TEST_CASE("parse_profile — bogus rejects with invalid kind")
{
    auto err = parse_profile("bogus");
    REQUIRE_FALSE(err.has_value());
    CHECK(err.error().kind == ErrorKind::InvalidArgument);
}

TEST_CASE("ryzenadj_argv — inference matches design doc")
{
    auto v = ryzenadj_argv(PowerProfile::Inference);
    REQUIRE(v.size() == 4);
    CHECK(v[0] == "--tctl-temp=95");
    CHECK(v[1] == "--slow-limit=75000");
    CHECK(v[2] == "--fast-limit=80000");
    CHECK(v[3] == "--stapm-limit=65000");
}

TEST_CASE("list_profiles — stable order")
{
    auto v = list_profiles();
    REQUIRE(v.size() == 3);
    CHECK(v[0] == PowerProfile::Inference);
    CHECK(v[1] == PowerProfile::Chat);
    CHECK(v[2] == PowerProfile::Idle);
}

TEST_CASE("summarize_info — extracts the four headline fields")
{
    constexpr const char* kSample =
        "CPU Family        |      Strix Halo\n"
        "STAPM LIMIT       |     45.000\n"
        "PPT LIMIT FAST    |     65.000\n"
        "PPT LIMIT SLOW    |     55.000\n"
        "THM LIMIT CORE    |     90.000\n";
    auto s = summarize_info(kSample);
    CHECK(s.find("stapm=45.000") != std::string::npos);
    CHECK(s.find("fast=65.000")  != std::string::npos);
    CHECK(s.find("slow=55.000")  != std::string::npos);
    CHECK(s.find("tctl=90.000")  != std::string::npos);
}
