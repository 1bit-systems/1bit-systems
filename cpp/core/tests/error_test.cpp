#include <doctest/doctest.h>

#include "onebit/core/error.hpp"

#include <string>

using onebit::core::HaloError;

TEST_CASE("error: io message includes path and ec")
{
    auto e = HaloError::io("/tmp/missing.h1b",
                           std::make_error_code(std::errc::no_such_file_or_directory));
    const std::string m = e.what();
    CHECK(m.find("/tmp/missing.h1b") != std::string::npos);
}

TEST_CASE("error: bad-magic prints both magics")
{
    auto e = HaloError::bad_magic({0x48, 0x31, 0x42, 0x00},
                                  {0xDE, 0xAD, 0xBE, 0xEF});
    const std::string m = e.what();
    CHECK(m.find("48314200") != std::string::npos);
    CHECK(m.find("deadbeef") != std::string::npos);
}

TEST_CASE("error: unsupported version mentions range")
{
    auto e = HaloError::unsupported_version(99, 1, 4);
    const std::string m = e.what();
    CHECK(m.find("99") != std::string::npos);
    CHECK(m.find("1") != std::string::npos);
    CHECK(m.find("4") != std::string::npos);
}

TEST_CASE("error: truncated mentions byte counts")
{
    auto e = HaloError::truncated(128, 1024, 256);
    const std::string m = e.what();
    CHECK(m.find("128")  != std::string::npos);
    CHECK(m.find("1024") != std::string::npos);
    CHECK(m.find("256")  != std::string::npos);
}

TEST_CASE("error: invalid_config preserves message")
{
    auto e = HaloError::invalid_config("bad rope theta");
    const std::string m = e.what();
    CHECK(m.find("bad rope theta") != std::string::npos);
}

TEST_CASE("error: sampler preserves message")
{
    auto e = HaloError::sampler("empty logits");
    const std::string m = e.what();
    CHECK(m.find("empty logits") != std::string::npos);
}
