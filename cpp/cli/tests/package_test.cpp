#include <doctest/doctest.h>

#include "onebit/cli/package.hpp"

#include <cstdlib>

using onebit::cli::expand_placeholder;
using onebit::cli::is_placeholder_seed;
using onebit::cli::PackageOrigin;
using onebit::cli::origin_label;

TEST_CASE("placeholder seed detection")
{
    CHECK(is_placeholder_seed("$USER"));
    CHECK(is_placeholder_seed("$HOME"));
    CHECK_FALSE(is_placeholder_seed("USER"));
    CHECK_FALSE(is_placeholder_seed("$XDG_CONFIG_HOME"));
}

TEST_CASE("origin_label is stable")
{
    CHECK(origin_label(PackageOrigin::Canonical) == "canonical");
    CHECK(origin_label(PackageOrigin::Overlay)   == "overlay");
}

TEST_CASE("expand_placeholder reads the env on $USER")
{
    ::setenv("USER", "halo-test-user", 1);
    CHECK(expand_placeholder("$USER") == "halo-test-user");
}

TEST_CASE("expand_placeholder leaves literals untouched")
{
    CHECK(expand_placeholder("strix") == "strix");
    CHECK(expand_placeholder("/etc/x") == "/etc/x");
}
