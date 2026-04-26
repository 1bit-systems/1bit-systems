#include <doctest/doctest.h>

#include "onebit/core/h1b.hpp"

using namespace onebit::core::h1b;

TEST_CASE("h1b: open on missing path returns Io")
{
    auto r = File::open("/tmp/__nope_does_not_exist__.h1b");
    REQUIRE_FALSE(r.has_value());
}

TEST_CASE("h1b: format flags map to enum")
{
    Config c;
    c.flags = FLAG_HADAMARD_ROTATED;
    File   f; // default
    (void)f;  // smoke
    CHECK((c.flags & FLAG_HADAMARD_ROTATED) != 0u);
    CHECK((c.flags & FLAG_SHERRY_FP16) == 0u);
}

TEST_CASE("h1b: layer offsets default zero")
{
    LayerOffsets lo;
    CHECK(lo.attn_q   == 0u);
    CHECK(lo.ffn_norm == 0u);
}

TEST_CASE("h1b: magic constant")
{
    CHECK(MAGIC[0] == 'H');
    CHECK(MAGIC[1] == '1');
    CHECK(MAGIC[2] == 'B');
    CHECK(MAGIC[3] == 0u);
}

TEST_CASE("h1b: bonsai group size")
{
    CHECK(BONSAI_GROUP_SIZE == 32u);
}
