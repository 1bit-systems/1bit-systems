// dyloader_test.cpp — exercise the dlopen wrapper without requiring
// libxrt to actually be installed.

#include <doctest/doctest.h>

#include "onebit/aie/dyloader.hpp"

using namespace onebit::aie;

TEST_CASE("DyLoader starts closed")
{
    DyLoader dl;
    CHECK_FALSE(dl.is_open());
    CHECK(dl.soname().empty());
    CHECK(dl.resolve("anything") == nullptr);
}

TEST_CASE("open() on a host without libxrt returns LibraryUnavailable")
{
    DyLoader dl;
    auto r = dl.open();
    if (r.has_value()) {
        // libxrt IS present on this host — exercise the success path.
        CHECK(dl.is_open());
        CHECK_FALSE(dl.soname().empty());
        DOCTEST_WARN_MESSAGE(true, "libxrt loaded — XRT-aware tests live");
    } else {
        CHECK(r.error().kind() == ErrorKind::LibraryUnavailable);
        // detail() should mention each soname we tried.
        CHECK(!r.error().detail().empty());
        DOCTEST_WARN_MESSAGE(true, "libxrt absent — stub fallback exercised (expected)");
    }
}

TEST_CASE("open() is idempotent")
{
    DyLoader dl;
    auto r1 = dl.open();
    auto r2 = dl.open();
    // Both calls have the same outcome.
    CHECK(r1.has_value() == r2.has_value());
    if (r1.has_value() && r2.has_value()) {
        CHECK(*r1 == *r2);
    }
}

TEST_CASE("close() on never-opened loader is a no-op")
{
    DyLoader dl;
    dl.close();
    CHECK_FALSE(dl.is_open());
    // Second close also fine.
    dl.close();
    CHECK_FALSE(dl.is_open());
}

TEST_CASE("move construction transfers ownership")
{
    DyLoader a;
    auto opened = a.open();
    DyLoader b{std::move(a)};
    // a is now empty regardless of whether libxrt was present.
    CHECK_FALSE(a.is_open());
    CHECK(a.soname().empty());
    if (opened.has_value()) {
        CHECK(b.is_open());
        CHECK_FALSE(b.soname().empty());
    }
}

TEST_CASE("resolve() on closed loader returns nullptr")
{
    DyLoader dl;
    CHECK(dl.resolve("xrt_shim_device_open")     == nullptr);
    CHECK(dl.resolve(nullptr)                    == nullptr);
}

TEST_CASE("kXrtSonames table is non-empty")
{
    CHECK(std::size(kXrtSonames) >= 2);
    for (auto s : kXrtSonames) {
        CHECK_FALSE(s.empty());
    }
}
