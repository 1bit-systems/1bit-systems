// error_test.cpp — labels + what() for aie::Error.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "onebit/aie/error.hpp"
#include "onebit/aie/types.hpp"

using namespace onebit::aie;

TEST_CASE("aie labels stable")
{
    CHECK(label(ErrorKind::XclbinNotFound)     == "xclbin_not_found");
    CHECK(label(ErrorKind::ShapeMismatch)      == "shape_mismatch");
    CHECK(label(ErrorKind::DtypeMismatch)      == "dtype_mismatch");
    CHECK(label(ErrorKind::Xrt)                == "xrt");
    CHECK(label(ErrorKind::NotYetWired)        == "not_yet_wired");
    CHECK(label(ErrorKind::LibraryUnavailable) == "library_unavailable");
}

TEST_CASE("Xrt error code threaded through what()")
{
    Error e{ErrorKind::Xrt, "BO sync failed", -7};
    auto s = e.what();
    CHECK(s.find("xrt")             != std::string::npos);
    CHECK(s.find("BO sync failed")  != std::string::npos);
    CHECK(s.find("-7")              != std::string::npos);
    CHECK(e.xrt_code() == -7);
}

TEST_CASE("non-Xrt errors do not include code in what()")
{
    Error e{ErrorKind::NotYetWired, "load_xclbin"};
    auto s = e.what();
    CHECK(s.find("(code")  == std::string::npos);
}

TEST_CASE("Dtype labels stable")
{
    CHECK(label(Dtype::PackedT2) == "packed_t2");
    CHECK(label(Dtype::I8)       == "i8");
    CHECK(label(Dtype::I32)      == "i32");
    CHECK(label(Dtype::Bf16)     == "bf16");
}

TEST_CASE("Buffer / KernelHandle equality")
{
    Buffer a{1, 64, Dtype::I8};
    Buffer b{1, 64, Dtype::I8};
    Buffer c{2, 64, Dtype::I8};
    CHECK(a == b);
    CHECK_FALSE(a == c);
    CHECK(KernelHandle{0} == KernelHandle{0});
    CHECK_FALSE(KernelHandle{0} == KernelHandle{1});
}

TEST_CASE("dtype enum values do not silently collide")
{
    static_assert(static_cast<int>(Dtype::PackedT2) == 0);
    static_assert(static_cast<int>(Dtype::I8)       == 1);
    static_assert(static_cast<int>(Dtype::I32)      == 2);
    static_assert(static_cast<int>(Dtype::Bf16)     == 3);
    CHECK(true);
}
