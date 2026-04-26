// backend_test.cpp — StubBackend + XrtBackend behavioural tests.
//
// All tests pass without libxrt installed because XrtBackend falls
// back to LibraryUnavailable / NotYetWired in that case.

#include <doctest/doctest.h>

#include "onebit/aie/backend.hpp"

#include <filesystem>
#include <fstream>

using namespace onebit::aie;
namespace fs = std::filesystem;

namespace {
fs::path make_tmp_xclbin() {
    auto p = fs::temp_directory_path() / "onebit_aie_test.xclbin";
    std::ofstream{p, std::ios::binary} << "fake-xclbin";
    return p;
}
} // namespace

TEST_CASE("StubBackend refuses every dispatch op with NotYetWired")
{
    StubBackend be;

    auto h = be.load_xclbin("/nonexistent/halo_bitnet.xclbin");
    REQUIRE_FALSE(h.has_value());
    CHECK(h.error().kind() == ErrorKind::NotYetWired);

    Buffer w{0, 64, Dtype::PackedT2};
    Buffer x{1, 64, Dtype::I8};
    Buffer s{2, 4,  Dtype::Bf16};
    Buffer o{3, 32, Dtype::Bf16};
    auto r = be.bitnet_gemv(KernelHandle{0}, w, x, o, s);
    REQUIRE_FALSE(r.has_value());
    CHECK(r.error().kind() == ErrorKind::NotYetWired);
}

TEST_CASE("StubBackend::device_info returns deterministic placeholder")
{
    StubBackend be;
    auto info = be.device_info();
    CHECK(info.device_name      == "stub");
    CHECK(info.firmware_version == "0.0.0-stub");
    CHECK(info.columns          == 0);
    CHECK(info.tile_class       == "AIE2P");
}

TEST_CASE("XrtBackend probe_runtime reports loadable / unloadable cleanly")
{
    XrtBackend be;
    auto r = be.probe_runtime();
    if (r.has_value()) {
        CHECK_FALSE(r->empty());
        DOCTEST_WARN_MESSAGE(true, "libxrt loaded — real probe exercised");
    } else {
        CHECK(r.error().kind() == ErrorKind::LibraryUnavailable);
        DOCTEST_WARN_MESSAGE(true, "libxrt absent — fallback exercised (expected)");
    }
}

TEST_CASE("XrtBackend::load_xclbin reports XclbinNotFound for missing file")
{
    XrtBackend be;
    auto rt = be.probe_runtime();
    if (!rt.has_value()) {
        // Without libxrt loaded we get LibraryUnavailable instead of
        // XclbinNotFound — both are valid stops on the failure path.
        auto h = be.load_xclbin("/no/such/file.xclbin");
        REQUIRE_FALSE(h.has_value());
        CHECK(h.error().kind() == ErrorKind::LibraryUnavailable);
        return;
    }
    auto h = be.load_xclbin("/no/such/file.xclbin");
    REQUIRE_FALSE(h.has_value());
    // libxrt loaded — the missing-file branch fires.
    CHECK(h.error().kind() == ErrorKind::XclbinNotFound);
}

TEST_CASE("XrtBackend::load_xclbin on a real file returns NotYetWired or LibraryUnavailable")
{
    XrtBackend be;
    auto p = make_tmp_xclbin();
    auto h = be.load_xclbin(p);
    REQUIRE_FALSE(h.has_value());
    auto k = h.error().kind();
    CHECK((k == ErrorKind::NotYetWired || k == ErrorKind::LibraryUnavailable));
    fs::remove(p);
}

TEST_CASE("make_default_backend returns a usable Backend")
{
    auto be = make_default_backend();
    REQUIRE(be != nullptr);
    auto info = be->device_info();
    // Either "stub" (no libxrt) or "RyzenAI-npu5" (loaded). Both are
    // strings; just confirm it's non-empty.
    CHECK_FALSE(info.device_name.empty());
    CHECK(info.tile_class == "AIE2P");
}

TEST_CASE("Backend dispatch returns NotYetWired or LibraryUnavailable end-to-end")
{
    auto be = make_default_backend();
    Buffer w{0, 64, Dtype::PackedT2};
    Buffer x{1, 64, Dtype::I8};
    Buffer s{2, 4,  Dtype::Bf16};
    Buffer o{3, 32, Dtype::Bf16};
    auto r = be->bitnet_gemv(KernelHandle{0}, w, x, o, s);
    REQUIRE_FALSE(r.has_value());
    auto k = r.error().kind();
    CHECK((k == ErrorKind::NotYetWired || k == ErrorKind::LibraryUnavailable));
}
