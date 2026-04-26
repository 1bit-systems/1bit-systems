# cpp/aie — NPU runtime: dlopen(libxrt_coreutil.so) + xclbin loading
# + kernel dispatch. AIE2P (XDNA2) on Strix Halo's `npu5` is the
# primary target.
#
# Hard rule: NO link-time dependency on libxrt. We dlopen at runtime
# and resolve symbols via dlsym. This means the binary builds clean on
# CI hosts without libxrt installed; the operator only needs libxrt
# present at runtime to actually dispatch a kernel.
#
# We do NOT FetchContent the XRT tree — its build needs the AMD-public
# Vitis tooling. This is the same deferral pattern used by the retired
# `1bit-aie` Rust crate (`real-npu` feature flag).
#
# System-lib pins captured from retired `crates/1bit-aie/build.rs`
# (deleted 2026-04-26 in the Rust gut):
#   - libxrt_coreutil  (Arch/CachyOS path: /usr/lib/libxrt_coreutil.so.2)
#   - libxrt++         (Arch/CachyOS path: /usr/lib/libxrt++.so.2)
# No version constraint was pinned on the Rust side either; we go off
# whatever `xrt` package CachyOS ships and gate behavior on the dlsym
# probe at runtime. Any pin tighter than that belongs at packaging
# time (PKGBUILD `depends=('xrt')`), not here.

# Always-on library: trait + types + dlopen-backed XrtBackend that
# returns NotYetWired when libxrt is absent.
add_library(onebit_aie_lib STATIC
    src/error.cpp
    src/backend.cpp
    src/dyloader.cpp
)
target_include_directories(onebit_aie_lib PUBLIC include)
target_compile_features(onebit_aie_lib PUBLIC cxx_std_23)
add_library(onebit::aie ALIAS onebit_aie_lib)

# dl on Linux. macOS has it in libc; we don't target macOS for the NPU
# lane so just link -ldl unconditionally on POSIX.
if(UNIX AND NOT APPLE)
    target_link_libraries(onebit_aie_lib PUBLIC ${CMAKE_DL_LIBS})
endif()

# Optional: detect libxrt headers for compile-time enums (status codes,
# bo flag values). We don't link against any libxrt symbols — that's
# the dlopen path's job — but we do consume the public xrt error code
# constants when the headers are present, falling back to documented
# defaults otherwise.
find_path(XRT_INCLUDE_DIR
    NAMES xrt/xrt.h
    HINTS
        /opt/xilinx/xrt/include
        /usr/include
        /usr/local/include
        $ENV{XILINX_XRT}/include
)

if(XRT_INCLUDE_DIR)
    message(STATUS "onebit::aie — XRT headers at ${XRT_INCLUDE_DIR} (used for enums only)")
    target_include_directories(onebit_aie_lib PUBLIC ${XRT_INCLUDE_DIR})
    target_compile_definitions(onebit_aie_lib PUBLIC ONEBIT_AIE_HAVE_XRT_HEADERS=1)
else()
    message(STATUS "onebit::aie — XRT headers absent; using built-in error codes. "
                   "Runtime dlopen probe is independent of headers.")
    target_compile_definitions(onebit_aie_lib PUBLIC ONEBIT_AIE_HAVE_XRT_HEADERS=0)
endif()

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_aie_test
        tests/error_test.cpp
        tests/backend_test.cpp
        tests/dyloader_test.cpp
    )
    target_link_libraries(onebit_aie_test PRIVATE onebit::aie doctest::doctest)
    add_test(NAME onebit_aie COMMAND onebit_aie_test)
endif()
