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

# -----------------------------------------------------------------------------
# onebit_aie_bitnet_gemm — direct-link libxrt wrapper around the authored
# AIE2P ternary-GEMM xclbin (bitnet_gemm_bf16, M=K=N=64).
#
# This is a *separate* library from onebit_aie_lib above. The dlopen path
# stays for the lemonade-routed BitNet GEMV; this lib is the ORT/Engine
# direct-call lane that mirrors run_pyxrt_bitnet.py.
#
# Gating: ONEBIT_BUILD_XDNA (top-level option) OR BUILD_XDNA (inherited
# from rocm-cpp when ONEBIT_BUILD_ROCM_CPP=ON). Either turns it on.
# When neither is set, the library still builds but every entry point
# returns LibraryUnavailable — same compile-clean stance as onebit::onnx.
# -----------------------------------------------------------------------------
option(ONEBIT_BUILD_XDNA "Build XDNA NPU lane (link libxrt directly)" OFF)

set(_onebit_bitnet_gemm_xrt_enabled FALSE)
if(ONEBIT_BUILD_XDNA OR BUILD_XDNA)
    if(NOT UNIX OR APPLE)
        message(FATAL_ERROR
            "ONEBIT_BUILD_XDNA requires Linux + libxrt (amdxdna driver)")
    endif()
    set(_onebit_bitnet_gemm_xrt_enabled TRUE)
endif()

set(_onebit_bitnet_gemm_xrt_inc "")
set(_onebit_bitnet_gemm_xrt_libs "")
if(_onebit_bitnet_gemm_xrt_enabled)
    # Same probe pattern as rocm-cpp/CMakeLists.txt:350 — pkg-config first,
    # manual /usr/lib fallback for Arch which ships xrt without a .pc file.
    find_package(PkgConfig QUIET)
    set(_xrt_via_pc FALSE)
    if(PkgConfig_FOUND)
        pkg_check_modules(_PC_XRT QUIET xrt)
        if(_PC_XRT_FOUND)
            set(_onebit_bitnet_gemm_xrt_inc  ${_PC_XRT_INCLUDE_DIRS})
            set(_onebit_bitnet_gemm_xrt_libs ${_PC_XRT_LIBRARIES})
            set(_xrt_via_pc TRUE)
        endif()
    endif()

    if(NOT _xrt_via_pc)
        if(NOT EXISTS "/usr/include/xrt/xrt_kernel.h")
            message(FATAL_ERROR
                "ONEBIT_BUILD_XDNA: /usr/include/xrt/xrt_kernel.h missing. "
                "Install xrt (Arch/CachyOS: pacman -S xrt).")
        endif()
        if(NOT EXISTS "/usr/lib/libxrt_coreutil.so")
            message(FATAL_ERROR
                "ONEBIT_BUILD_XDNA: /usr/lib/libxrt_coreutil.so missing.")
        endif()
        set(_onebit_bitnet_gemm_xrt_inc  "/usr/include")

        # Required: xrt_coreutil. Optional: xrt_driver_xdna (when xrt was
        # built with the AMDXDNA driver in-tree — Arch packaging does this).
        # The C++ surface in xrt_kernel.h links against coreutil only; the
        # xdna driver lib is loaded by coreutil at runtime via dlopen, so
        # we don't need to put it on the link line. We DO add it as a
        # secondary lib when present so static analysis tools see it.
        set(_onebit_bitnet_gemm_xrt_libs "/usr/lib/libxrt_coreutil.so")
        if(EXISTS "/usr/lib/libxrt_driver_xdna.so.2")
            list(APPEND _onebit_bitnet_gemm_xrt_libs
                 "/usr/lib/libxrt_driver_xdna.so.2")
        endif()
    endif()

    message(STATUS
        "onebit::aie::bitnet_gemm — XDNA direct-link ENABLED "
        "(includes=${_onebit_bitnet_gemm_xrt_inc} libs=${_onebit_bitnet_gemm_xrt_libs})")
else()
    message(STATUS
        "onebit::aie::bitnet_gemm — XDNA direct-link DISABLED "
        "(set -DONEBIT_BUILD_XDNA=ON to wire libxrt). "
        "Library still builds; entry points return LibraryUnavailable.")
endif()

add_library(onebit_aie_bitnet_gemm STATIC
    src/bitnet_gemm_aie2p.cpp
)
target_include_directories(onebit_aie_bitnet_gemm PUBLIC include)
target_compile_features(onebit_aie_bitnet_gemm PUBLIC cxx_std_23)
# Pull in onebit::aie for label() / Error::what() (defined in src/error.cpp
# under that target). Avoids ODR-bloat from re-linking error.cpp here.
target_link_libraries(onebit_aie_bitnet_gemm PUBLIC onebit::aie)
add_library(onebit::aie_bitnet_gemm ALIAS onebit_aie_bitnet_gemm)

if(_onebit_bitnet_gemm_xrt_enabled)
    target_compile_definitions(onebit_aie_bitnet_gemm PRIVATE
        ONEBIT_AIE_HAVE_XRT_RT=1)
    target_include_directories(onebit_aie_bitnet_gemm SYSTEM PRIVATE
        ${_onebit_bitnet_gemm_xrt_inc})
    target_link_libraries(onebit_aie_bitnet_gemm PRIVATE
        ${_onebit_bitnet_gemm_xrt_libs})
else()
    target_compile_definitions(onebit_aie_bitnet_gemm PRIVATE
        ONEBIT_AIE_HAVE_XRT_RT=0)
endif()

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_aie_bitnet_gemm_test
        tests/bitnet_gemm_aie2p_test.cpp
    )
    target_link_libraries(onebit_aie_bitnet_gemm_test PRIVATE
        onebit::aie_bitnet_gemm doctest::doctest)
    add_test(NAME onebit_aie_bitnet_gemm
             COMMAND onebit_aie_bitnet_gemm_test)
endif()

# -----------------------------------------------------------------------------
# onebit_aie_tiled_gemv — host-side pad-and-tile dispatcher around the
# fixed-tile BitnetGemmAIE2P leaf. Pure C++23, no XRT linkage of its own
# (the kernel reference does the dispatch). Builds clean on any host that
# can build onebit_aie_bitnet_gemm.
# -----------------------------------------------------------------------------
add_library(onebit_aie_tiled_gemv STATIC
    src/tiled_gemv.cpp
)
target_include_directories(onebit_aie_tiled_gemv PUBLIC include)
target_compile_features(onebit_aie_tiled_gemv PUBLIC cxx_std_23)
target_link_libraries(onebit_aie_tiled_gemv PUBLIC onebit::aie_bitnet_gemm)
add_library(onebit::aie_tiled_gemv ALIAS onebit_aie_tiled_gemv)

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_aie_tiled_gemv_test
        tests/tiled_gemv_test.cpp
    )
    target_link_libraries(onebit_aie_tiled_gemv_test PRIVATE
        onebit::aie_tiled_gemv doctest::doctest)
    add_test(NAME onebit_aie_tiled_gemv
             COMMAND onebit_aie_tiled_gemv_test)

    # dispatch_smoke_test — engine-level dispatch contract:
    #   * row-scale fold post-mmul (Phase-2 contract)
    #   * handle reuse across consecutive gemm() calls
    # Skips with WARN when ONEBIT_REAL_BACKEND=0 or fixture xclbin missing
    # so CI hosts without /dev/accel/accel0 still pass green.
    add_executable(onebit_aie_dispatch_smoke_test
        tests/dispatch_smoke_test.cpp
    )
    target_link_libraries(onebit_aie_dispatch_smoke_test PRIVATE
        onebit::aie_bitnet_gemm doctest::doctest)
    add_test(NAME onebit_aie_dispatch_smoke
             COMMAND onebit_aie_dispatch_smoke_test)
endif()
