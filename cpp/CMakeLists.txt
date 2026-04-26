cmake_minimum_required(VERSION 3.24)
project(onebit_systems CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

include(cmake/deps.cmake)

option(ONEBIT_BUILD_DESKTOP   "Build helm desktop (Qt6)" ON)
option(ONEBIT_BUILD_NPU       "Build aie/NPU runtime (libxrt)" ON)
option(ONEBIT_BUILD_TESTS     "Build test suite" ON)
option(ONEBIT_BUILD_TOOLS     "Build offline tools" ON)
# ONEBIT_BUILD_XDNA: link onebit::aie_bitnet_gemm directly against libxrt
# (default OFF — the cpp/aie/CMakeLists.txt holds the matching option).
# Surfacing it here lets the release-strix preset turn it on without
# poking into the aie/ subtree from the preset file.
option(ONEBIT_BUILD_XDNA      "Direct-link libxrt for AIE2P bitnet_gemm" OFF)
# rocm-cpp is the HIP kernel subtree (folded 2026-04-20). Its top-level
# CMakeLists.txt pins TheRock's clang as the HIP compiler. Build it in-tree
# only when the TheRock toolchain is present at ~/therock/build/dist/rocm.
# Default OFF — host without ROCm/TheRock still builds the C++ tower clean.
option(ONEBIT_BUILD_ROCM_CPP  "Build rocm-cpp HIP kernels in-tree (needs TheRock)" OFF)

if(ONEBIT_BUILD_TESTS)
    enable_testing()
endif()

add_compile_options(
    -Wall -Wextra -Wpedantic
    $<$<CONFIG:Release>:-O3>
)

# Crates ported so far. add_subdirectory entries appended as ports land.
add_subdirectory(core)
add_subdirectory(mcp)
add_subdirectory(mcp-clients)
add_subdirectory(mcp-linuxgsm)
add_subdirectory(landing)
add_subdirectory(power)
add_subdirectory(watchdog)
add_subdirectory(halo-ralph)
add_subdirectory(retrieval)
add_subdirectory(voice)
add_subdirectory(echo)
add_subdirectory(onnx)
add_subdirectory(kokoro)

# `aie` MUST be added before `rocm-cpp` so the onebit::aie_bitnet_gemm
# target exists when rocm-cpp's CMakeLists probes for it (the engine NPU
# dispatch shim links against it under -DBUILD_XDNA=ON).  Re-ordering this
# pair triggers a silent build regression: rocm-cpp falls back to the
# "no aie target" branch and engine.cpp loses the short-circuit.
if(ONEBIT_BUILD_NPU)
    add_subdirectory(aie)
endif()

if(ONEBIT_BUILD_ROCM_CPP)
    if(EXISTS "$ENV{HOME}/therock/build/dist/rocm/lib/llvm/bin/clang++")
        message(STATUS "rocm-cpp: building in-tree against TheRock toolchain.")
        # Forward the NPU build flag.  rocm-cpp's CMakeLists keys its
        # XDNA wrapper + engine bridge on BUILD_XDNA, not ONEBIT_BUILD_XDNA.
        if(ONEBIT_BUILD_XDNA OR ONEBIT_BUILD_NPU)
            set(BUILD_XDNA ON CACHE BOOL
                "Forwarded from cpp/ tower for NPU dispatch lane" FORCE)
        endif()
        add_subdirectory(${CMAKE_SOURCE_DIR}/../rocm-cpp ${CMAKE_BINARY_DIR}/rocm-cpp)
    else()
        message(WARNING
            "ONEBIT_BUILD_ROCM_CPP=ON but ~/therock/build/dist/rocm not found. "
            "Build TheRock first (see project_therock_watch.md) or set OFF.")
    endif()
endif()
add_subdirectory(ingest)
add_subdirectory(stream)
add_subdirectory(tier-mint)
# `cli` MUST be added before `agent` — agent's static lib does a
# `target_link_libraries(... PUBLIC onebit::cli)` for the proc helper,
# and CMake 3.28 (Ubuntu 24.04 CI runner) errors on the deferred ALIAS
# lookup if cli's add_subdirectory hasn't run yet.  CMake 4+ tolerates
# the reverse order via deferred resolution; CI doesn't.
add_subdirectory(cli)
add_subdirectory(agent)

if(ONEBIT_BUILD_DESKTOP)
    add_subdirectory(helm)
    add_subdirectory(helm-tui)
    # helm-tauri is opt-in via -DONEBIT_BUILD_HELM_TAURI=ON. Its own
    # CMakeLists.txt gates everything inside that option, so listing
    # the subdirectory unconditionally here is a no-op when OFF.
    add_subdirectory(helm-tauri)
endif()

# Pending ports — add_subdirectory lines uncommented as each crate lands.

if(ONEBIT_BUILD_TOOLS)
    add_subdirectory(tools/h1b-sherry)
    add_subdirectory(tools/gguf-to-h1b)
    add_subdirectory(tools/bitnet-to-tq2)
endif()

# -----------------------------------------------------------------------------
# clang-tidy `tidy` target — Core Guidelines audit across our crates.
#
# Runs the configuration in <cpp>/.clang-tidy against every *.cpp under our
# crate roots, using compile_commands.json from the active build directory.
# FetchContent + third_party trees are excluded by both the file glob and the
# HeaderFilterRegex inside .clang-tidy.
#
# Usage:  cmake --build <build-dir> --target tidy
# -----------------------------------------------------------------------------
find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if(CLANG_TIDY_EXE)
    set(_onebit_tidy_crates
        core mcp cli landing power voice echo helm helm-tui
        watchdog halo-ralph mcp-clients mcp-linuxgsm onnx kokoro
        aie stream ingest retrieval tier-mint tools agent)

    set(_onebit_tidy_sources)
    foreach(_crate IN LISTS _onebit_tidy_crates)
        file(GLOB_RECURSE _crate_sources
            CONFIGURE_DEPENDS
            "${CMAKE_SOURCE_DIR}/${_crate}/*.cpp")
        # Exclude anything that snuck into a build/ or _deps/ subtree.
        list(FILTER _crate_sources EXCLUDE REGEX "/build/")
        list(FILTER _crate_sources EXCLUDE REGEX "/_deps/")
        list(FILTER _crate_sources EXCLUDE REGEX "/third_party/")
        list(APPEND _onebit_tidy_sources ${_crate_sources})
    endforeach()

    if(_onebit_tidy_sources)
        add_custom_target(tidy
            COMMAND ${CLANG_TIDY_EXE}
                    -p ${CMAKE_BINARY_DIR}
                    --config-file=${CMAKE_SOURCE_DIR}/.clang-tidy
                    --quiet
                    ${_onebit_tidy_sources}
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            COMMENT "Running clang-tidy across ported crates"
            VERBATIM
            USES_TERMINAL)
    else()
        message(STATUS "tidy target: no .cpp sources found yet (skipping)")
    endif()
else()
    message(STATUS "clang-tidy not found — `tidy` target unavailable")
endif()
