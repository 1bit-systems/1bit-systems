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
if(ONEBIT_BUILD_ROCM_CPP)
    if(EXISTS "$ENV{HOME}/therock/build/dist/rocm/lib/llvm/bin/clang++")
        message(STATUS "rocm-cpp: building in-tree against TheRock toolchain.")
        add_subdirectory(${CMAKE_SOURCE_DIR}/../rocm-cpp ${CMAKE_BINARY_DIR}/rocm-cpp)
    else()
        message(WARNING
            "ONEBIT_BUILD_ROCM_CPP=ON but ~/therock/build/dist/rocm not found. "
            "Build TheRock first (see project_therock_watch.md) or set OFF.")
    endif()
endif()

if(ONEBIT_BUILD_NPU)
    add_subdirectory(aie)
endif()
add_subdirectory(ingest)
add_subdirectory(stream)
add_subdirectory(tier-mint)

if(ONEBIT_BUILD_DESKTOP)
    add_subdirectory(helm)
    add_subdirectory(helm-tui)
endif()

add_subdirectory(cli)

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
        aie stream ingest retrieval tier-mint tools)

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
