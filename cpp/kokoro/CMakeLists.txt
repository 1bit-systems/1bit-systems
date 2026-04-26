# cpp/kokoro — TTS engine; wraps upstream halo-kokoro (forked from
# olokobayusuf/kokoro.cpp, single-header C++ around onnxruntime).
#
# We do NOT reimplement kokoro.cpp internals here. This component is a
# thin `onebit::kokoro::Engine` facade with input validation + RAII +
# typed errors. The heavy lifting lives in halo-kokoro.
#
# System-lib pins captured from retired `crates/1bit-kokoro/build.rs`
# (deleted 2026-04-26 in the Rust gut):
#   - onnxruntime (no version constraint on the Rust side either —
#     whatever the host package manager / sibling onnxruntime-openvino
#     install ships).
#   - Optional install prefix override via env: ONEBIT_KOKORO_PREFIX.
#     Used on sliger where the OpenVINO-enabled onnxruntime build
#     lives at /opt/onnxruntime-openvino/ for the Arc B580 voice
#     pipeline. CMake honors this via the HALO_KOKORO_DIR cache var
#     below; for the onnxruntime side, append the prefix to
#     CMAKE_PREFIX_PATH at configure time.
#
# Detection strategy:
#   1. Cache var HALO_KOKORO_DIR — operator points us at a checkout.
#   2. Convention: /home/bcloud/repos/halo-kokoro (per project memory).
#   3. Fallback: stub mode. Engine::create returns UnsupportedStub;
#      tests that need real synthesis self-skip via DOCTEST_WARN_MESSAGE.

set(HALO_KOKORO_DIR "" CACHE PATH "Path to halo-kokoro source checkout")

set(_onebit_kokoro_real OFF)
if(NOT HALO_KOKORO_DIR)
    if(EXISTS "/home/bcloud/repos/halo-kokoro/CMakeLists.txt")
        set(HALO_KOKORO_DIR "/home/bcloud/repos/halo-kokoro")
    endif()
endif()

if(HALO_KOKORO_DIR AND EXISTS "${HALO_KOKORO_DIR}/CMakeLists.txt")
    message(STATUS "onebit::kokoro — wrapping halo-kokoro at ${HALO_KOKORO_DIR}")
    add_subdirectory(${HALO_KOKORO_DIR}
                     ${CMAKE_BINARY_DIR}/halo-kokoro
                     EXCLUDE_FROM_ALL)
    set(_onebit_kokoro_real ON)
else()
    message(WARNING "onebit::kokoro — halo-kokoro source not found "
                    "(set HALO_KOKORO_DIR=<path>); building stub engine. "
                    "Real synth path returns UnsupportedStub.")
endif()

add_library(onebit_kokoro_lib STATIC
    src/error.cpp
    src/engine.cpp
)
target_include_directories(onebit_kokoro_lib PUBLIC include)
target_compile_features(onebit_kokoro_lib PUBLIC cxx_std_23)
add_library(onebit::kokoro ALIAS onebit_kokoro_lib)

if(_onebit_kokoro_real)
    target_compile_definitions(onebit_kokoro_lib PUBLIC ONEBIT_KOKORO_HAVE_HALO=1)
    # Try common halo-kokoro target names. If none of them exist the
    # operator can override via HALO_KOKORO_TARGET cache var.
    set(HALO_KOKORO_TARGET "" CACHE STRING "Override halo-kokoro CMake target name")
    if(HALO_KOKORO_TARGET AND TARGET ${HALO_KOKORO_TARGET})
        target_link_libraries(onebit_kokoro_lib PUBLIC ${HALO_KOKORO_TARGET})
    elseif(TARGET halo_kokoro)
        target_link_libraries(onebit_kokoro_lib PUBLIC halo_kokoro)
    elseif(TARGET halo-kokoro)
        target_link_libraries(onebit_kokoro_lib PUBLIC halo-kokoro)
    elseif(TARGET kokoro)
        target_link_libraries(onebit_kokoro_lib PUBLIC kokoro)
    else()
        message(WARNING "onebit::kokoro — halo-kokoro added but no known target name "
                        "found; falling back to stub. Set HALO_KOKORO_TARGET=<name>.")
        target_compile_definitions(onebit_kokoro_lib PUBLIC ONEBIT_KOKORO_HAVE_HALO=0)
        set(_onebit_kokoro_real OFF)
    endif()
else()
    target_compile_definitions(onebit_kokoro_lib PUBLIC ONEBIT_KOKORO_HAVE_HALO=0)
endif()

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_kokoro_test
        tests/error_test.cpp
        tests/engine_test.cpp
        tests/validation_test.cpp
    )
    target_link_libraries(onebit_kokoro_test PRIVATE onebit::kokoro doctest::doctest)
    add_test(NAME onebit_kokoro COMMAND onebit_kokoro_test)
endif()
