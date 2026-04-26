# cpp/onnx — ONNX Runtime lane (VitisAI EP for AMD NPU; CPU EP fallback).
#
# Soft-fails when libonnxruntime + onnxruntime_cxx_api.h aren't installed
# on the box: the artifact-discovery + genai_config parser still build
# (header-only, std + nlohmann/json), only the live `OnnxSession::load`
# path is gated behind the runtime presence.
#
# Mirrors the Rust crate's `load_config_only` / `load` split.

# Always-on library: artifact discovery + genai_config parsing + session
# entry points. session.cpp + generate.cpp internally cfg-gate on
# ONEBIT_ONNX_HAVE_ORT so the same translation units compile with or
# without ORT — they just return OrtRuntimeUnavailable in the no-ORT
# path. Pattern matches the Rust crate's `stub` / `real-kokoro`
# feature split.
add_library(onebit_onnx_lib STATIC
    src/error.cpp
    src/model.cpp
    src/session.cpp
    src/generate.cpp
)
target_include_directories(onebit_onnx_lib PUBLIC include)
target_compile_features(onebit_onnx_lib PUBLIC cxx_std_23)
target_link_libraries(onebit_onnx_lib PUBLIC nlohmann_json::nlohmann_json)
add_library(onebit::onnx ALIAS onebit_onnx_lib)

# --------------------------------------------------------------------------
# Optional ORT-backed session.cpp + generate.cpp targets.
#
# We deliberately do NOT use FetchContent for ORT — its build pulls in a
# huge tree (eigen, abseil, protobuf, …) and would dwarf the rest of the
# repo. We expect the operator to install ORT 1.18+ system-wide
# (`pacman -S onnxruntime` on CachyOS or AMD's onnxruntime-vitisai
# package on STX-H Linux when it ships).
# --------------------------------------------------------------------------
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATH_SUFFIXES onnxruntime onnxruntime/core/session
    HINTS
        /usr/include
        /usr/local/include
        /opt/onnxruntime/include
        $ENV{ORT_HOME}/include
        $ENV{ORT_HOME}/include/onnxruntime
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS
        /usr/lib
        /usr/local/lib
        /opt/onnxruntime/lib
        $ENV{ORT_HOME}/lib
)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    message(STATUS "onebit::onnx — ORT found: ${ONNXRUNTIME_LIBRARY}")
    target_include_directories(onebit_onnx_lib PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(onebit_onnx_lib PUBLIC ${ONNXRUNTIME_LIBRARY})
    target_compile_definitions(onebit_onnx_lib PUBLIC ONEBIT_ONNX_HAVE_ORT=1)
else()
    message(WARNING "onebit::onnx — onnxruntime_cxx_api.h / libonnxruntime not found; "
                    "session + generate paths return OrtRuntimeUnavailable. "
                    "Install onnxruntime to enable live decode.")
    target_compile_definitions(onebit_onnx_lib PUBLIC ONEBIT_ONNX_HAVE_ORT=0)
endif()

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_onnx_test
        tests/error_test.cpp
        tests/model_test.cpp
        tests/session_test.cpp
        tests/generate_test.cpp
    )
    target_link_libraries(onebit_onnx_test PRIVATE onebit::onnx doctest::doctest)
    add_test(NAME onebit_onnx COMMAND onebit_onnx_test)
endif()
