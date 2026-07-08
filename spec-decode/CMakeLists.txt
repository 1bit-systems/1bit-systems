cmake_minimum_required(VERSION 3.20)
project(1bit-spec-decode VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find XRT (for NPU interaction)
find_package(XRT REQUIRED)
find_package(OpenMP REQUIRED)

# Options
option(BUILD_TESTING "Build tests" ON)
option(ENABLE_NPU "Enable NPU target" ON)
option(ENABLE_CURL "Enable HTTP daemon with curl" OFF)

# DSpark Draft library (header-only) — DEFAULT draft model
add_library(dspark_draft INTERFACE)
target_include_directories(dspark_draft INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# MTP Draft library (header-only — legacy Eagle3)
add_library(mtp_draft INTERFACE)
target_include_directories(mtp_draft INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Speculative Decode Engine (header-only, templated on draft model)
add_library(spec_decode INTERFACE)
target_link_libraries(spec_decode INTERFACE dspark_draft mtp_draft)
target_include_directories(spec_decode INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/engine)

# ── Single Binary: ALL capabilities merged ──
if(ENABLE_NPU)
    add_executable(npu_spec_decode
        engine/npu_single_binary.cpp
        /home/bcloud/engine/npu/src/dequant_q4nx.c
    )
    target_link_libraries(npu_spec_decode PRIVATE
        spec_decode
        XRT::xrt_coreutil
    )
    target_include_directories(npu_spec_decode PRIVATE
        ${XRT_INCLUDE_DIR}
        /home/bcloud/engine/npu/src
    )

    # Install single binary
    install(TARGETS npu_spec_decode
        RUNTIME DESTINATION bin
    )
endif()

# ── Legacy spec_proxy (optional, needs libcurl) ──
if(ENABLE_CURL AND ENABLE_NPU)
    add_executable(spec_proxy
        engine/spec_proxy.cpp
    )
    target_link_libraries(spec_proxy PRIVATE
        spec_decode
        curl
    )
    install(TARGETS spec_proxy RUNTIME DESTINATION bin)
endif()

# ── Benchmark tool (simulated target, no NPU needed) ──
add_executable(spec_decode_bench
    bench/spec_decode_bench.cpp
)
target_link_libraries(spec_decode_bench PRIVATE
    spec_decode
    OpenMP::OpenMP_CXX
)

# ── Test suite ──
if(BUILD_TESTING)
    enable_testing()

    add_executable(test_draft_model
        tests/test_draft_model.cpp
    )
    target_link_libraries(test_draft_model PRIVATE
        mtp_draft
    )
    add_test(NAME draft_model COMMAND test_draft_model)

    add_executable(test_verify_loop
        tests/test_decode_loop.cpp
    )
    target_link_libraries(test_verify_loop PRIVATE
        spec_decode
    )
    add_test(NAME verify_loop COMMAND test_verify_loop)

    # NPU acceptance test (needs XRT + NPU hardware)
    if(ENABLE_NPU)
        add_executable(test_acceptance
            tools/test_acceptance.cpp
            /home/bcloud/engine/npu/src/dequant_q4nx.c
        )
        target_link_libraries(test_acceptance PRIVATE
            spec_decode
            XRT::xrt_coreutil
        )
        target_include_directories(test_acceptance PRIVATE
            ${XRT_INCLUDE_DIR}
            /home/bcloud/engine/npu/src
        )
        add_test(NAME acceptance COMMAND test_acceptance)
    endif()
endif()

# ── Install headers ──
install(DIRECTORY draft/ engine/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
