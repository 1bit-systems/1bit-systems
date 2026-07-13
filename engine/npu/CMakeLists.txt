cmake_minimum_required(VERSION 3.16)
project(npu-bf16-engine C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find XRT (AMD NPU runtime — xrt_coreutil, xrt_core)
find_library(XRT_COREUTIL xrt_coreutil)
find_library(XRT_CORE  xrt_core)

# Sources: model loader + BF16 engine variant
set(NPU_BF16_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../npu-infer/src/model.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/npu_engine_v2_bf16.cpp
)

add_executable(npu_engine_v2_bf16 ${NPU_BF16_SOURCES})

target_include_directories(npu_engine_v2_bf16 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include                    # helper.h, gemm_atb_layout.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../npu-infer/include     # common.h, model.h
)

target_link_libraries(npu_engine_v2_bf16
    ${XRT_COREUTIL}
    ${XRT_CORE}
    m dl pthread
)

target_compile_options(npu_engine_v2_bf16 PRIVATE
    -O3 -ffast-math -march=native -funroll-loops -fno-math-errno
    -Wno-unused-parameter -Wno-unused-variable
)

message(STATUS "npu-bf16-engine: XRT_COREUTIL=${XRT_COREUTIL}")
message(STATUS "npu-bf16-engine: XRT_CORE=${XRT_CORE}")

# Install target
install(TARGETS npu_engine_v2_bf16 DESTINATION bin)
