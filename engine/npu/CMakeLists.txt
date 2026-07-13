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

# Real gemm_atb_layout.h from mlir-aie (AMD AI Engine block data layout)
# /home/bcloud/mlir-aie/programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering
set(NPU_MLIR_AIE_DIR "$ENV{HOME}/mlir-aie" CACHE PATH "Path to mlir-aie install")
if(EXISTS "${NPU_MLIR_AIE_DIR}/programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering/gemm_atb_layout.h")
    target_include_directories(npu_engine_v2_bf16 PRIVATE
        "${NPU_MLIR_AIE_DIR}/programming_examples/ml/block_datatypes/gemm_asymmetric_tile_buffering"
    )
    message(STATUS "npu-bf16-engine: using real gemm_atb_layout.h from mlir-aie")
else()
    message(WARNING "npu-bf16-engine: mlir-aie not found at ${NPU_MLIR_AIE_DIR} — using stub gemm_atb_layout.h")
endif()

# torch2aie examples (helper.h, test_bf16_npu.cpp deps)
set(NPU_TORCH2AIE_DIR "$ENV{HOME}/torch2aie" CACHE PATH "Path to torch2aie install")
if(EXISTS "${NPU_TORCH2AIE_DIR}/examples/helper.h")
    target_include_directories(npu_engine_v2_bf16 PRIVATE
        "${NPU_TORCH2AIE_DIR}/examples"
    )
    message(STATUS "npu-bf16-engine: using torch2aie headers at ${NPU_TORCH2AIE_DIR}/examples")
endif()
