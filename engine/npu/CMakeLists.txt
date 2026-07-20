cmake_minimum_required(VERSION 3.21)
project(npu-engine LANGUAGES C CXX)

# ── Toolchain ────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm/HIP used for TheRock amdclang++ runtime (tools only, not the engine)
find_package(hip REQUIRED)
find_package(XRT QUIET)

# ── npu_engine_universal (flagship — all ops: QKV, O, GU, D, attention) ──
set(NPU_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(NPU_ENGINE_SOURCES
    ${NPU_SRC_DIR}/npu_engine_universal.cpp
    ${NPU_SRC_DIR}/dequant_q4nx.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../npu-infer/src/flm_bridge.cpp
)

add_executable(npu_engine_universal ${NPU_ENGINE_SOURCES})

target_include_directories(npu_engine_universal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../npu-infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include   # npu_instr.hpp, npu_app.hpp
)

if(XRT_FOUND)
    target_include_directories(npu_engine_universal PRIVATE ${XRT_INCLUDE_DIRS})
    target_link_libraries(npu_engine_universal ${XRT_CORE_LIBRARIES})
    target_compile_definitions(npu_engine_universal PRIVATE -DXRT_ENABLE)
endif()

target_link_libraries(npu_engine_universal
    m dl pthread uuid
    OpenMP::OpenMP_CXX
    aiebu
)

target_compile_options(npu_engine_universal PRIVATE
    -O3 -ffast-math -march=native -funroll-loops -fno-math-errno
    -Wno-unused-parameter -Wno-unused-variable
)

install(TARGETS npu_engine_universal DESTINATION bin)

# ── Optional: torch2aie integration for bf16 kernel generation ──
set(NPU_TORCH2AIE_DIR "$ENV{HOME}/torch2aie" CACHE PATH "Path to torch2aie repo root")
if(EXISTS "${NPU_TORCH2AIE_DIR}/examples/helper.h")
    target_include_directories(npu_engine_universal PRIVATE
        ${NPU_TORCH2AIE_DIR}/iron
        ${NPU_TORCH2AIE_DIR}/examples
    )
    target_compile_definitions(npu_engine_universal PRIVATE NPU_HAVE_TORCH2AIE=1)
    message(STATUS "npu-bf16-engine: using torch2aie headers at ${NPU_TORCH2AIE_DIR}/examples")
endif()

# ── npu_engine_cb (v4) — dynamic instructions, xrt::ext::kernel ──
find_package(OpenMP REQUIRED)
find_library(AIEBU aiebu)
find_library(XRT_COREUTIL xrt_coreutil)

set(NPU_CB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/npu_engine_cb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dequant_q4nx.c
)

add_executable(npu_engine_cb ${NPU_CB_SOURCES})

target_include_directories(npu_engine_cb PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../npu-infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include   # npu_instr.hpp, npu_app.hpp
)

target_link_libraries(npu_engine_cb
    ${XRT_COREUTIL}
    m dl pthread uuid
)

target_compile_options(npu_engine_cb PRIVATE
    -O3 -ffast-math -march=native -funroll-loops -fno-math-errno
    -Wno-unused-parameter -Wno-unused-variable
)

install(TARGETS npu_engine_cb DESTINATION bin)
