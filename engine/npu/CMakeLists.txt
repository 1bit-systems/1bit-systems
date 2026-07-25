cmake_minimum_required(VERSION 3.21)
project(npu-engine LANGUAGES C CXX)

# ── Toolchain ────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(XRT QUIET)

# ── C++ source files (ported from Zig) ───────────────────────────
set(NPU_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(NPU_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(NPU_CXX_SOURCES
    ${NPU_SRC_DIR}/xrt_wrapper.cpp
)

# ── npu_engine_universal (flagship — all ops: QKV, O, GU, D, attention) ──
set(NPU_ENGINE_SOURCES
    ${NPU_SRC_DIR}/npu_engine_universal.cpp
    ${NPU_SRC_DIR}/dequant_q4nx.cpp
    ${NPU_CXX_SOURCES}
)

add_executable(npu_engine_universal ${NPU_ENGINE_SOURCES})

target_include_directories(npu_engine_universal PRIVATE
    ${NPU_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../npu-infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include   # npu_instr.hpp, npu_app.hpp
)

if(XRT_FOUND)
    target_include_directories(npu_engine_universal PRIVATE ${XRT_INCLUDE_DIRS})
    target_link_libraries(npu_engine_universal ${XRT_CORE_LIBRARIES})
    target_compile_definitions(npu_engine_universal PRIVATE -DXRT_ENABLE)
endif()

option(ONEBP_SUPPORT "Enable native 1BP model format loading" ON)
if(ONEBP_SUPPORT)
    target_compile_definitions(npu_engine_universal PRIVATE ONEBP_SUPPORT)
endif()

target_link_libraries(npu_engine_universal
    m dl pthread uuid
    OpenMP::OpenMP_CXX
    aiebu
)

target_compile_options(npu_engine_universal PRIVATE
    -O3 -ffast-math -march=native -funroll-loops -fno-math-errno
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
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dequant_q4nx.cpp
    ${NPU_CXX_SOURCES}
)

add_executable(npu_engine_cb ${NPU_CB_SOURCES})

target_include_directories(npu_engine_cb PRIVATE
    ${NPU_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../npu-infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include   # npu_instr.hpp, npu_app.hpp
)

target_link_libraries(npu_engine_cb
    ${XRT_COREUTIL}
    m dl pthread uuid
)

target_compile_options(npu_engine_cb PRIVATE
    -O3 -ffast-math -march=native -funroll-loops -fno-math-errno
)

install(TARGETS npu_engine_cb DESTINATION bin)

# ── npu_engine_v13 — Cross-layer pipelined async decode ──
# Uses async launch/wait to overlap CPU ops with NPU execution.
set(NPU_V13_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/npu_engine_v13.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dequant_q4nx.cpp
)

add_executable(npu_engine_v13 ${NPU_V13_SOURCES})

target_include_directories(npu_engine_v13 PRIVATE
    ${NPU_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../npu-infer/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(npu_engine_v13
    ${XRT_CORE_LIBRARIES}
    m dl pthread uuid
    OpenMP::OpenMP_CXX
    aiebu
)

target_compile_options(npu_engine_v13 PRIVATE
    -O3 -ffast-math -march=native -funroll-loops -fno-math-errno -mavx512f
)

install(TARGETS npu_engine_v13 DESTINATION bin)

# ── npu_engine_fused — GPU+NPU fused zero-copy decode ──
# GPU does QKV+Attn+O, NPU does GU+D, SharedBO zero-copy overlap.
# ── GPU-accelerated targets — compiled with hipcc (handles __global__ + C++) ──
execute_process(COMMAND hipconfig --rocmlibdir OUTPUT_VARIABLE HIP_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_program(HIPCC hipcc)

if(HIPCC AND HIP_LIB_DIR)
    macro(add_npu_hip_target target_name source_name)
        set(NPU_HIP_SRC
            "${CMAKE_CURRENT_SOURCE_DIR}/src/${source_name}.cpp"
        )
        add_executable(${target_name} IMPORTED)
        
        # Use hipcc to compile AND link
        set(TARGET_BIN "${CMAKE_CURRENT_BINARY_DIR}/${target_name}")
        add_custom_target(${target_name}_build ALL
            COMMAND ${HIPCC}
                -D__HIP_PLATFORM_AMD__=1
                -I${NPU_INCLUDE_DIR}
                -I${CMAKE_CURRENT_SOURCE_DIR}/../../npu-infer/include
                -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
                -I/opt/rocm/include
                ${NPU_HIP_SRC}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/dequant_q4nx.cpp
                -O3 -ffast-math -march=native -funroll-loops -fno-math-errno -mavx512f
                `pkg-config --cflags --libs xrt-core 2>/dev/null || echo "-lxrt_coreutil -lxrt_core -luuid -lpthread"`
                -L${HIP_LIB_DIR} -lamdhip64 -lOpenMP -laiebu -lm -ldl
                -fopenmp
                -o ${TARGET_BIN}
            DEPENDS ${NPU_HIP_SRC}
            COMMENT "Building ${target_name} with hipcc..."
        )
        
        add_dependencies(${target_name} ${target_name}_build)
        
        install(TARGETS ${target_name} DESTINATION bin)
    endmacro()

    add_npu_hip_target(npu_engine_fused npu_engine_fused)
    add_npu_hip_target(npu_engine_overlap npu_engine_overlap)
    add_npu_hip_target(npu_engine_spec npu_engine_spec)
endif()
