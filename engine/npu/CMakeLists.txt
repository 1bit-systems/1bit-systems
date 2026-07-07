# CMakeLists.txt — NPU Engine Build (cross-platform: Linux + Windows)
#
# Linux:
#   cmake -B build -S . && cmake --build build
#   (requires XRT at /usr/include + -lxrt_coreutil -lxrt_core)
#
# Windows (MSVC):
#   cmake -B build -S . -DXRT_DIR="C:/Program Files/AMD/XRT" -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   (requires AMD NPU driver + Ryzen AI LT installed for xrt_coreutil.lib)
#
cmake_minimum_required(VERSION 3.20)
project(npu-engine
    VERSION 1.0.0
    DESCRIPTION "1bit.systems NPU Inference Engine — XDNA 2 INT8"
    LANGUAGES CXX C
)

# ---- C++ Standard ----
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# ---- Options ----
option(NPU_ENABLE_OPENMP "Enable OpenMP parallel attention/LM head" ON)
option(NPU_BUILD_ALL    "Build all 5 model variants" ON)

# ---- Detect platform ----
if(WIN32)
    set(PLATFORM_WIN32 TRUE)
    message(STATUS "Platform: Windows")
else()
    set(PLATFORM_LINUX TRUE)
    message(STATUS "Platform: Linux")
endif()

# ---- Source directories ----
set(SRC_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")

# ---- Find XRT ----
if(PLATFORM_LINUX)
    # Linux: XRT at /usr/include /usr/lib
    set(XRT_INCLUDE_DIR "/usr/include")
    set(XRT_LIB_DIR "/usr/lib")
    if(EXISTS "${XRT_INCLUDE_DIR}/xrt/xrt_device.h")
        message(STATUS "XRT found: ${XRT_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "XRT headers not found at ${XRT_INCLUDE_DIR}/xrt/")
    endif()
elseif(PLATFORM_WIN32)
    # Windows: XRT from AMD NPU driver / Ryzen AI LT
    # User sets XRT_DIR or we search common install paths
    if(NOT XRT_DIR)
        set(XRT_DIR "C:/Program Files/AMD/XRT" CACHE PATH "XRT installation root")
    endif()
    set(XRT_INCLUDE_DIR "${XRT_DIR}/include")
    set(XRT_LIB_DIR "${XRT_DIR}/lib")
    if(NOT EXISTS "${XRT_INCLUDE_DIR}/xrt/xrt_device.h")
        message(WARNING "XRT headers not found at ${XRT_INCLUDE_DIR}/xrt/ — set -DXRT_DIR=...")
    endif()
endif()

# ---- Find OpenMP ----
if(NPU_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        set(HAVE_OPENMP TRUE)
        message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
    else()
        message(STATUS "OpenMP not found — attention will use single-threaded fallback")
    endif()
endif()

# ---- Compile dequantizer (pure C, platform-independent) ----
add_library(dequant_q4nx STATIC
    "${SRC_DIR}/dequant_q4nx.c"
)
target_include_directories(dequant_q4nx PUBLIC "${SRC_DIR}")
target_compile_options(dequant_q4nx PRIVATE
    $<$<CONFIG:Release>:-O3>
)
set_target_properties(dequant_q4nx PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ---- Compile platform abstraction (if Windows) ----
if(PLATFORM_WIN32)
    # Win32 mmap shim lives here
    add_library(platform_shim STATIC
        "${SRC_DIR}/platform_mmap.cpp"
    )
    target_include_directories(platform_shim PUBLIC "${SRC_DIR}")
endif()

# ---- NPU engine function (shared across variants) ----
# The engine .cpp files all follow the same pattern:
#   - Include model_config.h for model dimensions
#   - Define MODEL_xxx to select dimensions
#   - Link: dequant_q4nx + XRT

function(add_npu_engine TARGET MODEL_DEF)
    set(ONE_SOURCE "${SRC_DIR}/npu_engine_universal.cpp")

    add_executable(${TARGET} "${ONE_SOURCE}")

    target_compile_definitions(${TARGET} PRIVATE ${MODEL_DEF})

    target_include_directories(${TARGET} PRIVATE
        "${SRC_DIR}"
        "${XRT_INCLUDE_DIR}"
    )

    target_link_libraries(${TARGET} PRIVATE
        dequant_q4nx
    )

    if(PLATFORM_WIN32)
        target_link_libraries(${TARGET} PRIVATE platform_shim)
    endif()

    # XRT libraries
    if(PLATFORM_LINUX)
        target_link_directories(${TARGET} PRIVATE "${XRT_LIB_DIR}")
        target_link_libraries(${TARGET} PRIVATE
            xrt_coreutil
            xrt_core
            uuid
            m
            dl
        )
    elseif(PLATFORM_WIN32)
        target_link_directories(${TARGET} PRIVATE "${XRT_LIB_DIR}")
        target_link_libraries(${TARGET} PRIVATE
            xrt_coreutil   # links xrt_coreutil.dll
            # xrt_core     # uncomment if needed
            # uuid         # Windows has uuid.lib via runtime
        )
    endif()

    # OpenMP
    if(HAVE_OPENMP)
        target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    # Platform-specific compile options
    if(PLATFORM_LINUX)
        target_compile_options(${TARGET} PRIVATE
            $<$<CONFIG:Release>:-O3 -march=native>
            -fopenmp
        )
    elseif(PLATFORM_WIN32)
        target_compile_options(${TARGET} PRIVATE
            $<$<CONFIG:Release>:/O2 /arch:AVX2>
            $<$<BOOL:${HAVE_OPENMP}>:/openmp>
        )
    endif()

    # Install
    install(TARGETS ${TARGET} RUNTIME DESTINATION bin)
endfunction()

# ---- Build targets ----
if(NPU_BUILD_ALL OR NPU_BUILD_QWEN3_0_6B)
    add_npu_engine(npu_engine_qwen3_0_6b MODEL_qwen3_0_6b)
endif()

if(NPU_BUILD_ALL OR NPU_BUILD_QWEN3_8B)
    add_npu_engine(npu_engine_qwen3_8b MODEL_qwen3_8b)
endif()

if(NPU_BUILD_ALL OR NPU_BUILD_QWEN3_VL_4B)
    add_npu_engine(npu_engine_qwen3_vl_4b MODEL_qwen3_vl_4b)
endif()

if(NPU_BUILD_ALL OR NPU_BUILD_LLAMA)
    add_npu_engine(npu_engine_llama MODEL_llama)
endif()

if(NPU_BUILD_ALL OR NPU_BUILD_GEMMA4_E2B)
    add_npu_engine(npu_engine_gemma4_e2b MODEL_gemma4_e2b)
endif()

# Default build (qwen3_0_6b as "npu_engine")
if(NOT TARGET npu_engine)
    add_npu_engine(npu_engine MODEL_qwen3_0_6b)
endif()

# ---- Fused engine (self-contained, one xclbin call per layer) ----
if(NPU_BUILD_ALL OR NPU_BUILD_FUSED)
    add_executable(npu_engine_fused "${SRC_DIR}/npu_engine_fused.cpp")
    target_include_directories(npu_engine_fused PRIVATE
        "${SRC_DIR}"
        "${XRT_INCLUDE_DIR}"
    )
    if(PLATFORM_LINUX)
        target_link_directories(npu_engine_fused PRIVATE "${XRT_LIB_DIR}")
        target_link_libraries(npu_engine_fused PRIVATE
            xrt_coreutil
            xrt_core
            uuid
            m
            dl
        )
    elseif(PLATFORM_WIN32)
        target_link_directories(npu_engine_fused PRIVATE "${XRT_LIB_DIR}")
        target_link_libraries(npu_engine_fused PRIVATE
            xrt_coreutil
        )
    endif()
    if(PLATFORM_LINUX)
        target_compile_options(npu_engine_fused PRIVATE
            $<$<CONFIG:Release>:-O3 -march=native>
        )
    elseif(PLATFORM_WIN32)
        target_compile_options(npu_engine_fused PRIVATE
            $<$<CONFIG:Release>:/O2 /arch:AVX2>
        )
    endif()
    install(TARGETS npu_engine_fused RUNTIME DESTINATION bin)
endif()

message(STATUS "NPU engine targets:")
foreach(t IN ITEMS npu_engine_qwen3_0_6b npu_engine_qwen3_8b npu_engine_qwen3_vl_4b npu_engine_llama npu_engine_gemma4_e2b npu_engine_fused)
    if(TARGET ${t})
        message(STATUS "  + ${t}")
    endif()
endforeach()
