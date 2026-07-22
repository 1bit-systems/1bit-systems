cmake_minimum_required(VERSION 3.21)

# Pin HIP compiler to TheRock's clang for native gfx1151. Discover TheRock
# from multiple locations in priority order:
#   1. /opt/rocm-therock — TheRock nightly C++ SDK (this repo)
#   2. THEROCK_PIP_ROOT env var — pip-installed TheRock
#   3. lemonade cache (~/.cache/lemonade/bin/therock)
#   4. legacy therock path (~/therock/build/dist/rocm)
#   5. system ROCm (/opt/rocm)
set(_RC_THEROCK_OPT     "/opt/rocm-therock" CACHE PATH "TheRock opt install root")
set(_RC_THEROCK_PIP     "$ENV{HOME}/.cache/pip/therock" CACHE PATH "pip-installed TheRock root")
set(_RC_LEMONADE_THEROCK "$ENV{HOME}/.cache/lemonade/bin/therock")
set(_RC_THEROCK_LEGACY "$ENV{HOME}/therock/build/dist/rocm")
set(_RC_SYSTEM_BIN     "/opt/rocm/lib/llvm/bin")

set(_RC_LLVM_BIN "")
set(_RC_ROCM_DEFAULT "")

# 1. /opt/rocm-therock — TheRock nightly C++ SDK (system-wide install)
if(EXISTS "${_RC_THEROCK_OPT}/bin/amdclang++")
    set(_RC_LLVM_BIN "${_RC_THEROCK_OPT}/bin")
    set(_RC_ROCM_DEFAULT "${_RC_THEROCK_OPT}")
    message(STATUS "rocm-cpp: using TheRock opt at ${_RC_ROCM_DEFAULT}")
endif()

# 2. Try pip-installed TheRock (from virtual env) — check for rocm-sdk path
if(NOT _RC_LLVM_BIN AND DEFINED ENV{THEROCK_PIP_ROOT})
    set(_RC_PIP_PATH "$ENV{THEROCK_PIP_ROOT}")
    if(EXISTS "${_RC_PIP_PATH}/bin/amdclang++")
        set(_RC_LLVM_BIN "${_RC_PIP_PATH}/bin")
        set(_RC_ROCM_DEFAULT "${_RC_PIP_PATH}")
        message(STATUS "rocm-cpp: using pip TheRock at ${_RC_ROCM_DEFAULT}")
    endif()
endif()

# 3. Lemonade cache
if(NOT _RC_LLVM_BIN)
    file(GLOB _RC_THEROCK_DIRS LIST_DIRECTORIES true "${_RC_LEMONADE_THEROCK}/gfx*")
    list(SORT _RC_THEROCK_DIRS)
    list(REVERSE _RC_THEROCK_DIRS)
    list(LENGTH _RC_THEROCK_DIRS _RC_THEROCK_N)
    if(_RC_THEROCK_N GREATER 0)
        list(GET _RC_THEROCK_DIRS 0 _RC_THEROCK_DIR)
        if(EXISTS "${_RC_THEROCK_DIR}/bin/clang++")
            set(_RC_LLVM_BIN "${_RC_THEROCK_DIR}/bin")
            set(_RC_ROCM_DEFAULT "${_RC_THEROCK_DIR}")
            message(STATUS "rocm-cpp: using TheRock at ${_RC_ROCM_DEFAULT}")
        endif()
    endif()
endif()

# 4. Legacy therock path
if(NOT _RC_LLVM_BIN)
    if(EXISTS "${_RC_THEROCK_LEGACY}/lib/llvm/bin/clang++")
        set(_RC_LLVM_BIN "${_RC_THEROCK_LEGACY}/lib/llvm/bin")
        set(_RC_ROCM_DEFAULT "${_RC_THEROCK_LEGACY}")
        message(STATUS "rocm-cpp: using legacy TheRock at ${_RC_ROCM_DEFAULT}")
    endif()
endif()

# 5. System ROCm
if(NOT _RC_LLVM_BIN)
    if(EXISTS "${_RC_SYSTEM_BIN}/amdclang++")
        set(_RC_LLVM_BIN "${_RC_SYSTEM_BIN}")
        set(_RC_ROCM_DEFAULT "/opt/rocm")
        message(STATUS "rocm-cpp: using system ROCm at ${_RC_ROCM_DEFAULT}")
    else()
        message(FATAL_ERROR
            "rocm-cpp needs an AMD clang: set THEROCK_PIP_ROOT env var, install "
            "via 'pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ rocm[devel,libraries]', "
            "or install system ROCm at /opt/rocm.")
    endif()
endif()

set(CMAKE_HIP_COMPILER "${_RC_LLVM_BIN}/amdclang++" CACHE FILEPATH "HIP compiler")
set(CMAKE_C_COMPILER   "${_RC_LLVM_BIN}/amdclang"   CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "${_RC_LLVM_BIN}/amdclang++" CACHE FILEPATH "C++ compiler")

project(1bit-systems LANGUAGES C CXX HIP)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

# NDEBUG only for Release — let Debug builds keep assert() guards
add_compile_definitions($<$<CONFIG:Release>:NDEBUG>)

set(ROCM_ROOT "${_RC_ROCM_DEFAULT}" CACHE PATH "ROCm install root")
list(PREPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}" "${ROCM_ROOT}/lib/cmake")

find_package(hip REQUIRED CONFIG PATHS "${ROCM_ROOT}")

set(ROCWMMA_FOUND FALSE)
foreach(_rc_base "${_RC_PIP_PATH}" "${_RC_THEROCK_DIR}" "${_RC_THEROCK_LEGACY}" "/opt/rocm-therock" "/opt/rocm")
    if(_rc_base AND EXISTS "${_rc_base}/include/rocwmma/rocwmma.hpp")
        set(ROCWMMA_INCLUDE_DIR "${_rc_base}/include")
        set(ROCWMMA_FOUND TRUE)
        message(STATUS "rocm-cpp: rocWMMA found at ${ROCWMMA_INCLUDE_DIR}")
        break()
    endif()
    if(_rc_base AND EXISTS "${_rc_base}/include/rocwmma/rocwmma.hpp")
        set(ROCWMMA_INCLUDE_DIR "${_rc_base}/include")
        set(ROCWMMA_FOUND TRUE)
        message(STATUS "rocm-cpp: rocWMMA found at ${ROCWMMA_INCLUDE_DIR} (TheRock SDK)")
        break()
    endif()
endforeach()

if(NOT ROCWMMA_FOUND)
    file(GLOB_RECURSE _RC_ROCWMMA_HPP "$ENV{HOME}/.local/lib/python*/site-packages/_rocm_sdk_devel/include/rocwmma/rocwmma.hpp")
    list(LENGTH _RC_ROCWMMA_HPP _RC_ROCWMMA_N)
    if(_RC_ROCWMMA_N GREATER 0)
        list(GET _RC_ROCWMMA_HPP 0 _RC_ROCWMMA_PATH)
        get_filename_component(ROCWMMA_INCLUDE_DIR "${_RC_ROCWMMA_PATH}" DIRECTORY)
        get_filename_component(ROCWMMA_INCLUDE_DIR "${ROCWMMA_INCLUDE_DIR}" DIRECTORY)
        set(ROCWMMA_FOUND TRUE)
        message(STATUS "rocm-cpp: rocWMMA found at ${ROCWMMA_INCLUDE_DIR} (pip fallback)")
    endif()
endif()
if(NOT ROCWMMA_FOUND)
    message(STATUS "rocm-cpp: rocWMMA NOT found — WMMA intrinsics disabled")
endif()


# -- Discover TheRock SDK under Python site-packages (any Python version) --
# TheRock's ROCm SDK may install under a Python-versioned path like
#   lib/python3.X/site-packages/_rocm_sdk_devel/
# Instead of hardcoding a specific Python version (issue #355), search
# under all Python site-packages for the _rocm_sdk_devel directory.
function(_find_therock_sdk_base out_var)
    if(DEFINED CACHE{THEROCK_SDK_BASE})
        set(${out_var} "${THEROCK_SDK_BASE}" PARENT_SCOPE)
        return()
    endif()
    # Search common locations, preferring newest Python version
    foreach(_base IN ITEMS "$ENV{HOME}/.local/lib" "/usr/local/lib" "/usr/lib")
        file(GLOB _py_dirs LIST_DIRECTORIES true "${_base}/python*")
        list(SORT _py_dirs)
        list(REVERSE _py_dirs)
        foreach(_py_dir IN LISTS _py_dirs)
            set(_candidate "${_py_dir}/site-packages/_rocm_sdk_devel")
            if(EXISTS "${_candidate}/include/rocwmma/rocwmma.hpp" OR EXISTS "${_candidate}")
                set(${out_var} "${_candidate}" PARENT_SCOPE)
                set(THEROCK_SDK_BASE "${_candidate}" CACHE INTERNAL "Found TheRock SDK base")
                message(STATUS "rocm-cpp: TheRock SDK found at ${_candidate}")
                return()
            endif()
        endforeach()
    endforeach()
    # Fallback: system ROCm
    if(EXISTS "/opt/rocm/lib/llvm/bin/amdclang++")
        set(${out_var} "/opt/rocm" PARENT_SCOPE)
        return()
    endif()
    set(${out_var} "" PARENT_SCOPE)
endfunction()

_find_therock_sdk_base(THEROCK_SDK_BASE)

include(FetchContent)
FetchContent_Declare(cpp_httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.18.1)
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3)
FetchContent_Declare(ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
    GIT_TAG        v5.0.0)
set(FTXUI_BUILD_DOCS     OFF CACHE INTERNAL "")
set(FTXUI_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(FTXUI_BUILD_TESTS    OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(cpp_httplib nlohmann_json ftxui)

set(RCPP_HAVE_CK FALSE)
find_package(composable_kernel QUIET CONFIG PATHS "${ROCM_ROOT}")
if(composable_kernel_FOUND)
    set(RCPP_HAVE_CK TRUE)
    message(STATUS "rocm-cpp: composable_kernel FOUND — building ck_gemm.cpp")
else()
    message(STATUS "rocm-cpp: composable_kernel NOT found — skipping ck_gemm.cpp, using standalone prefill")
endif()

option(USE_VULKAN "Build the portable Vulkan backend proof (test_vulkan_gemv)" OFF)
set(RCPP_HAVE_VULKAN FALSE)
if(USE_VULKAN)
    find_package(Vulkan QUIET)
    find_program(GLSLC_EXECUTABLE glslc)
    if(Vulkan_FOUND AND GLSLC_EXECUTABLE)
        set(RCPP_HAVE_VULKAN TRUE)
        message(STATUS "rocm-cpp: Vulkan FOUND (${Vulkan_LIBRARY}), glslc at ${GLSLC_EXECUTABLE} — building test_vulkan_gemv")
    else()
        message(WARNING "rocm-cpp: USE_VULKAN=ON but Vulkan SDK or glslc not found — skipping test_vulkan_gemv")
    endif()
endif()

# Auto-detect GPU architecture via rocminfo, fall back to gfx1151
if(NOT CMAKE_HIP_ARCHITECTURES)
    execute_process(
        COMMAND bash -c "rocminfo 2>/dev/null | grep -oP 'gfx[0-9a-z]+' | head -1"
        OUTPUT_VARIABLE _DETECTED_ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_DETECTED_ARCH)
        set(CMAKE_HIP_ARCHITECTURES "${_DETECTED_ARCH}")
    else()
        set(CMAKE_HIP_ARCHITECTURES "gfx1151")
    endif()
endif()
set(CMAKE_HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}" CACHE STRING "GPU arch(es) to build")

# -- librocm_cpp : shared library exposing the C API ----------------------------
set(RCPP_SOURCES
    src/ternary_gemv_launchers.hip
    src/prefill_standalone.hip
    src/wmma_peak_probe.hip
    src/prim_kernels.hip
    src/kv_cache_attn.hip
    src/kv_cache_attn_i8.hip
    src/kv_cache_attn_fd.hip
    src/kv_cache_attn_fd_rotor.hip
    src/kv_rotorquant.cpp
    src/h1b_loader.cpp
    src/gguf_loader.cpp
    src/onnx_loader.cpp
    src/tokenizer.cpp
    src/ternary_gemm_smallm.hip
    src/ternary_gemm_smallm_scalar_ref.hip
    src/sherry_gemv.hip
    src/sherry_gemv_scalar_ref.hip
    src/bonsai_q1_gemv.hip
    kernels/bonsai_q1_gemv_soa.hip
    src/bonsai_tq2_gemv.hip
    src/bonsai_gemv_scalar_ref.hip
    src/medusa_tree_attn.hip
    src/medusa_small_m_gemv.hip
    kernels/ternary_gemv_phase5_dot4.hip
    kernels/ternary_gemv_phase5_halo.hip
    kernels/ternary_gemv_sherry.hip
    kernels/ternary_gemv_tq1_halo.hip
    kernels/ternary_gemv_wmma.hip
    kernels/ternary_gemv_block_scaled.hip
    kernels/rotorquant_pack.hip
    kernels/hadamard_rotate_butterfly.hip
    kernels/zaya_moe_ternary_gemv.hip
    kernels/zaya_moe_tiled_gemv.hip
    kernels/zaya_moe_wmma_batched.hip
    kernels/wmma_i8_gemv.hip
    src/zaya_moe_launcher.hip
    src/mamba1_engine.hip
    src/bonsai_q1_1024.hip
    src/bonsai_tq2_1024.hip
    src/fused_gemv_tq2_1024.hip
    src/bonsai_tq2_1024_opt.hip
    kernels/vl_resize_norm.hip
    src/laguna_gemv.hip
    src/laguna_fused.hip
    src/laguna_ops.hip)
if(RCPP_HAVE_CK)
    list(APPEND RCPP_SOURCES src/ck_gemm.cpp)
endif()

# NPU backend (worker subprocess) + Q4NX reader — compiled into shared lib
list(APPEND RCPP_SOURCES src/backend_npu.cpp)
list(APPEND RCPP_SOURCES src/q4nx_reader.cpp)

# -- libgguf_reader : single shared GGUF parser (no HIP dependency) -----------
# Split into its own tiny static lib (rather than compiled directly into both
# backend_manager and rocm_cpp) so targets that link both don't hit duplicate-
# symbol errors.
add_library(gguf_reader STATIC src/gguf_reader.cpp)
target_include_directories(gguf_reader PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# -- libvl_image : lightweight image loading + preprocessing (no HIP dep) ----
# Uses stb_image.h (vendored at third_party/stb/) for decode. No OpenCV dep.
# Provides: vl_load_image, vl_resize_normalize, vl_decode_base64_image, etc.
add_library(vl_image STATIC src/vl_processor.cpp)
target_include_directories(vl_image PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_options(vl_image PRIVATE -O3 -std=c++23 -Wno-unused-result)

add_library(rocm_cpp SHARED ${RCPP_SOURCES})
if(ROCWMMA_FOUND)
    target_include_directories(rocm_cpp SYSTEM PRIVATE "${ROCWMMA_INCLUDE_DIR}")
endif()
target_sources(rocm_cpp PRIVATE src/prefill_dispatcher.cpp)

# -- zinc_cpp : C++ ZINC GPU backend (Vulkan compute inference)
#    Replaces the Zig-based ZINC project (engine/gpu/) which has been
#    fully reverse-engineered and ported to C++. Builds from the
#    engine/gpu/zinc_cpp/ subdirectory.
add_subdirectory(engine/gpu/zinc_cpp zinc_cpp_build)
set(ZINC_CPP_AVAILABLE TRUE)

# -- libbackend_manager : Backend abstraction layer (requires rocm_cpp for HIP) ---------
add_library(backend_manager STATIC
    src/backend_factory.cpp
    src/backend_manager.cpp
    src/backend_monitor.cpp
    src/backend_plugin.cpp
    src/backend_cpu.cpp
    src/backend_generic.cpp
    src/backend_mamba1.cpp
    src/backend_laguna.cpp
    src/simple_tokenizer.cpp
    src/q4nx_reader.cpp
    src/safetensors_reader.cpp
    src/model_discovery.cpp)
target_link_libraries(backend_manager PUBLIC gguf_reader)

target_include_directories(backend_manager PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_include_directories(backend_manager PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_include_directories(backend_manager PRIVATE "${ROCM_ROOT}/include")
target_compile_definitions(backend_manager PRIVATE __HIP_PLATFORM_AMD__=1)
target_link_libraries(backend_manager PUBLIC dl rocm_cpp hip::host hip::device)
target_compile_options(backend_manager PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)

if(ZINC_ENABLED)
    target_sources(backend_manager PRIVATE src/backend_zinc.cpp)
    # add_dependencies removed - zinc_cpp is cmake target
    target_link_libraries(backend_manager PUBLIC zinc_cpp)
else()
    target_compile_definitions(backend_manager PRIVATE ZINC_DISABLED=1)
endif()

set(RCPP_HIP_SOURCES ${RCPP_SOURCES} src/prefill_dispatcher.cpp)
list(FILTER RCPP_HIP_SOURCES INCLUDE REGEX "\\.hip$|prefill_dispatcher\\.cpp$")
if(RCPP_HAVE_CK)
    list(APPEND RCPP_HIP_SOURCES src/ck_gemm.cpp)
endif()
set_source_files_properties(${RCPP_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
target_include_directories(rocm_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_include_directories(rocm_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
target_include_directories(rocm_cpp SYSTEM PRIVATE "${ROCM_ROOT}/include")
target_compile_definitions(rocm_cpp PRIVATE MAMBA1_ENGINE_AS_LIB=1)
target_compile_options(rocm_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffast-math -munsafe-fp-atomics
        -Wno-unused-result>)
target_link_libraries(rocm_cpp PRIVATE hip::host hip::device gguf_reader)
set_target_properties(rocm_cpp PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

if(RCPP_HAVE_CK)
    add_executable(test_ck_gemm tests/test_ck_gemm.cpp)
    set_source_files_properties(tests/test_ck_gemm.cpp PROPERTIES LANGUAGE HIP)
    target_compile_options(test_ck_gemm PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
    target_link_libraries(test_ck_gemm PRIVATE rocm_cpp hip::host hip::device)
    set_target_properties(test_ck_gemm PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endif()

if(RCPP_HAVE_CK)
    add_executable(test_standalone tests/test_standalone.cpp)
    set_source_files_properties(tests/test_standalone.cpp PROPERTIES LANGUAGE HIP)
    target_compile_options(test_standalone PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
    target_link_libraries(test_standalone PRIVATE rocm_cpp hip::host hip::device)
    set_target_properties(test_standalone PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endif()


function(rocm_hip_target TARGET)
    add_executable(${TARGET} ${ARGN})
    set_source_files_properties(${ARGN} PROPERTIES LANGUAGE HIP)
    target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
    target_link_libraries(${TARGET} PRIVATE rocm_cpp hip::host hip::device)
    set_target_properties(${TARGET} PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endfunction()

rocm_hip_target(test_standalone_nock tests/test_standalone_nock.cpp)
rocm_hip_target(test_prim_and_attn tests/test_prim_and_attn.cpp)
rocm_hip_target(test_ternary_gemm_smallm tests/test_ternary_gemm_smallm.cpp)
rocm_hip_target(test_sherry_gemv tests/test_sherry_gemv.cpp)
rocm_hip_target(test_bonsai_gemv tests/test_bonsai_gemv.cpp)
rocm_hip_target(bench_bonsai_tile8 tests/bench_bonsai_tile8.cpp)
rocm_hip_target(bench_bonsai_tile8_model tests/bench_bonsai_tile8_model.cpp)
rocm_hip_target(bench_bonsai_q1_1024 tests/bench_bonsai_q1_1024.cpp)
rocm_hip_target(bench_bonsai_tq2_1024 tests/bench_bonsai_tq2_1024.cpp)
rocm_hip_target(bench_fused_tq2_1024 tests/bench_fused_tq2_1024.cpp)

add_executable(bench_cb_compare tests/bench_cb_compare.cpp)
set_source_files_properties(tests/bench_cb_compare.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_cb_compare PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_cb_compare PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_cb_compare PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

if(RCPP_HAVE_VULKAN)
    set(VK_SHADER_DIR "${CMAKE_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${VK_SHADER_DIR}")
    set(VK_SHADER_SOURCES kernels/vulkan/dmmv_tq2_bonsai.comp kernels/vulkan/dmmv_q1_bonsai.comp)
    set(VK_SHADER_SPVS "")
    foreach(_vk_src ${VK_SHADER_SOURCES})
        get_filename_component(_vk_name "${_vk_src}" NAME_WE)
        set(_vk_spv "${VK_SHADER_DIR}/${_vk_name}.spv")
        add_custom_command(OUTPUT "${_vk_spv}"
            COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.2 -O -o "${_vk_spv}" "${CMAKE_SOURCE_DIR}/${_vk_src}"
            DEPENDS "${CMAKE_SOURCE_DIR}/${_vk_src}"
            COMMENT "glslc ${_vk_src} -> ${_vk_name}.spv")
        list(APPEND VK_SHADER_SPVS "${_vk_spv}")
    endforeach()
    add_custom_target(vulkan_shaders DEPENDS ${VK_SHADER_SPVS})
    add_executable(test_vulkan_gemv tests/test_vulkan_gemv.cpp)
    target_link_libraries(test_vulkan_gemv PRIVATE Vulkan::Vulkan)
    target_compile_definitions(test_vulkan_gemv PRIVATE VK_SHADER_DIR="${VK_SHADER_DIR}")
    add_dependencies(test_vulkan_gemv vulkan_shaders)
endif()

rocm_hip_target(test_sherry_e2e tests/test_sherry_e2e.cpp)
rocm_hip_target(test_zaya_moe_gemv tests/test_zaya_moe_gemv.cpp)
rocm_hip_target(test_block_scaled_ternary tests/test_block_scaled_ternary.cpp)

add_executable(test_cca_attn tests/test_cca_attn.cpp kernels/zaya_cca_attn.hip)
set_source_files_properties(tests/test_cca_attn.cpp kernels/zaya_cca_attn.hip PROPERTIES LANGUAGE HIP)
target_compile_options(test_cca_attn PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_cca_attn PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_cca_attn PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

rocm_hip_target(zaya_full tests/zaya_full.cpp)

# -- zaya_gpu_decode : GPU decode benchmarking tool (not a test) -------------
add_executable(zaya_gpu_decode tests/zaya_gpu_decode.cpp kernels/zaya_cca_attn.hip)
set_source_files_properties(tests/zaya_gpu_decode.cpp kernels/zaya_cca_attn.hip PROPERTIES LANGUAGE HIP)
target_compile_options(zaya_gpu_decode PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_include_directories(zaya_gpu_decode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(zaya_gpu_decode PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(zaya_gpu_decode PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(test_lora_merge tests/test_lora_merge.cpp src/zaya_engine.cpp)
set_source_files_properties(tests/test_lora_merge.cpp src/zaya_engine.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(test_lora_merge PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/kernels)
target_compile_options(test_lora_merge PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_lora_merge PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_lora_merge PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- zaya_server : multi-model multi-backend inference server (main binary,
#    NOT a test — not registered with add_test(), see install() below) ------
set(ZAYA_SERVER_SRC tests/zaya_server.cpp tests/backends/backend_cpu.cpp tests/backends/backend_hip.cpp tests/backends/backend_npu.cpp tests/backends/backend_generic.cpp)
if(RCPP_HAVE_VULKAN)
    list(APPEND ZAYA_SERVER_SRC tests/backends/backend_vulkan.cpp)
endif()
if(ZINC_ENABLED)
    list(APPEND ZAYA_SERVER_SRC tests/backends/backend_zinc.cpp)
endif()
add_executable(zaya_server ${ZAYA_SERVER_SRC})
set_source_files_properties(tests/zaya_server.cpp tests/backends/backend_hip.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(zaya_server PRIVATE ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/backends)
target_compile_options(zaya_server PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(zaya_server PRIVATE rocm_cpp hip::host hip::device httplib::httplib nlohmann_json::nlohmann_json)
set_target_properties(zaya_server PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
install(TARGETS zaya_server RUNTIME DESTINATION bin)
if(RCPP_HAVE_VULKAN)
    target_link_libraries(zaya_server PRIVATE Vulkan::Vulkan)
endif()

rocm_hip_target(test_wmma_i8_gemv tests/test_wmma_i8_gemv.cpp)
rocm_hip_target(test_bonsai_e2e tests/test_bonsai_e2e.cpp)

add_executable(test_medusa_skeleton tests/test_medusa_skeleton.cpp)
target_compile_options(test_medusa_skeleton PRIVATE -Wall -Wextra -Wno-unused-result)
target_link_libraries(test_medusa_skeleton PRIVATE rocm_cpp)

# -- test_backend: CPU-only backend validation (issue #347) --
# Previously only had a broken standalone build command that couldn't link
# due to transitive HIP deps through backend_factory → backend_generic → gguf_loader.
# Now wired as a proper ninja target linking backend_manager (which already handles
# the rocm_cpp dependency). Run without a model to see usage.
add_executable(test_backend tests/test_backend.cpp)
target_include_directories(test_backend PRIVATE src)
target_compile_options(test_backend PRIVATE -O3 -Wall -Wno-unused-result -std=c++23 -DZINC_DISABLED)
target_link_libraries(test_backend PRIVATE backend_manager)

set(BONSAI_H1B_MODEL "" CACHE FILEPATH "Path to a Bonsai .h1b model for test_bonsai_e2e (empty = skip)")
set(SHERRY_H1B_MODEL "" CACHE FILEPATH "Path to a Sherry .h1b model for test_sherry_e2e (empty = skip)")

add_test(NAME test_medusa_skeleton         COMMAND test_medusa_skeleton)
set_tests_properties(test_medusa_skeleton   PROPERTIES LABELS "host")

# -- test_gguf_reader_dequant: no GPU, no downloaded model, no Python at ------
# test time. Byte-exact dequant check for every dtype gguf_reader.cpp
# supports, against reference values baked in from the Python `gguf`
# package. Runs unconditionally in every CI invocation.
add_executable(test_gguf_reader_dequant tests/test_gguf_reader_dequant.cpp)
target_include_directories(test_gguf_reader_dequant PRIVATE tests/)
target_link_libraries(test_gguf_reader_dequant PRIVATE gguf_reader)
add_test(NAME test_gguf_reader_dequant COMMAND test_gguf_reader_dequant)
set_tests_properties(test_gguf_reader_dequant PROPERTIES LABELS "host")

add_test(NAME test_backend                 COMMAND test_backend)
set_tests_properties(test_backend           PROPERTIES LABELS "host" SKIP_RETURN_CODE 77)

# test_backend needs weights at /tmp/zaya_weights or the passed path
# Skips gracefully (exit 77) when weights aren't found.

add_test(NAME test_standalone_nock         COMMAND test_standalone_nock)
set_tests_properties(test_standalone_nock   PROPERTIES LABELS "gpu_required")
add_test(NAME test_prim_and_attn           COMMAND test_prim_and_attn)
set_tests_properties(test_prim_and_attn     PROPERTIES LABELS "gpu_required")
add_test(NAME test_ternary_gemm_smallm     COMMAND test_ternary_gemm_smallm)
set_tests_properties(test_ternary_gemm_smallm PROPERTIES LABELS "gpu_required")
add_test(NAME test_sherry_gemv             COMMAND test_sherry_gemv)
set_tests_properties(test_sherry_gemv       PROPERTIES LABELS "gpu_required")
add_test(NAME test_bonsai_gemv             COMMAND test_bonsai_gemv)
set_tests_properties(test_bonsai_gemv       PROPERTIES LABELS "gpu_required")
add_test(NAME test_zaya_moe_gemv           COMMAND test_zaya_moe_gemv)
set_tests_properties(test_zaya_moe_gemv     PROPERTIES LABELS "gpu_required")
add_test(NAME test_block_scaled_ternary    COMMAND test_block_scaled_ternary)
set_tests_properties(test_block_scaled_ternary PROPERTIES LABELS "gpu_required")
add_test(NAME test_cca_attn                COMMAND test_cca_attn)
set_tests_properties(test_cca_attn          PROPERTIES LABELS "gpu_required")
add_test(NAME test_wmma_i8_gemv            COMMAND test_wmma_i8_gemv)
set_tests_properties(test_wmma_i8_gemv     PROPERTIES LABELS "gpu_required")

add_test(NAME test_bonsai_e2e              COMMAND test_bonsai_e2e ${BONSAI_H1B_MODEL})
set_tests_properties(test_bonsai_e2e       PROPERTIES LABELS "gpu_required;model_required;e2e" SKIP_RETURN_CODE 77)
add_test(NAME test_sherry_e2e              COMMAND test_sherry_e2e ${SHERRY_H1B_MODEL})
set_tests_properties(test_sherry_e2e       PROPERTIES LABELS "gpu_required;model_required;e2e" SKIP_RETURN_CODE 77)

# -- test_hip_generic_attention: regression guard for the HIP backend's ------
# generic (non-Zaya) attention path (fixed in #474/#486 — see the source
# file's header comment). Compiles tests/backends/backend_hip.cpp directly,
# same as zaya_server/unified_server do, since it isn't part of any shared
# library target.
set(GGUF_TEST_MODEL "" CACHE FILEPATH "Path to a non-Zaya GGUF model for test_hip_generic_attention (empty = skip)")
add_executable(test_hip_generic_attention
    tests/test_hip_generic_attention.cpp
    tests/backends/backend_hip.cpp)
set_source_files_properties(tests/backends/backend_hip.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(test_hip_generic_attention PRIVATE ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/backends)
target_compile_options(test_hip_generic_attention PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_hip_generic_attention PRIVATE backend_manager rocm_cpp hip::host hip::device)
set_target_properties(test_hip_generic_attention PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
add_test(NAME test_hip_generic_attention COMMAND test_hip_generic_attention ${GGUF_TEST_MODEL})
set_tests_properties(test_hip_generic_attention PROPERTIES LABELS "gpu_required;model_required;e2e" SKIP_RETURN_CODE 77)

add_executable(bitnet_decode tools/bitnet_decode.cpp)
set_source_files_properties(tools/bitnet_decode.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bitnet_decode PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bitnet_decode PRIVATE rocm_cpp hip::host hip::device httplib::httplib nlohmann_json::nlohmann_json)
set_target_properties(bitnet_decode PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(hadamard_export tools/hadamard_export.cpp)
target_compile_options(hadamard_export PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
target_link_libraries(hadamard_export PRIVATE rocm_cpp)

add_executable(bitnet_debate tools/bitnet_debate.cpp)
set_source_files_properties(tools/bitnet_debate.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bitnet_debate PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bitnet_debate PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bitnet_debate PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(bench_sherry tools/bench_sherry.cpp)
set_source_files_properties(tools/bench_sherry.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_sherry PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_sherry PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_sherry PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(bench_kv_i8 tools/bench_kv_i8.cpp)
set_source_files_properties(tools/bench_kv_i8.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_kv_i8 PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_kv_i8 PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_kv_i8 PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(bench_kv_fd tools/bench_kv_fd.cpp)
set_source_files_properties(tools/bench_kv_fd.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_kv_fd PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_kv_fd PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_kv_fd PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(bench_rotor tools/bench_rotor.cpp)
set_source_files_properties(tools/bench_rotor.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_rotor PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_rotor PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_rotor PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(bench_prefill_variants tools/bench_prefill_variants.cpp)
set_source_files_properties(tools/bench_prefill_variants.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_prefill_variants PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffast-math -munsafe-fp-atomics
        -Wno-unused-result>)
target_link_libraries(bench_prefill_variants PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_prefill_variants PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(unified_server
    tools/unified_server.cpp
    src/strategy_engine.cpp
    src/agent_watchdog.cpp
    src/backend_hip.cpp
    src/zaya_engine.cpp
    src/backend_npu.cpp
    src/backend_generic.cpp
    src/model_router.cpp
)
set_source_files_properties(src/backend_hip.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/zaya_engine.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/backend_generic.cpp PROPERTIES LANGUAGE CXX)
set_source_files_properties(tools/unified_server.cpp PROPERTIES LANGUAGE CXX)
set_source_files_properties(src/backend_npu.cpp PROPERTIES LANGUAGE CXX)
set_source_files_properties(src/model_router.cpp PROPERTIES LANGUAGE CXX)
target_include_directories(unified_server PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kernels>
    $<INSTALL_INTERFACE:include>)
target_compile_options(unified_server PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wno-unused-result -std=c++23>)
target_compile_definitions(unified_server PRIVATE ROCM_CPP_STATIC_HIP=1 ROCM_CPP_STATIC_NPU=1)
target_link_libraries(unified_server PRIVATE
    backend_manager vl_image dl
    httplib::httplib
    nlohmann_json::nlohmann_json
    rocm_cpp hip::host hip::device)
target_link_options(unified_server PRIVATE -rdynamic -luuid)
set_target_properties(unified_server PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
if(ZINC_ENABLED)
    set_target_properties(unified_server PROPERTIES
        INSTALL_RPATH "${ZINC_REPO_DIR}/zig-out/lib"
        BUILD_RPATH "${ZINC_REPO_DIR}/zig-out/lib")
endif()

add_custom_target(zaya_agent ALL
    DEPENDS unified_server
    COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE:unified_server>"
        "${CMAKE_BINARY_DIR}/zaya_agent"
    COMMENT "Creating zaya_agent -> unified_server copy")
set_property(TARGET unified_server APPEND PROPERTY
    LINK_OPTIONS "-Wl,-rpath,\$ORIGIN")


add_executable(backend_demo tools/backend_demo.cpp)
target_include_directories(backend_demo PRIVATE src/)
target_compile_options(backend_demo PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
target_link_libraries(backend_demo PRIVATE backend_manager dl)

add_executable(test_mamba1_backend tools/test_mamba1_backend.cpp)
target_include_directories(test_mamba1_backend PRIVATE src/ include/)
target_compile_options(test_mamba1_backend PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
target_link_libraries(test_mamba1_backend PRIVATE backend_manager dl)
if(ZINC_ENABLED)
    set_target_properties(backend_demo PROPERTIES
        INSTALL_RPATH "${ZINC_REPO_DIR}/zig-out/lib"
        BUILD_RPATH "${ZINC_REPO_DIR}/zig-out/lib")
endif()

add_executable(vision_qwen2vl_poc tools/vision_qwen2vl_poc.cpp)
target_include_directories(vision_qwen2vl_poc PRIVATE src/)
target_compile_options(vision_qwen2vl_poc PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
target_link_libraries(vision_qwen2vl_poc PRIVATE backend_manager vl_image dl)

# -- vision_server : HTTP server for vision-language inference ---------------
# OpenAI-compatible /v1/chat/completions with image_url support.
# Auto-detects GPU (HIP) and falls back to CPU for image preprocessing.
# Requires a VL-capable text model + mmproj GGUF file.
add_executable(vision_server tools/vision_server.cpp)
target_include_directories(vision_server PRIVATE src/ include/)
target_compile_options(vision_server PRIVATE -O3 -Wall -Wno-unused-result -std=c++23)
target_link_libraries(vision_server PRIVATE
    backend_manager vl_image dl
    httplib::httplib
    nlohmann_json::nlohmann_json)

add_executable(bitnet_tui tools/bitnet_tui.cpp)
target_link_libraries(bitnet_tui PRIVATE
    httplib::httplib
    nlohmann_json::nlohmann_json
    ftxui::screen ftxui::dom ftxui::component)

if(RCPP_HAVE_CK)
    add_subdirectory(ck-prefill)
endif()


# -- npu_engine_hybrid : NPU+GPU hybrid inference engine ------------------
# Guarded: requires a pre-built dequant_q4nx.o from the FastFlowLM toolchain
# which may not exist in all clones. Skipped silently when absent.
if(EXISTS "${CMAKE_SOURCE_DIR}/engine/npu/build/dequant_q4nx.o")
    add_executable(npu_engine_hybrid
        engine/npu/src/npu_engine_hybrid.cpp
        engine/npu/build/dequant_q4nx.o
)
else()
    message(STATUS "npu_engine_hybrid: engine/npu/build/dequant_q4nx.o not found — skipping NPU hybrid target")
endif()
if(TARGET npu_engine_hybrid)
    target_include_directories(npu_engine_hybrid PRIVATE
        ${CMAKE_SOURCE_DIR}/engine/npu
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR})
    target_compile_options(npu_engine_hybrid PRIVATE
        -O3 -march=native -ffast-math -std=c++23 -fopenmp)
    target_link_libraries(npu_engine_hybrid PRIVATE xrt_coreutil uuid rt pthread)
    target_link_options(npu_engine_hybrid PRIVATE
        -L${THEROCK_SDK_BASE}/lib -lamdhip64 -Wl,-rpath,${THEROCK_SDK_BASE}/lib -fopenmp)
    set_target_properties(npu_engine_hybrid PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/engine/npu/build)
endif()

# -- test_pipeline_real: GPU+NPU fused pipeline with real kernels (#446) ------
# Requires HIP, XRT, and the zero-copy pipeline sources.
# Uses relative paths (no variable indirection) so the CI reference checker
# can verify they resolve.
find_package(aiebu QUIET)
find_package(xrt QUIET)
if(hip_FOUND AND xrt_FOUND AND aiebu_FOUND)
    if(EXISTS "${CMAKE_SOURCE_DIR}/engine/fusion/zero_copy/shared_bo.cpp" AND
       EXISTS "${CMAKE_SOURCE_DIR}/engine/fusion/zero_copy/pipeline_overlap.cpp")
        add_executable(test_pipeline_real
            engine/fusion/zero_copy/test_pipeline_real.cpp
            engine/fusion/zero_copy/shared_bo.cpp
            engine/fusion/zero_copy/pipeline_overlap.cpp
            src/kv_cache_attn.hip
        )
        target_include_directories(test_pipeline_real PRIVATE
            engine/fusion/zero_copy
            ${CMAKE_SOURCE_DIR}/include
            ${THEROCK_SDK_BASE}/include)
        target_compile_options(test_pipeline_real PRIVATE
            -O3 -march=native -ffast-math -std=c++23 -fopenmp -D__HIP_PLATFORM_AMD__)
        target_link_libraries(test_pipeline_real PRIVATE
            xrt_coreutil xrt_core uuid rt pthread
            /usr/lib/x86_64-linux-gnu/libaiebu.a)
        target_link_options(test_pipeline_real PRIVATE
            -L${THEROCK_SDK_BASE}/lib -lamdhip64 -Wl,-rpath,${THEROCK_SDK_BASE}/lib -fopenmp)
        set_target_properties(test_pipeline_real PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/engine/fusion/zero_copy
            HIP_ARCHITECTURES "gfx1151")
        add_test(NAME test_pipeline_real COMMAND test_pipeline_real)
        set_tests_properties(test_pipeline_real PROPERTIES
            LABELS "gpu_required;npu_required" SKIP_RETURN_CODE 77
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    else()
        message(STATUS "test_pipeline_real: shared_bo.cpp or pipeline_overlap.cpp not found — skipping")
    endif()
else()
    message(STATUS "test_pipeline_real: HIP/XRT/aiebu not all found — skipping GPU+NPU pipeline test")
endif()

# -- DSpark full pipeline: Eagle3 draft (NPU/CPU) + GPU verify ---------------
add_executable(dspark_full
    tools/dspark_full.cpp
    engine/fusion/cpu_layer.cpp)
target_include_directories(dspark_full PRIVATE
    ${CMAKE_SOURCE_DIR}/engine/fusion
    ${CMAKE_SOURCE_DIR}/spec-decode/draft
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR})
target_compile_options(dspark_full PRIVATE
    -O3 -march=native -ffast-math -std=c++23 -fopenmp)
target_link_libraries(dspark_full PRIVATE
    rocm_cpp
    ${CMAKE_DL_LIBS} m pthread)
target_link_options(dspark_full PRIVATE
    -L${THEROCK_SDK_BASE}/lib -lamdhip64 -Wl,-rpath,${THEROCK_SDK_BASE}/lib)
set_target_properties(dspark_full PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tools)

# -- DSpark GPU bench: full GPU-accelerated speculative decode --------------
find_package(hipblas QUIET)
if(hipblas_FOUND)
add_executable(dspark_gpu_bench
    tools/dspark_gpu_bench.cpp
    engine/fusion/cpu_layer.cpp)
target_include_directories(dspark_gpu_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/engine/fusion
    ${CMAKE_SOURCE_DIR}/spec-decode/draft
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/spec-decode
    ${THEROCK_SDK_BASE}/include)
target_compile_options(dspark_gpu_bench PRIVATE
    -O3 -march=native -ffast-math -std=c++23 -fopenmp -D__HIP_PLATFORM_AMD__)
target_link_libraries(dspark_gpu_bench PRIVATE
    rocm_cpp
    ${CMAKE_DL_LIBS} m pthread hipblas)
target_link_options(dspark_gpu_bench PRIVATE
    -L${THEROCK_SDK_BASE}/lib -lamdhip64 -lhipblas -Wl,-rpath,${THEROCK_SDK_BASE}/lib)
set_target_properties(dspark_gpu_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tools)
else()
    message(STATUS "rocm-cpp: hipblas NOT found — skipping dspark_gpu_bench (needs ROCm hipblas)")
endif()

# -- Q4NX -> TRG converter --
add_executable(trg_save
    tools/trg_save.cpp
    engine/fusion/cpu_layer.cpp)
target_include_directories(trg_save PRIVATE
    ${CMAKE_SOURCE_DIR}/engine/fusion
    ${CMAKE_SOURCE_DIR})
target_compile_options(trg_save PRIVATE
    -O3 -march=native -ffast-math -std=c++23 -fopenmp)
target_link_libraries(trg_save PRIVATE
    ${CMAKE_DL_LIBS} m pthread)
set_target_properties(trg_save PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tools)

# -- OSCAR calibration tool --
add_executable(oscar_calib
    tools/oscar_calib.cpp
    engine/fusion/cpu_layer.cpp)
target_include_directories(oscar_calib PRIVATE
    ${CMAKE_SOURCE_DIR}/engine/fusion
    ${CMAKE_SOURCE_DIR})
target_compile_options(oscar_calib PRIVATE
    -O3 -march=native -ffast-math -std=c++23 -fopenmp)
target_link_libraries(oscar_calib PRIVATE
    ${CMAKE_DL_LIBS} m pthread)
set_target_properties(oscar_calib PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tools)

# -- OSCAR INT2 quant + decode HIP library --
add_library(oscar SHARED
    src/oscar.cpp
    kernels/oscar_quant.hip)
target_include_directories(oscar PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR})
target_compile_options(oscar PRIVATE
    -O3 -ffast-math -std=c++23)
target_link_libraries(oscar PRIVATE
    ${CMAKE_DL_LIBS} m pthread)
target_link_options(oscar PRIVATE
    -L${THEROCK_SDK_BASE}/lib -lamdhip64 -Wl,-rpath,${THEROCK_SDK_BASE}/lib)

add_executable(test_oscar_attn_decode tests/test_oscar_attn_decode.cpp)
target_include_directories(test_oscar_attn_decode PRIVATE
    ${CMAKE_SOURCE_DIR}/include)
target_compile_options(test_oscar_attn_decode PRIVATE -O2 -Wno-unused-result)
target_link_libraries(test_oscar_attn_decode PRIVATE oscar hip::host hip::device)
set_target_properties(test_oscar_attn_decode PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
add_test(NAME test_oscar_attn_decode COMMAND test_oscar_attn_decode)
set_tests_properties(test_oscar_attn_decode PROPERTIES LABELS "gpu_required" SKIP_RETURN_CODE 77)
