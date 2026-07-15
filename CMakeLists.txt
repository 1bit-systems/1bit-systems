cmake_minimum_required(VERSION 3.21)

# Pin HIP compiler to TheRock's clang for native gfx1151. Discover TheRock
# from multiple locations in priority order:
#   1. pip-installed TheRock ROCm SDK (from rocm.nightlies.amd.com)
#   2. lemonade cache (~/.cache/lemonade/bin/therock)
#   3. legacy therock path (~/therock/build/dist/rocm)
#   4. system ROCm (/opt/rocm)
set(_RC_THEROCK_PIP     "$ENV{HOME}/.cache/pip/therock" CACHE PATH "pip-installed TheRock root")
set(_RC_LEMONADE_THEROCK "$ENV{HOME}/.cache/lemonade/bin/therock")
set(_RC_THEROCK_LEGACY "$ENV{HOME}/therock/build/dist/rocm")
set(_RC_SYSTEM_BIN     "/opt/rocm/lib/llvm/bin")

set(_RC_LLVM_BIN "")
set(_RC_ROCM_DEFAULT "")

# 1. Try pip-installed TheRock (from virtual env) — check for rocm-sdk path
if(DEFINED ENV{THEROCK_PIP_ROOT})
    set(_RC_PIP_PATH "$ENV{THEROCK_PIP_ROOT}")
    if(EXISTS "${_RC_PIP_PATH}/bin/amdclang++")
        set(_RC_LLVM_BIN "${_RC_PIP_PATH}/bin")
        set(_RC_ROCM_DEFAULT "${_RC_PIP_PATH}")
        message(STATUS "rocm-cpp: using pip TheRock at ${_RC_ROCM_DEFAULT}")
    endif()
endif()

# 2. Lemonade cache
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

# 3. Legacy therock path
if(NOT _RC_LLVM_BIN)
    if(EXISTS "${_RC_THEROCK_LEGACY}/lib/llvm/bin/clang++")
        set(_RC_LLVM_BIN "${_RC_THEROCK_LEGACY}/lib/llvm/bin")
        set(_RC_ROCM_DEFAULT "${_RC_THEROCK_LEGACY}")
        message(STATUS "rocm-cpp: using legacy TheRock at ${_RC_ROCM_DEFAULT}")
    endif()
endif()

# 4. System ROCm
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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CK's host-tensor reference code uses 1D access into 2D tensors — only valid
# when assertions are disabled. Match upstream Release build.
add_compile_definitions(NDEBUG)

# ROCm dist root (CK, HIP, rocBLAS) — TheRock if present, else system ROCm.
set(ROCM_ROOT "${_RC_ROCM_DEFAULT}" CACHE PATH "ROCm install root")
list(PREPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}" "${ROCM_ROOT}/lib/cmake")

find_package(hip REQUIRED CONFIG PATHS "${ROCM_ROOT}")

# rocWMMA header library — locate the include path for hardware matrix unit intrinsics.
# Search in: pip TheRock SDK, legacy therock, and system ROCm.
set(ROCWMMA_FOUND FALSE)
foreach(_rc_base "${_RC_PIP_PATH}" "${_RC_THEROCK_DIR}" "${_RC_THEROCK_LEGACY}" "/opt/rocm")
    if(_rc_base AND EXISTS "${_rc_base}/include/rocwmma/rocwmma.hpp")
        set(ROCWMMA_INCLUDE_DIR "${_rc_base}/include")
        set(ROCWMMA_FOUND TRUE)
        message(STATUS "rocm-cpp: rocWMMA found at ${ROCWMMA_INCLUDE_DIR}")
        break()
    endif()
endforeach()
if(NOT ROCWMMA_FOUND)
    # Fallback: search in pip site-packages under ~/.local
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


# Header-only HTTP + JSON for bitnet_server's OpenAI-compatible API.
# Pulled at configure time; no runtime deps added beyond libc + pthreads.
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

# CK is needed for the device-side GEMM template (header-only) in ck_gemm.cpp.
# Option B phases 1-4 will replace it entirely. Make it optional: if CK is
# available, build ck_gemm.cpp; otherwise fall back to the standalone prefill.
set(RCPP_HAVE_CK FALSE)
find_package(composable_kernel QUIET CONFIG PATHS "${ROCM_ROOT}")
if(composable_kernel_FOUND)
    set(RCPP_HAVE_CK TRUE)
    message(STATUS "rocm-cpp: composable_kernel FOUND — building ck_gemm.cpp")
else()
    message(STATUS "rocm-cpp: composable_kernel NOT found — skipping ck_gemm.cpp, using standalone prefill")
endif()

# Portable Vulkan backend proof (test_vulkan_gemv): standalone correctness +
# throughput tool for the TQ2/Q1 Bonsai formats on Vulkan, alongside the
# existing ROCm path. Off by default -- opt in with -DUSE_VULKAN=ON. Not
# wired into bitnet_decode.cpp yet; see docs/vulkan-backend.md-style notes
# in tests/test_vulkan_gemv.cpp for why.
option(USE_VULKAN "Build the portable Vulkan backend proof (test_vulkan_gemv)" OFF)
# Note: Vulkan is OFF by default. The README's claimed Vulkan backend (22 tok/s)
# requires USE_VULKAN=ON at cmake configure time. Without it, only ROCm HIP is active.
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

# GPU arch(es) to build for. Default is Strix Halo; overridable from the
# command line with -DCMAKE_HIP_ARCHITECTURES="gfx1151;gfx1201" to produce
# a multi-arch binary (HIP bundles code objects per-arch, runtime picks).
set(CMAKE_HIP_ARCHITECTURES "gfx1151" CACHE STRING "GPU arch(es) to build")

# -- librocm_cpp : shared library exposing the C API ----------------------------
# Host packer is pure C++ (no CK). Device kernel uses CK headers only — no
# libutility.a link. Once we vendor the device kernel (Phase 1+), this target
# has no CK dependency at all.
# Collect all sources; conditionally include ck_gemm.cpp if CK is available.
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
    kernels/rotorquant_pack.hip
    kernels/hadamard_rotate_butterfly.hip
    kernels/zaya_moe_ternary_gemv.hip
    kernels/zaya_moe_tiled_gemv.hip
    kernels/zaya_moe_wmma_batched.hip
    kernels/wmma_i8_gemv.hip
    src/zaya_moe_launcher.hip
    src/bonsai_q1_1024.hip
    src/bonsai_tq2_1024.hip
    src/fused_gemv_tq2_1024.hip
    src/bonsai_tq2_1024_opt.hip)
if(RCPP_HAVE_CK)
    list(APPEND RCPP_SOURCES src/ck_gemm.cpp)
endif()
# NOTE: Some source files listed in RCPP_SOURCES are governed by the PolyForm
# Noncommercial License (see LICENSE-SHERRY.md and SHERRY-FILES.txt).
# The repo's overall license is MIT, but those specific files are restricted
# to noncommercial use. The 'halo' kernel path (not sherry) is the MIT fallback.
# To build without sherry-restricted files, exclude the files listed in SHERRY-FILES.txt
# from RCPP_SOURCES (see docs/ for instructions).
add_library(rocm_cpp SHARED ${RCPP_SOURCES})
if(ROCWMMA_FOUND)
    target_include_directories(rocm_cpp SYSTEM PRIVATE "${ROCWMMA_INCLUDE_DIR}")
endif()
target_sources(rocm_cpp PRIVATE src/prefill_dispatcher.cpp)

# -- libbackend_manager : Pure C++ backend abstraction layer (no HIP) ---------
# Windows ML-style orchestrator: auto-detect, select, failover, monitor, plugins.
# Standalone library — no GPU dependencies. Links against libdl for plugin loading.
add_library(backend_manager STATIC
    src/backend_factory.cpp
    src/backend_manager.cpp
    src/backend_monitor.cpp
    src/backend_plugin.cpp
    src/backend_cpu.cpp)          # pure C++ reference backend, always compiled in
target_include_directories(backend_manager PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(backend_manager PUBLIC dl)
# backend_manager stays HIP-free (no ROCM_CPP_STATIC_HIP) so pure-C++
# consumers like backend_demo can link without HIP symbols.
# ROCM_CPP_STATIC_HIP is set only on targets (e.g., unified_server)
# that also link src/backend_hip.cpp directly.
target_compile_options(backend_manager PRIVATE -O3 -Wall -Wno-unused-result -std=c++17)

# Derive HIP source list from RCPP_SOURCES to avoid maintaining a duplicate list.
# Files with .hip extension need LANGUAGE HIP. prefill_dispatcher.cpp uses HIP
# runtime calls and is added separately via target_sources, so include it too.
set(RCPP_HIP_SOURCES ${RCPP_SOURCES} src/prefill_dispatcher.cpp)
list(FILTER RCPP_HIP_SOURCES INCLUDE REGEX "\\.hip$|prefill_dispatcher\\.cpp$")
if(RCPP_HAVE_CK)
    list(APPEND RCPP_HIP_SOURCES src/ck_gemm.cpp)
endif()
set_source_files_properties(${RCPP_HIP_SOURCES} PROPERTIES LANGUAGE HIP)
target_include_directories(rocm_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_include_directories(rocm_cpp SYSTEM PRIVATE "${ROCM_ROOT}/include")
target_compile_options(rocm_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffast-math -munsafe-fp-atomics
        -Wno-unused-result -Wno-unused-parameter -Wno-unused-variable>)
target_link_libraries(rocm_cpp PRIVATE hip::host hip::device)
set_target_properties(rocm_cpp PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/test_ck_gemm : end-to-end functional proof (requires CK) -------------
if(RCPP_HAVE_CK)
    add_executable(test_ck_gemm tests/test_ck_gemm.cpp)
    set_source_files_properties(tests/test_ck_gemm.cpp PROPERTIES LANGUAGE HIP)
    target_compile_options(test_ck_gemm PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
    target_link_libraries(test_ck_gemm PRIVATE rocm_cpp hip::host hip::device)
    set_target_properties(test_ck_gemm PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endif()

# -- tests/test_standalone : correctness diff CK vs standalone kernel ----------
# Only built when CK is available (guarded by RCPP_HAVE_CK above).
# Otherwise, test_standalone_nock replaces it with a CPU reference.
if(RCPP_HAVE_CK)
    add_executable(test_standalone tests/test_standalone.cpp)
    set_source_files_properties(tests/test_standalone.cpp PROPERTIES LANGUAGE HIP)
    target_compile_options(test_standalone PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
    target_link_libraries(test_standalone PRIVATE rocm_cpp hip::host hip::device)
    set_target_properties(test_standalone PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endif()

# -- tests/test_standalone_nock : standalone kernels vs CPU reference ----------
# Always compiled; replaces test_standalone when CK is unavailable.
add_executable(test_standalone_nock tests/test_standalone_nock.cpp)
set_source_files_properties(tests/test_standalone_nock.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_standalone_nock PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_standalone_nock PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_standalone_nock PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(test_prim_and_attn tests/test_prim_and_attn.cpp)
set_source_files_properties(tests/test_prim_and_attn.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_prim_and_attn PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_prim_and_attn PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_prim_and_attn PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/test_ternary_gemm_smallm : tiled small-M ternary GEMM parity + timing
add_executable(test_ternary_gemm_smallm tests/test_ternary_gemm_smallm.cpp)
set_source_files_properties(tests/test_ternary_gemm_smallm.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_ternary_gemm_smallm PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_ternary_gemm_smallm PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_ternary_gemm_smallm PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/test_sherry_gemv : Sherry 1.25-bpw GEMV differential test ----------
add_executable(test_sherry_gemv tests/test_sherry_gemv.cpp)
set_source_files_properties(tests/test_sherry_gemv.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_sherry_gemv PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_sherry_gemv PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_sherry_gemv PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/test_bonsai_gemv : Bonsai Q1 / TQ2 GEMV differential test ---------
add_executable(test_bonsai_gemv tests/test_bonsai_gemv.cpp)
set_source_files_properties(tests/test_bonsai_gemv.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_bonsai_gemv PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_bonsai_gemv PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_bonsai_gemv PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/bench_bonsai_tile8 : Q1_0 tile8-interleaved GEMV benchmark -------
add_executable(bench_bonsai_tile8 tests/bench_bonsai_tile8.cpp)
set_source_files_properties(tests/bench_bonsai_tile8.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_bonsai_tile8 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_bonsai_tile8 PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_bonsai_tile8 PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/bench_bonsai_tile8_model : full 28-layer model with tile8 GEMVs ---
add_executable(bench_bonsai_tile8_model tests/bench_bonsai_tile8_model.cpp)
set_source_files_properties(tests/bench_bonsai_tile8_model.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_bonsai_tile8_model PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_bonsai_tile8_model PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_bonsai_tile8_model PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/bench_bonsai_q1_1024 : Q1_1024 1024-weight block format benchmark --
add_executable(bench_bonsai_q1_1024 tests/bench_bonsai_q1_1024.cpp)
set_source_files_properties(tests/bench_bonsai_q1_1024.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_bonsai_q1_1024 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_bonsai_q1_1024 PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_bonsai_q1_1024 PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/bench_bonsai_tq2_1024 : TQ2_1024 1024-weight block benchmark ------
add_executable(bench_bonsai_tq2_1024 tests/bench_bonsai_tq2_1024.cpp)
set_source_files_properties(tests/bench_bonsai_tq2_1024.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_bonsai_tq2_1024 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_bonsai_tq2_1024 PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_bonsai_tq2_1024 PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/bench_fused_tq2_1024 : Fused QKV + Gate/Up TQ2_1024 benchmark -----
add_executable(bench_fused_tq2_1024 tests/bench_fused_tq2_1024.cpp)
set_source_files_properties(tests/bench_fused_tq2_1024.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_fused_tq2_1024 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_fused_tq2_1024 PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_fused_tq2_1024 PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/bench_cb_compare : cache-blocking vs original comparison ----------
add_executable(bench_cb_compare tests/bench_cb_compare.cpp)
set_source_files_properties(tests/bench_cb_compare.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_cb_compare PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_cb_compare PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_cb_compare PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/test_vulkan_gemv : portable Vulkan TQ2/Q1 GEMV correctness+bench --
if(RCPP_HAVE_VULKAN)
    set(VK_SHADER_DIR "${CMAKE_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${VK_SHADER_DIR}")
    set(VK_SHADER_SOURCES
        kernels/vulkan/dmmv_tq2_bonsai.comp
        kernels/vulkan/dmmv_q1_bonsai.comp)
    set(VK_SHADER_SPVS "")
    foreach(_vk_src ${VK_SHADER_SOURCES})
        get_filename_component(_vk_name "${_vk_src}" NAME_WE)
        set(_vk_spv "${VK_SHADER_DIR}/${_vk_name}.spv")
        add_custom_command(
            OUTPUT "${_vk_spv}"
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

# -- tests/test_sherry_e2e : fp16-Sherry loader + decode smoke test -----------
# Loads the real halo-1bit-2b-sherry-v4.h1b file, runs one decode token
# through the fp16-Sherry dispatch path, asserts finite + non-trivial logits.
# Build-only by default; executable is run by hand on the strix-halo box.
add_executable(test_sherry_e2e tests/test_sherry_e2e.cpp)
set_source_files_properties(tests/test_sherry_e2e.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_sherry_e2e PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_sherry_e2e PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_sherry_e2e PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/test_zaya_moe_gemv : Zaya MoE ternary GEMV functional test ---------
add_executable(test_zaya_moe_gemv tests/test_zaya_moe_gemv.cpp)
set_source_files_properties(tests/test_zaya_moe_gemv.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_zaya_moe_gemv PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_zaya_moe_gemv PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_zaya_moe_gemv PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/test_cca_attn : Zaya CCA attention kernel test ---------------------
add_executable(test_cca_attn tests/test_cca_attn.cpp kernels/zaya_cca_attn.hip)
set_source_files_properties(tests/test_cca_attn.cpp kernels/zaya_cca_attn.hip PROPERTIES LANGUAGE HIP)
target_compile_options(test_cca_attn PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_cca_attn PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_cca_attn PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/zaya_full : Complete Zaya full GPU inference (40-layer loop) ---------
# Build: cmake --build . --target zaya_full -j8
# Run:   ./zaya_full
add_executable(zaya_full tests/zaya_full.cpp)
set_source_files_properties(tests/zaya_full.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(zaya_full PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(zaya_full PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(zaya_full PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/zaya_gpu_decode : Zaya GPU Decode Engine — loads Q4NX, autoregressive gen -
# Build: cmake --build . --target zaya_gpu_decode -j8
# Run:   ./zaya_gpu_decode /path/to/model.q4nx [--prompt N] [--tokens N]
add_executable(zaya_gpu_decode tests/zaya_gpu_decode.cpp kernels/zaya_cca_attn.hip)
set_source_files_properties(tests/zaya_gpu_decode.cpp kernels/zaya_cca_attn.hip PROPERTIES LANGUAGE HIP)
target_compile_options(zaya_gpu_decode PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_include_directories(zaya_gpu_decode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(zaya_gpu_decode PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(zaya_gpu_decode PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/zaya_server : Zaya HTTP server (multi-backend: HIP + CPU fallback) --
# Build: cmake --build . --target zaya_server -j8
# Run:   ./zaya_server
# The unified binary compiles HIP + CPU backends and auto-detects hardware.
add_executable(zaya_server tests/zaya_server.cpp tests/backends/backend_cpu.cpp tests/backends/backend_hip.cpp tests/backends/backend_npu.cpp)
set_source_files_properties(tests/zaya_server.cpp tests/backends/backend_hip.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(zaya_server PRIVATE ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/backends)
target_compile_options(zaya_server PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(zaya_server PRIVATE rocm_cpp hip::host hip::device httplib::httplib nlohmann_json::nlohmann_json)
set_target_properties(zaya_server PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- kernels/zaya_moe_tiled_gemv.hip : WMMA-tiled MoE expert FFN kernel ---------------
# Part of zaya_full/zaya_server at link time; no standalone target needed.
# The WMMA kernel is compiled into the `rocm_cpp` device library or linked
# directly. To verify the WMMA kernel in isolation, use test_zaya_moe_gemv.

# -- tests/test_wmma_i8_gemv : WMMA INT8×INT8 GEMV functional test -------------
# Verifies Hadamard-rotated INT8 decode against scalar reference.
# Build: cmake --build . --target test_wmma_i8_gemv -j8
# Run:   ./test_wmma_i8_gemv
add_executable(test_wmma_i8_gemv tests/test_wmma_i8_gemv.cpp)
set_source_files_properties(tests/test_wmma_i8_gemv.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_wmma_i8_gemv PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_wmma_i8_gemv PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_wmma_i8_gemv PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/test_bonsai_e2e : Bonsai (Qwen3) loader + decode smoke test --------
# Loads bonsai-1.7b-tq2.h1b + sidecar GGUF, runs one decode token through
# the Qwen3-flavored dispatch path (per-head q/k norms, SwiGLU, tied LM
# head via the dequantized TQ2 embedding). Asserts no launch error + non-
# zero + finite logits. Works even when the sibling HIP kernel pass hasn't
# landed — the bonsai_*_gemv_launch symbols are weak, dispatch short-
# circuits to zeros, and the test still validates the loader + sidecar.
add_executable(test_bonsai_e2e tests/test_bonsai_e2e.cpp)
set_source_files_properties(tests/test_bonsai_e2e.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_bonsai_e2e PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_bonsai_e2e PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_bonsai_e2e PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/test_medusa_skeleton : ABI/link smoke for Medusa kernel stubs ------
# Scope-pass test (2026-04-24): both launchers return RCPP_NOT_IMPLEMENTED.
# Replaced by a proper differential test once the kernel bodies are authored.
# No GPU device required; this test exercises only the host-side launcher
# early-return path. Plain C++17 TU — no HIP language tagging needed.
add_executable(test_medusa_skeleton tests/test_medusa_skeleton.cpp)
target_compile_options(test_medusa_skeleton PRIVATE -Wall -Wextra -Wno-unused-result)
target_link_libraries(test_medusa_skeleton PRIVATE rocm_cpp)

# -- tools/bitnet_decode : end-to-end MVP decode CLI ---------------------------
add_executable(bitnet_decode tools/bitnet_decode.cpp)
set_source_files_properties(tools/bitnet_decode.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bitnet_decode PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bitnet_decode PRIVATE rocm_cpp hip::host hip::device
                                            httplib::httplib nlohmann_json::nlohmann_json)
set_target_properties(bitnet_decode PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tools/hadamard_export : F16 GGUF → Hadamard-rotated INT8 .h1b converter --
add_executable(hadamard_export tools/hadamard_export.cpp)
target_compile_options(hadamard_export PRIVATE -O3 -Wall -Wno-unused-result -std=c++17)
target_link_libraries(hadamard_export PRIVATE rocm_cpp)

# -- tools/bitnet_debate : two BitNet instances arguing a topic ----------------
add_executable(bitnet_debate tools/bitnet_debate.cpp)
set_source_files_properties(tools/bitnet_debate.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bitnet_debate PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bitnet_debate PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bitnet_debate PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tools/bench_sherry : Lane B microbench — halo v2 vs Sherry v3 packing ---
add_executable(bench_sherry tools/bench_sherry.cpp)
set_source_files_properties(tools/bench_sherry.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_sherry PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_sherry PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_sherry PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tools/bench_kv_i8 : INT8 KV cache decode attention microbench ------------
add_executable(bench_kv_i8 tools/bench_kv_i8.cpp)
set_source_files_properties(tools/bench_kv_i8.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_kv_i8 PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_kv_i8 PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_kv_i8 PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tools/bench_kv_fd : split-KV Flash-Decoding attention microbench ---------
add_executable(bench_kv_fd tools/bench_kv_fd.cpp)
set_source_files_properties(tools/bench_kv_fd.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_kv_fd PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_kv_fd PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_kv_fd PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tools/bench_rotor : rotorquant PlanarQuant-3 KV compression microbench --
add_executable(bench_rotor tools/bench_rotor.cpp)
set_source_files_properties(tools/bench_rotor.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_rotor PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_rotor PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_rotor PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tools/bench_prefill_variants : benchmark all prefill kernel variants -------
# Uses the same -O3/-ffast-math/-munsafe-fp-atomics kernel flags as
# librocm_cpp so the bench binary measures the same codegen it produces.
# (Inlined in the genex rather than a CMake list var, matching every other
# target in this file and avoiding the "-O" arg-split footgun.)
add_executable(bench_prefill_variants tools/bench_prefill_variants.cpp)
set_source_files_properties(tools/bench_prefill_variants.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_prefill_variants PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffast-math -munsafe-fp-atomics
        -Wno-unused-result -Wno-unused-parameter -Wno-unused-variable>)
target_link_libraries(bench_prefill_variants PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_prefill_variants PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tools/unified_server : One binary, all backends, auto-failover ----------
add_executable(unified_server
    tools/unified_server.cpp
    src/backend_hip.cpp
    src/zaya_engine.cpp)
set_source_files_properties(src/backend_hip.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(src/zaya_engine.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(tools/unified_server.cpp PROPERTIES LANGUAGE CXX)
target_include_directories(unified_server PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>)
target_compile_options(unified_server PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result -Wno-unused-parameter -Wno-unused-variable>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wno-unused-result -std=c++17>)
target_compile_definitions(unified_server PRIVATE ROCM_CPP_STATIC_HIP=1)
target_link_libraries(unified_server PRIVATE
    backend_manager dl
    httplib::httplib
    nlohmann_json::nlohmann_json
    rocm_cpp hip::host hip::device)
target_link_options(unified_server PRIVATE -rdynamic)
set_target_properties(unified_server PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tools/backend_demo : Backend Manager live dashboard (Task Manager NPU tab) ---
# Pure C++ (no HIP) — demonstrates auto-detect, selection, failover, monitoring.
add_executable(backend_demo tools/backend_demo.cpp)
target_compile_options(backend_demo PRIVATE -O3 -Wall -Wno-unused-result -std=c++17)
target_link_libraries(backend_demo PRIVATE backend_manager dl)

# -- tools/bitnet_tui : FTXUI terminal UI (talks to bitnet_decode --server) ---
add_executable(bitnet_tui tools/bitnet_tui.cpp)
target_link_libraries(bitnet_tui PRIVATE
    httplib::httplib
    nlohmann_json::nlohmann_json
    ftxui::screen ftxui::dom ftxui::component)

# -- ck-prefill : research binaries (kept building for tuning) ------------------
# ck-prefill requires composable_kernel; skip it when CK is unavailable.
if(RCPP_HAVE_CK)
    add_subdirectory(ck-prefill)
endif()
