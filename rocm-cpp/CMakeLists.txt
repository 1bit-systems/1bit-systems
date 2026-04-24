cmake_minimum_required(VERSION 3.21)

# Pin HIP compiler to TheRock's clang for native gfx1151.
set(CMAKE_HIP_COMPILER "$ENV{HOME}/therock/build/dist/rocm/lib/llvm/bin/clang++"
    CACHE FILEPATH "HIP compiler")
set(CMAKE_C_COMPILER   "$ENV{HOME}/therock/build/dist/rocm/lib/llvm/bin/clang"
    CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "$ENV{HOME}/therock/build/dist/rocm/lib/llvm/bin/clang++"
    CACHE FILEPATH "C++ compiler")

project(rocm_cpp LANGUAGES C CXX HIP)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CK's host-tensor reference code uses 1D access into 2D tensors — only valid
# when assertions are disabled. Match upstream Release build.
add_compile_definitions(NDEBUG)

# TheRock ROCm dist (CK, HIP, rocBLAS)
set(ROCM_ROOT "$ENV{HOME}/therock/build/dist/rocm")
list(PREPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}" "${ROCM_ROOT}/lib/cmake")

find_package(hip REQUIRED CONFIG PATHS "${ROCM_ROOT}")

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

# CK is still needed for the device-side GEMM template (header-only). Option B
# phases 1-4 will replace that; once they land, this find_package goes away too.
find_package(composable_kernel REQUIRED CONFIG PATHS "${ROCM_ROOT}")

# GPU arch(es) to build for. Default covers every AMD consumer arch with
# Wave32 WMMA — RDNA3, RDNA3.5, and RDNA4 — so one fat binary Just Works
# across the mainstream Radeon + APU lineup:
#
#   RDNA3   (half16, __builtin_amdgcn_wmma_*_w32):
#     gfx1100  Navi 31      RX 7900 XTX / 7900 XT / W7900
#     gfx1101  Navi 32      RX 7800 XT / 7700 XT
#     gfx1102  Navi 33      RX 7600 / 7600 XT / 7700S
#     gfx1103  Phoenix      Radeon 780M / 760M (Ryzen 7040/8040 iGPU)
#
#   RDNA3.5 (half16, __builtin_amdgcn_wmma_*_w32):
#     gfx1150  Strix Point  Radeon 880M / 890M (Ryzen AI 9 HX 370)
#     gfx1151  Strix Halo   Radeon 8060S (Ryzen AI Max+ 395)  <-- reference
#
#   RDNA4   (half8, __builtin_amdgcn_wmma_*_w32_gfx12):
#     gfx1200  Navi 44      RX 9060 (expected)
#     gfx1201  Navi 48      RX 9070 / RX 9070 XT              <-- 2nd target
#
# All eight share the WMMA family src/wmma_compat.hpp already dispatches:
# the RDNA3/3.5 block uses the original `_w32` builtin with half16, the
# RDNA4 block uses the `_w32_gfx12` builtin with half8 + the blocked C
# accumulator mapping. HIP bundles code objects per-arch into one fat
# binary; runtime picks by device.
#
# Override for a narrower slice when disk/build-time matters:
#   -DCMAKE_HIP_ARCHITECTURES="gfx1151"              # Strix Halo only
#   -DCMAKE_HIP_ARCHITECTURES="gfx1151;gfx1201"      # both our hosts
#   -DCMAKE_HIP_ARCHITECTURES="gfx1201"              # RX 9070 XT only
#
# Not on the default list (and why):
#   gfx1030-1036 (RDNA2)   — no WMMA; would need a wave64 / no-WMMA fallback.
#   gfx90a / gfx942 (CDNA) — MFMA, not WMMA; separate port via mfma_compat.hpp.
#   gfx900-906 (GCN)       — too old, no ternary-GEMV ISA win.
#
# Port status (2026-04-22):
#   PORTED    — wmma_peak_probe.hip
#             — prefill_standalone.hip (all 14 kernels, bit-exact vs CK on
#               both gfx1151 + gfx1201)
#             — sdot4/sudot4 kernels (arch-independent builtins, no changes)
#   Track per-arch dispatch in src/wmma_compat.hpp.
set(CMAKE_HIP_ARCHITECTURES
    "gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1200;gfx1201"
    CACHE STRING "GPU arch(es) to build")

# -- librocm_cpp : shared library exposing the C API ----------------------------
# Host packer is pure C++ (no CK). Device kernel uses CK headers only — no
# libutility.a link. Once we vendor the device kernel (Phase 1+), this target
# has no CK dependency at all.
add_library(rocm_cpp SHARED
    src/ck_gemm.cpp
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
    src/bonsai_tq2_gemv.hip
    src/bonsai_gemv_scalar_ref.hip
    kernels/ternary_gemv_phase5_dot4.hip
    kernels/ternary_gemv_phase5_halo.hip
    kernels/ternary_gemv_sherry.hip
    kernels/ternary_gemv_tq1_halo.hip
    kernels/ternary_gemv_tq2_0.hip
    kernels/rotorquant_pack.hip
    kernels/hadamard_rotate_butterfly.hip)
set_source_files_properties(
    src/ck_gemm.cpp
    src/prefill_standalone.hip
    src/wmma_peak_probe.hip
    src/prim_kernels.hip
    src/kv_cache_attn.hip
    src/kv_cache_attn_i8.hip
    src/kv_cache_attn_fd.hip
    src/kv_cache_attn_fd_rotor.hip
    src/ternary_gemm_smallm.hip
    src/ternary_gemm_smallm_scalar_ref.hip
    src/sherry_gemv.hip
    src/sherry_gemv_scalar_ref.hip
    src/bonsai_q1_gemv.hip
    src/bonsai_tq2_gemv.hip
    src/bonsai_gemv_scalar_ref.hip
    kernels/ternary_gemv_phase5_dot4.hip
    kernels/ternary_gemv_phase5_halo.hip
    kernels/ternary_gemv_sherry.hip
    kernels/ternary_gemv_tq1_halo.hip
    kernels/ternary_gemv_tq2_0.hip
    kernels/rotorquant_pack.hip
    kernels/hadamard_rotate_butterfly.hip
    PROPERTIES LANGUAGE HIP)
target_include_directories(rocm_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_include_directories(rocm_cpp SYSTEM PRIVATE "${ROCM_ROOT}/include")
target_compile_options(rocm_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -ffast-math -munsafe-fp-atomics
        -Wno-unused-result -Wno-unused-parameter -Wno-unused-variable>)
target_link_libraries(rocm_cpp PRIVATE hip::host hip::device)
set_target_properties(rocm_cpp PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- Optional tests gate ------------------------------------------------------
# Flip ON to build TQ2_0 (and future opt-in) unit tests. OFF by default so
# library consumers don't pay the compile cost.
option(BUILD_ROCM_CPP_TESTS "Build opt-in rocm-cpp unit tests" OFF)

# -- tests/test_ternary_gemv_tq2_0 : TQ2_0 × Q8_1 GEMV vs CPU reference -------
# Gated by BUILD_ROCM_CPP_TESTS. Usage:
#   cmake -DBUILD_ROCM_CPP_TESTS=ON -B build
#   cmake --build build --target test_ternary_gemv_tq2_0
#   ./build/test_ternary_gemv_tq2_0
if(BUILD_ROCM_CPP_TESTS)
    add_executable(test_ternary_gemv_tq2_0 tests/test_ternary_gemv_tq2_0.cpp)
    set_source_files_properties(tests/test_ternary_gemv_tq2_0.cpp
        PROPERTIES LANGUAGE HIP)
    target_compile_options(test_ternary_gemv_tq2_0 PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
    target_link_libraries(test_ternary_gemv_tq2_0 PRIVATE
        rocm_cpp hip::host hip::device)
    set_target_properties(test_ternary_gemv_tq2_0 PROPERTIES
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endif()

# -- tests/test_ck_gemm : end-to-end functional proof --------------------------
add_executable(test_ck_gemm tests/test_ck_gemm.cpp)
set_source_files_properties(tests/test_ck_gemm.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_ck_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_ck_gemm PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_ck_gemm PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/test_standalone : correctness diff CK vs standalone kernel ----------
add_executable(test_standalone tests/test_standalone.cpp)
set_source_files_properties(tests/test_standalone.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_standalone PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_standalone PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_standalone PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

add_executable(test_prim_and_attn tests/test_prim_and_attn.cpp)
set_source_files_properties(tests/test_prim_and_attn.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_prim_and_attn PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_prim_and_attn PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_prim_and_attn PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/test_attn_degenerate : v0.1.1 NaN-guard regression test ------------
# Passes all-zero K/V into every attention kernel variant (fp16 decode/prefill,
# fp16 split-KV Flash-Decoding, int8 decode) and asserts the output stays
# finite. Catches a future regression on the `inv_l = (l > eps) ? 1/l : 0`
# guard in kv_cache_attn*.hip.
add_executable(test_attn_degenerate tests/test_attn_degenerate.cpp)
set_source_files_properties(tests/test_attn_degenerate.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_attn_degenerate PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_attn_degenerate PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_attn_degenerate PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

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

# -- tools/bitnet_decode : end-to-end MVP decode CLI ---------------------------
add_executable(bitnet_decode tools/bitnet_decode.cpp)
set_source_files_properties(tools/bitnet_decode.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bitnet_decode PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bitnet_decode PRIVATE rocm_cpp hip::host hip::device
                                            httplib::httplib nlohmann_json::nlohmann_json)
set_target_properties(bitnet_decode PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

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

# -- tools/attn_fd_sweep : FD vs single-block attn sweep (JSON stdout) --------
add_executable(attn_fd_sweep tools/attn_fd_sweep.cpp)
set_source_files_properties(tools/attn_fd_sweep.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(attn_fd_sweep PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(attn_fd_sweep PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(attn_fd_sweep PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tools/bench_rotor : rotorquant PlanarQuant-3 KV compression microbench --
add_executable(bench_rotor tools/bench_rotor.cpp)
set_source_files_properties(tools/bench_rotor.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(bench_rotor PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(bench_rotor PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(bench_rotor PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tools/bitnet_tui : FTXUI terminal UI (talks to bitnet_decode --server) ---
add_executable(bitnet_tui tools/bitnet_tui.cpp)
target_link_libraries(bitnet_tui PRIVATE
    httplib::httplib
    nlohmann_json::nlohmann_json
    ftxui::screen ftxui::dom ftxui::component)

# -- ck-prefill : research binaries (kept building for tuning) ------------------
add_subdirectory(ck-prefill)

# -- router-cpp : 1bit.cpp Phase 2 scaffolding (opt-in) ------------------------
# Default OFF — the Rust router is still the shipping one. Flip ON with
#   -DBUILD_ROUTER_CPP=ON
# to build the C++20 mirror side-by-side. See router-cpp/README.md for
# the port status + file map.
option(BUILD_ROUTER_CPP "Build the 1bit.cpp router scaffold (Phase 2 port)" OFF)
if(BUILD_ROUTER_CPP)
    add_subdirectory(router-cpp)
endif()
