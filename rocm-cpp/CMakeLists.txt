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

# GPU arch(es) to build for.
#   gfx1151 — Strix Halo iGPU (primary target, RDNA 3.5)
#   gfx1201 — RX 9070 XT / Navi 48 dGPU (secondary target, RDNA 4)
# Default builds a fat binary covering both: HIP bundles code objects per-arch
# and the runtime selects at load time. Override with a single arch for faster
# builds, e.g. -DCMAKE_HIP_ARCHITECTURES=gfx1151.
#
# Arch-specific tuning (block sizes, LDS budgets, VGPR allocation) is currently
# sized for gfx1151. Correctness holds across gfx1151/gfx1201 because the hot
# intrinsics (__builtin_amdgcn_wmma_*_w32, __builtin_amdgcn_sudot4/sdot4) are
# retained on RDNA 4, but peak bandwidth on gfx1201 (GDDR6 ~640 GB/s) needs a
# fresh K-outer tile sweep — tracked in docs/TODO.
set(CMAKE_HIP_ARCHITECTURES "gfx1151;gfx1201" CACHE STRING "GPU arch(es) to build")

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
    src/kv_cache_attn_fd_i8.hip
    src/kv_cache_attn_fd_rotor.hip
    src/kv_rotorquant.cpp
    src/h1b_loader.cpp
    src/tokenizer.cpp
    src/engine.cpp
    src/ternary_gemm_smallm.hip
    src/ternary_gemm_smallm_scalar_ref.hip
    src/sherry_gemv.hip
    src/sherry_gemv_scalar_ref.hip
    src/bonsai_q1_gemv.hip
    src/bonsai_tq2_gemv.hip
    src/bonsai_gemv_scalar_ref.hip
    src/medusa_tree_attn.hip
    src/medusa_small_m_gemv.hip
    src/medusa_loader.cpp
    kernels/ternary_gemv_phase5_dot4.hip
    kernels/ternary_gemv_phase5_halo.hip
    kernels/ternary_gemv_sherry.hip
    kernels/ternary_gemv_tq1_halo.hip
    kernels/ternary_gemv_devscale.hip
    kernels/rotorquant_pack.hip
    kernels/hadamard_rotate_butterfly.hip
    src/hadamard_rotate.hip)
set_source_files_properties(
    src/ck_gemm.cpp
    src/prefill_standalone.hip
    src/wmma_peak_probe.hip
    src/prim_kernels.hip
    src/kv_cache_attn.hip
    src/kv_cache_attn_i8.hip
    src/kv_cache_attn_fd.hip
    src/kv_cache_attn_fd_i8.hip
    src/kv_cache_attn_fd_rotor.hip
    src/ternary_gemm_smallm.hip
    src/ternary_gemm_smallm_scalar_ref.hip
    src/sherry_gemv.hip
    src/sherry_gemv_scalar_ref.hip
    src/bonsai_q1_gemv.hip
    src/bonsai_tq2_gemv.hip
    src/bonsai_gemv_scalar_ref.hip
    src/medusa_tree_attn.hip
    src/medusa_small_m_gemv.hip
    kernels/ternary_gemv_phase5_dot4.hip
    kernels/ternary_gemv_phase5_halo.hip
    kernels/ternary_gemv_sherry.hip
    kernels/ternary_gemv_tq1_halo.hip
    kernels/ternary_gemv_devscale.hip
    kernels/rotorquant_pack.hip
    kernels/hadamard_rotate_butterfly.hip
    src/hadamard_rotate.hip
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

# CTest registration — `ctest --tests-regex medusa` keys on the test names
# below. enable_testing() must come BEFORE add_test() calls, but AFTER the
# project() call (already on line 11). Tests register at the bottom of the
# file once their executables exist.
enable_testing()

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

# -- tests/test_hadamard_butterfly : Hadamard rotate butterfly correctness ----
# BitNet v2 W1.58A4 input pre-rotation kernel. Two checks: (1) bit-exact fp16
# vs scalar reference, (2) per-block bench. Needs ROCm device.
add_executable(test_hadamard_butterfly tests/test_hadamard_butterfly.cpp)
set_source_files_properties(tests/test_hadamard_butterfly.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_hadamard_butterfly PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_hadamard_butterfly PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_hadamard_butterfly PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")

# -- tests/test_medusa_skeleton : ABI/link smoke for Medusa kernel stubs ------
# Scope-pass test (2026-04-24): both launchers return RCPP_NOT_IMPLEMENTED.
# Replaced by a proper differential test once the kernel bodies are authored.
# No GPU device required; this test exercises only the host-side launcher
# early-return path. Plain C++17 TU — no HIP language tagging needed.
add_executable(test_medusa_skeleton tests/test_medusa_skeleton.cpp)
target_compile_options(test_medusa_skeleton PRIVATE -Wall -Wextra -Wno-unused-result)
target_link_libraries(test_medusa_skeleton PRIVATE rocm_cpp)
add_test(NAME medusa_skeleton COMMAND test_medusa_skeleton)

# -- tests/test_medusa_small_m_gemv : bit-exact diff for the small-M GEMV ----
# Compiles as HIP (uses hipLaunchKernelGGL via the launcher + __half ULP
# helpers) and links against rocm_cpp for rcpp_medusa_small_m_gemv. Needs a
# live ROCm device — built always but skipped on hosts without one.
add_executable(test_medusa_small_m_gemv tests/test_medusa_small_m_gemv.cpp)
set_source_files_properties(tests/test_medusa_small_m_gemv.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_medusa_small_m_gemv PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_medusa_small_m_gemv PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_medusa_small_m_gemv PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
add_test(NAME medusa_small_m_gemv COMMAND test_medusa_small_m_gemv)

# -- tests/test_medusa_tree_attn : differential test for tree-parallel FD ----
# Diffs rcpp_medusa_tree_attn_decode_fd at M=1 (bit-exact vs single-Q FD)
# and M>1 (vs host-side Flash-Decoding scalar reference, fp16 ULP <= 64).
# Needs a live ROCm device — early-exits 0 if no device present.
add_executable(test_medusa_tree_attn tests/test_medusa_tree_attn.cpp)
set_source_files_properties(tests/test_medusa_tree_attn.cpp PROPERTIES LANGUAGE HIP)
target_compile_options(test_medusa_tree_attn PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:-O3 -Wno-unused-result>)
target_link_libraries(test_medusa_tree_attn PRIVATE rocm_cpp hip::host hip::device)
set_target_properties(test_medusa_tree_attn PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
add_test(NAME medusa_tree_attn COMMAND test_medusa_tree_attn)

# -- tests/test_medusa_loader : .h1b-medusa v1 sidecar round-trip ------------
# Synthetic-zero writer + happy-path load + four negative cases (bad magic,
# hidden mismatch, vocab mismatch, truncated payload). Exercises the loader
# without needing real Medusa weights; hipMallocs the per-head buffers so it
# requires a live ROCm device (skips with exit 0 otherwise).
add_executable(test_medusa_loader tests/test_medusa_loader.cpp)
target_compile_options(test_medusa_loader PRIVATE -Wall -Wextra -Wno-unused-result)
target_link_libraries(test_medusa_loader PRIVATE rocm_cpp hip::host)
add_test(NAME medusa_loader COMMAND test_medusa_loader)

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

# -- tools/h1b_repack_sherry : C++ replacement for the buggy Python Sherry packer
# Reads HALO_V2 .h1b, writes Sherry v3 (flag SHERRY_FP16). Pure host C++ —
# no HIP language tagging. Links against rocm_cpp for rcpp_sherry_pack and
# the (host-side) loader path; hip::host pulls in libamdhip64 because
# librocm_cpp.so is HIP-linked.
add_executable(h1b_repack_sherry tools/h1b_repack_sherry.cpp)
target_link_libraries(h1b_repack_sherry PRIVATE rocm_cpp hip::host)

# -- tools/sherry_per_row_probe : one-shot diagnostic CPU oracle --------------
# Compares per-row dot products of v2 vs Sherry v3 on layer 0 q_proj using a
# fixed pseudo-Gaussian activation vector. Pure host C++; not linked into the
# runtime path. Used to localize residual PPL gap (uniform vs row-varying
# rel_err tells us scale-bug vs sign-distribution-bug).
add_executable(sherry_per_row_probe tools/sherry_per_row_probe.cpp)
target_link_libraries(sherry_per_row_probe PRIVATE)

# -- tools/sherry_calib_capture : per-projection calibration sidecar emitter --
# Reads .h1b RMSNorm weights and emits a per-(layer, projection) (mu, sigma)
# sidecar consumed by h1b_repack_sherry's round-5 distribution-matched scale
# path. Loader-only host C++ — no GPU, no rocm_cpp link. MIT (the calibration
# step is not part of the Sherry algorithm; the kernel + repacker stay
# PolyForm-Noncommercial).
add_executable(sherry_calib_capture tools/sherry_calib_capture.cpp)
target_link_libraries(sherry_calib_capture PRIVATE)

# -- ck-prefill : research binaries (kept building for tuning) ------------------
add_subdirectory(ck-prefill)

# ============================================================================
# XDNA / AIE2P NPU wrapper (rocm_cpp_xdna)
# ============================================================================
# Default OFF so non-NPU builds keep working without a libxrt dep. When ON,
# we link against the system /usr/lib/libxrt_coreutil.so (XRT 2.21+) and ship
# a sibling library `librocm_cpp_xdna.so` exposing rocm_cpp::XdnaKernel.
#
# The wrapper is intentionally a *separate* shared library (not folded into
# librocm_cpp.so) so the HIP/ROCm path stays libxrt-free for callers that
# only want gfx1151 GPU inference.
option(BUILD_XDNA "Build XDNA2 NPU wrapper (rocm_cpp_xdna) against libxrt" OFF)

if(BUILD_XDNA)
    if(NOT UNIX OR APPLE)
        message(FATAL_ERROR "BUILD_XDNA requires Linux + libxrt (amdxdna driver)")
    endif()

    # XRT ships /usr/include/xrt/*.h and /usr/lib/libxrt_coreutil.so but no
    # CMake config on the Arch package. Use pkg-config; fall back to a manual
    # link if the .pc file is missing.
    find_package(PkgConfig QUIET)
    set(XRT_FOUND_VIA_PC FALSE)
    if(PkgConfig_FOUND)
        pkg_check_modules(XRT QUIET xrt)
        if(XRT_FOUND)
            set(XRT_FOUND_VIA_PC TRUE)
        endif()
    endif()

    if(NOT XRT_FOUND_VIA_PC)
        if(NOT EXISTS "/usr/include/xrt/xrt_kernel.h")
            message(FATAL_ERROR
                "BUILD_XDNA: /usr/include/xrt/xrt_kernel.h missing. "
                "Install xrt (Arch: pacman -S xrt) or set XRT_INCLUDE_DIRS.")
        endif()
        if(NOT EXISTS "/usr/lib/libxrt_coreutil.so")
            message(FATAL_ERROR
                "BUILD_XDNA: /usr/lib/libxrt_coreutil.so missing.")
        endif()
        set(XRT_INCLUDE_DIRS "/usr/include")
        set(XRT_LIBRARIES "/usr/lib/libxrt_coreutil.so")
    endif()

    add_library(rocm_cpp_xdna SHARED
        aie/src/xdna_kernel.cpp
    )
    target_include_directories(rocm_cpp_xdna PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/aie/include>
        $<INSTALL_INTERFACE:include>
    )
    target_include_directories(rocm_cpp_xdna SYSTEM PRIVATE ${XRT_INCLUDE_DIRS})
    target_link_libraries(rocm_cpp_xdna PRIVATE ${XRT_LIBRARIES})
    target_compile_features(rocm_cpp_xdna PUBLIC cxx_std_17)

    add_executable(test_xdna_kernel aie/tests/test_xdna_kernel.cpp)
    target_link_libraries(test_xdna_kernel PRIVATE rocm_cpp_xdna)
    add_test(NAME xdna_kernel_smoke COMMAND test_xdna_kernel)
    message(STATUS "XDNA wrapper: ENABLED (libxrt at ${XRT_LIBRARIES})")

    # ------------------------------------------------------------------
    # rocm_cpp_npu_dispatch — C++23 island bridging Engine::ternary_gemv
    # to the onebit::aie::BitnetGemmAIE2P xclbin engine.
    # ------------------------------------------------------------------
    # rocm_cpp itself is C++17 (HIP TUs), but BitnetGemmAIE2P's public
    # header pulls in <expected> (C++23).  We isolate the cast/host-bridge
    # logic in a single non-HIP TU compiled at cxx_std_23 and present it
    # to engine.cpp through a plain C ABI (engine_npu_dispatch.h).
    #
    # The onebit::aie_bitnet_gemm target lives in cpp/aie/CMakeLists.txt;
    # the parent cpp/CMakeLists.txt orders add_subdirectory(aie) BEFORE
    # add_subdirectory(rocm-cpp) so the alias resolves here.  We probe
    # for the target and degrade gracefully when rocm-cpp is being built
    # standalone (no aie sibling).
    if(TARGET onebit::aie_bitnet_gemm)
        add_library(rocm_cpp_npu_dispatch STATIC
            src/engine_npu_dispatch.cpp)
        target_include_directories(rocm_cpp_npu_dispatch PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>)
        target_include_directories(rocm_cpp_npu_dispatch SYSTEM PRIVATE
            "${ROCM_ROOT}/include")
        target_compile_features(rocm_cpp_npu_dispatch PUBLIC cxx_std_23)
        # tiled_gemv (the pad-and-tile dispatcher) is the wrapper we
        # actually call; it lives in onebit::aie_tiled_gemv next to the
        # leaf BitnetGemmAIE2P engine.  Both are PUBLIC here so the
        # transitive include path picks up the AIE headers.
        set(_npu_dispatch_aie_targets
            onebit::aie_bitnet_gemm)
        if(TARGET onebit::aie_tiled_gemv)
            list(APPEND _npu_dispatch_aie_targets onebit::aie_tiled_gemv)
        endif()
        target_link_libraries(rocm_cpp_npu_dispatch
            PUBLIC  ${_npu_dispatch_aie_targets}
            PRIVATE hip::host)

        # Fold into the main rocm_cpp shared lib + flag the engine TU so
        # the short-circuit compiles.  PRIVATE link is intentional: the
        # NPU bridge is an implementation detail of the engine, not part
        # of librocm_cpp.so's exported surface.
        target_link_libraries(rocm_cpp PRIVATE rocm_cpp_npu_dispatch)
        target_compile_definitions(rocm_cpp PRIVATE ROCM_CPP_HAVE_XDNA=1)
        message(STATUS
            "XDNA dispatch: ENABLED (engine.cpp short-circuits via "
            "rcpp_npu_dispatch_*; HALO_NPU=1 to activate at runtime)")
    else()
        message(STATUS
            "XDNA dispatch: skipped — onebit::aie_bitnet_gemm target not "
            "found.  Build via the cpp/ tower (release-strix preset) for "
            "the engine NPU short-circuit.")
    endif()

    # ------------------------------------------------------------------
    # XDNA ternary GEMV (rocm_cpp_xdna_ternary)
    # ------------------------------------------------------------------
    # Independent shared library + test that wraps the v0 ternary GEMV
    # xclbin. Built when both BUILD_XDNA and BUILD_XDNA_TERNARY are ON.
    # The xclbin is NOT built by CMake — author it via
    #     bash aie/scripts/build_ternary_xclbin.sh 64 64
    # The test skips with exit 0 if the xclbin isn't present, so a clean
    # checkout still passes ctest.
    option(BUILD_XDNA_TERNARY "Build XDNA2 ternary GEMV wrapper (rocm_cpp_xdna_ternary)" OFF)

    if(BUILD_XDNA_TERNARY)
        add_library(rocm_cpp_xdna_ternary SHARED
            aie/src/xdna_ternary_gemv.cpp)
        target_include_directories(rocm_cpp_xdna_ternary PUBLIC
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/aie/include>
            $<INSTALL_INTERFACE:include>)
        target_include_directories(rocm_cpp_xdna_ternary SYSTEM PRIVATE ${XRT_INCLUDE_DIRS})
        target_link_libraries(rocm_cpp_xdna_ternary PRIVATE ${XRT_LIBRARIES})
        # _Float16 is gnu++ extension; needs >=c++20 + clang/gcc.
        target_compile_features(rocm_cpp_xdna_ternary PUBLIC cxx_std_20)

        add_executable(test_xdna_ternary_gemv aie/tests/test_xdna_ternary_gemv.cpp)
        target_compile_features(test_xdna_ternary_gemv PRIVATE cxx_std_20)
        target_link_libraries(test_xdna_ternary_gemv PRIVATE rocm_cpp_xdna_ternary)
        add_test(NAME xdna_ternary_gemv_smoke COMMAND test_xdna_ternary_gemv)
        message(STATUS "XDNA ternary GEMV: ENABLED")
    else()
        message(STATUS "XDNA ternary GEMV: disabled (set -DBUILD_XDNA_TERNARY=ON to enable)")
    endif()
else()
    message(STATUS "XDNA wrapper: disabled (set -DBUILD_XDNA=ON to enable)")
endif()
