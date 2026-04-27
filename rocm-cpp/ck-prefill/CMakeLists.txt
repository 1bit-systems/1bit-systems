cmake_minimum_required(VERSION 3.21)

# Use TheRock's HIP compiler for native gfx1151. Fall back to system ROCm
# (`/opt/rocm`) when TheRock isn't present — same rocm-llvm tree, identical
# kernel codegen for ROCm 7.x packaged on CachyOS / Arch.
set(_CKPF_THEROCK_BIN "$ENV{HOME}/therock/build/dist/rocm/lib/llvm/bin")
set(_CKPF_SYSTEM_BIN  "/opt/rocm/lib/llvm/bin")
if(EXISTS "${_CKPF_THEROCK_BIN}/clang++")
    set(_CKPF_LLVM_BIN "${_CKPF_THEROCK_BIN}")
    set(_CKPF_ROCM_DEFAULT "$ENV{HOME}/therock/build/dist/rocm")
elseif(EXISTS "${_CKPF_SYSTEM_BIN}/clang++")
    set(_CKPF_LLVM_BIN "${_CKPF_SYSTEM_BIN}")
    set(_CKPF_ROCM_DEFAULT "/opt/rocm")
else()
    message(FATAL_ERROR
        "ck-prefill needs an AMD clang at either ${_CKPF_THEROCK_BIN}/clang++ "
        "or ${_CKPF_SYSTEM_BIN}/clang++.")
endif()

set(CMAKE_HIP_COMPILER "${_CKPF_LLVM_BIN}/clang++" CACHE FILEPATH "HIP compiler")
set(CMAKE_C_COMPILER   "${_CKPF_LLVM_BIN}/clang"   CACHE FILEPATH "C compiler")
set(CMAKE_CXX_COMPILER "${_CKPF_LLVM_BIN}/clang++" CACHE FILEPATH "C++ compiler")

project(ck_prefill LANGUAGES CXX HIP)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Upstream CK examples ship reference code that assumes NDEBUG (1D access into
# 2D host tensor via GetOffsetFromMultiIndex with missing index args works when
# assertions are disabled). Match that.
add_compile_definitions(NDEBUG)

# Point at TheRock's ROCm + CK; system ROCm fallback resolved above.
set(ROCM_ROOT "${_CKPF_ROCM_DEFAULT}" CACHE PATH "ROCm install root")
list(PREPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}" "${ROCM_ROOT}/lib/cmake")

find_package(hip REQUIRED CONFIG PATHS "${ROCM_ROOT}")
find_package(composable_kernel REQUIRED CONFIG COMPONENTS utility PATHS "${ROCM_ROOT}")

set(CMAKE_HIP_ARCHITECTURES "gfx1151")

function(add_ck_example name src)
    add_executable(${name} ${src})
    set_source_files_properties(${src} PROPERTIES LANGUAGE HIP)
    target_compile_options(${name} PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:
            -O3
            -ffast-math
            -munsafe-fp-atomics
            -Wno-unused-result
            -Wno-unused-parameter
            -Wno-unused-variable
        >)
    target_link_libraries(${name} PRIVATE
        composable_kernel::utility
        hip::host
        hip::device)
    set_target_properties(${name} PROPERTIES HIP_ARCHITECTURES "gfx1151")
endfunction()

add_ck_example(gemm_wmma_fp16_v3 gemm_wmma_fp16_v3.cpp)
add_ck_example(gemm_wmma_fp16_pk_i4_v3 gemm_wmma_fp16_pk_i4_v3.cpp)
add_ck_example(gemm_wmma_fp16_pk_i4_v3_pf gemm_wmma_fp16_pk_i4_v3_pf.cpp)
add_ck_example(gemm_wmma_fp16_ternary_as_pk_i4 gemm_wmma_fp16_ternary_as_pk_i4.cpp)
