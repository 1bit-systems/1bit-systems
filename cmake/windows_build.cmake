# ─────────────────────────────────────────────────────────────
# windows_build.cmake — Windows MSVC build for 1bit-systems
# ─────────────────────────────────────────────────────────────
# This is included by the root CMakeLists.txt when building on
# Windows with MSVC. It overrides the Linux/ROCm-specific paths
# and builds only the portable backends:
#   - CPU generic backend
#   - ZINC Vulkan GPU backend
#   - GGUF / 1BP / ONNX loaders
#   - unified_server, zaya_server (CPU fallback)
#   - CLI tools (onebit, gguf_to_onebp)
#
# Usage:
#   cmake -B build -G "Visual Studio 17 2022" -DCMAKE_TOOLCHAIN_FILE=cmake/windows_build.cmake
#   cmake --build build --config Release
# ─────────────────────────────────────────────────────────────

# ── Skip Linux/ROCm/HIP bootstrap entirely ──
# These are normally set in the root CMakeLists.txt before project()
# We set them here so the root file's discovery logic skips through
set(_RC_LLVM_BIN "MSVC")
set(_RC_ROCM_DEFAULT "" CACHE PATH "ROCm install root")
set(THEROCK_SDK_BASE "" CACHE INTERNAL "TheRock SDK base")

# Mark all ROCm/HIP-related variables as found/nonexistent
set(hip_FOUND FALSE)
set(ROCWMMA_FOUND FALSE)
set(composable_kernel_FOUND FALSE)
set(RCPP_HAVE_CUDA FALSE)
set(RCPP_HAVE_METAL FALSE)
set(RCPP_HAVE_CK FALSE)

# ── Platform detection ──
if(NOT WIN32)
    message(FATAL_ERROR "windows_build.cmake is for Windows/MSVC only")
endif()

# ── Vulkan ──
set(USE_VULKAN ON CACHE BOOL "Build Vulkan backend")
set(RCPP_HAVE_VULKAN FALSE)

# ── OpenMP ──
set(OpenMP_CXX_FOUND FALSE)

# ── Override compiler search ──
# The root CMakeLists.txt sets CMAKE_C_COMPILER and CMAKE_CXX_COMPILER
# to amdclang paths. On Windows/MSVC we must NOT do that.
# This file is loaded via -DCMAKE_TOOLCHAIN_FILE so it runs before project().
# We simply don't set any compiler — CMake will auto-detect MSVC.
