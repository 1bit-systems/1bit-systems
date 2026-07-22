
# Consider dependencies only in project.
set(CMAKE_DEPENDS_IN_PROJECT_ONLY OFF)

# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "HIP"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_HIP
  "/home/bcloud/1bit-systems/kernels/bonsai_q1_gemv_soa.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/kernels/bonsai_q1_gemv_soa.hip.o"
  "/home/bcloud/1bit-systems/kernels/hadamard_rotate_butterfly.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/kernels/hadamard_rotate_butterfly.hip.o"
  "/home/bcloud/1bit-systems/kernels/rotorquant_pack.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/kernels/rotorquant_pack.hip.o"
  "/home/bcloud/1bit-systems/kernels/ternary_gemv_block_scaled.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/kernels/ternary_gemv_block_scaled.hip.o"
  "/home/bcloud/1bit-systems/kernels/ternary_gemv_phase5_dot4.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/kernels/ternary_gemv_phase5_dot4.hip.o"
  "/home/bcloud/1bit-systems/kernels/ternary_gemv_phase5_halo.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/kernels/ternary_gemv_phase5_halo.hip.o"
  "/home/bcloud/1bit-systems/kernels/ternary_gemv_sherry.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/kernels/ternary_gemv_sherry.hip.o"
  "/home/bcloud/1bit-systems/kernels/ternary_gemv_tq1_halo.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/kernels/ternary_gemv_tq1_halo.hip.o"
  "/home/bcloud/1bit-systems/kernels/ternary_gemv_wmma.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/kernels/ternary_gemv_wmma.hip.o"
  "/home/bcloud/1bit-systems/kernels/vl_resize_norm.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/kernels/vl_resize_norm.hip.o"
  "/home/bcloud/1bit-systems/kernels/wmma_i8_gemv.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/kernels/wmma_i8_gemv.hip.o"
  "/home/bcloud/1bit-systems/kernels/zaya_moe_ternary_gemv.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/kernels/zaya_moe_ternary_gemv.hip.o"
  "/home/bcloud/1bit-systems/kernels/zaya_moe_tiled_gemv.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/kernels/zaya_moe_tiled_gemv.hip.o"
  "/home/bcloud/1bit-systems/kernels/zaya_moe_wmma_batched.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/kernels/zaya_moe_wmma_batched.hip.o"
  "/home/bcloud/1bit-systems/src/bonsai_gemv_scalar_ref.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/bonsai_gemv_scalar_ref.hip.o"
  "/home/bcloud/1bit-systems/src/bonsai_q1_1024.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/bonsai_q1_1024.hip.o"
  "/home/bcloud/1bit-systems/src/bonsai_q1_gemv.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/bonsai_q1_gemv.hip.o"
  "/home/bcloud/1bit-systems/src/bonsai_tq2_1024.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/bonsai_tq2_1024.hip.o"
  "/home/bcloud/1bit-systems/src/bonsai_tq2_1024_opt.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/bonsai_tq2_1024_opt.hip.o"
  "/home/bcloud/1bit-systems/src/bonsai_tq2_gemv.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/bonsai_tq2_gemv.hip.o"
  "/home/bcloud/1bit-systems/src/fused_gemv_tq2_1024.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/fused_gemv_tq2_1024.hip.o"
  "/home/bcloud/1bit-systems/src/kv_cache_attn.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/kv_cache_attn.hip.o"
  "/home/bcloud/1bit-systems/src/kv_cache_attn_fd.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/kv_cache_attn_fd.hip.o"
  "/home/bcloud/1bit-systems/src/kv_cache_attn_fd_rotor.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/kv_cache_attn_fd_rotor.hip.o"
  "/home/bcloud/1bit-systems/src/kv_cache_attn_i8.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/kv_cache_attn_i8.hip.o"
  "/home/bcloud/1bit-systems/src/laguna_gemv.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/laguna_gemv.hip.o"
  "/home/bcloud/1bit-systems/src/mamba1_engine.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/mamba1_engine.hip.o"
  "/home/bcloud/1bit-systems/src/medusa_small_m_gemv.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/medusa_small_m_gemv.hip.o"
  "/home/bcloud/1bit-systems/src/medusa_tree_attn.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/medusa_tree_attn.hip.o"
  "/home/bcloud/1bit-systems/src/prefill_dispatcher.cpp" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/prefill_dispatcher.cpp.o"
  "/home/bcloud/1bit-systems/src/prefill_standalone.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/prefill_standalone.hip.o"
  "/home/bcloud/1bit-systems/src/prim_kernels.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/prim_kernels.hip.o"
  "/home/bcloud/1bit-systems/src/sherry_gemv.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/sherry_gemv.hip.o"
  "/home/bcloud/1bit-systems/src/sherry_gemv_scalar_ref.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/sherry_gemv_scalar_ref.hip.o"
  "/home/bcloud/1bit-systems/src/ternary_gemm_smallm.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/ternary_gemm_smallm.hip.o"
  "/home/bcloud/1bit-systems/src/ternary_gemm_smallm_scalar_ref.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/ternary_gemm_smallm_scalar_ref.hip.o"
  "/home/bcloud/1bit-systems/src/ternary_gemv_launchers.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/ternary_gemv_launchers.hip.o"
  "/home/bcloud/1bit-systems/src/wmma_peak_probe.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/wmma_peak_probe.hip.o"
  "/home/bcloud/1bit-systems/src/zaya_moe_launcher.hip" "/home/bcloud/1bit-systems/build_cmake/CMakeFiles/rocm_cpp.dir/src/zaya_moe_launcher.hip.o"
  )
set(CMAKE_HIP_COMPILER_ID "Clang")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_HIP
  "MAMBA1_ENGINE_AS_LIB=1"
  "NDEBUG"
  "USE_PROF_API=1"
  "__HIP_PLATFORM_AMD__=1"
  "__HIP_ROCclr__=1"
  "rocm_cpp_EXPORTS"
  )

# The include file search paths:
set(CMAKE_HIP_TARGET_INCLUDE_PATH
  "/home/bcloud/1bit-systems/include"
  "/home/bcloud/1bit-systems/kernels"
  "/opt/rocm-therock/include"
  )

# The set of dependency files which are needed:
set(CMAKE_DEPENDS_DEPENDENCY_FILES
  "/home/bcloud/1bit-systems/src/backend_npu.cpp" "CMakeFiles/rocm_cpp.dir/src/backend_npu.cpp.o" "gcc" "CMakeFiles/rocm_cpp.dir/src/backend_npu.cpp.o.d"
  "/home/bcloud/1bit-systems/src/gguf_loader.cpp" "CMakeFiles/rocm_cpp.dir/src/gguf_loader.cpp.o" "gcc" "CMakeFiles/rocm_cpp.dir/src/gguf_loader.cpp.o.d"
  "/home/bcloud/1bit-systems/src/h1b_loader.cpp" "CMakeFiles/rocm_cpp.dir/src/h1b_loader.cpp.o" "gcc" "CMakeFiles/rocm_cpp.dir/src/h1b_loader.cpp.o.d"
  "/home/bcloud/1bit-systems/src/kv_rotorquant.cpp" "CMakeFiles/rocm_cpp.dir/src/kv_rotorquant.cpp.o" "gcc" "CMakeFiles/rocm_cpp.dir/src/kv_rotorquant.cpp.o.d"
  "/home/bcloud/1bit-systems/src/onnx_loader.cpp" "CMakeFiles/rocm_cpp.dir/src/onnx_loader.cpp.o" "gcc" "CMakeFiles/rocm_cpp.dir/src/onnx_loader.cpp.o.d"
  "/home/bcloud/1bit-systems/src/q4nx_reader.cpp" "CMakeFiles/rocm_cpp.dir/src/q4nx_reader.cpp.o" "gcc" "CMakeFiles/rocm_cpp.dir/src/q4nx_reader.cpp.o.d"
  "/home/bcloud/1bit-systems/src/tokenizer.cpp" "CMakeFiles/rocm_cpp.dir/src/tokenizer.cpp.o" "gcc" "CMakeFiles/rocm_cpp.dir/src/tokenizer.cpp.o.d"
  "" "librocm_cpp.so" "gcc" "CMakeFiles/rocm_cpp.dir/link.d"
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_LINKED_INFO_FILES
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_FORWARD_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
