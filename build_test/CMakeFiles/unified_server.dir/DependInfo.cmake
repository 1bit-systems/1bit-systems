
# Consider dependencies only in project.
set(CMAKE_DEPENDS_IN_PROJECT_ONLY OFF)

# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "HIP"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_HIP
  "/home/bcloud/1bit-systems/src/backend_hip.cpp" "/home/bcloud/1bit-systems/build_test/CMakeFiles/unified_server.dir/src/backend_hip.cpp.o"
  "/home/bcloud/1bit-systems/src/zaya_engine.cpp" "/home/bcloud/1bit-systems/build_test/CMakeFiles/unified_server.dir/src/zaya_engine.cpp.o"
  )
set(CMAKE_HIP_COMPILER_ID "Clang")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_HIP
  "CPPHTTPLIB_BROTLI_SUPPORT"
  "CPPHTTPLIB_OPENSSL_SUPPORT"
  "CPPHTTPLIB_ZLIB_SUPPORT"
  "NDEBUG"
  "ROCM_CPP_STATIC_HIP=1"
  "ROCM_CPP_STATIC_NPU=1"
  "USE_PROF_API=1"
  "__HIP_PLATFORM_AMD__=1"
  "__HIP_ROCclr__=1"
  )

# The include file search paths:
set(CMAKE_HIP_TARGET_INCLUDE_PATH
  "/home/bcloud/1bit-systems/include"
  "/home/bcloud/1bit-systems/src"
  "/home/bcloud/1bit-systems/kernels"
  "/home/bcloud/1bit-systems/engine/gpu/zinc_cpp/include"
  "_deps/nlohmann_json-src/include"
  "_deps/cpp_httplib-src"
  )

# The set of dependency files which are needed:
set(CMAKE_DEPENDS_DEPENDENCY_FILES
  "/home/bcloud/1bit-systems/src/agent_watchdog.cpp" "CMakeFiles/unified_server.dir/src/agent_watchdog.cpp.o" "gcc" "CMakeFiles/unified_server.dir/src/agent_watchdog.cpp.o.d"
  "/home/bcloud/1bit-systems/src/backend_generic.cpp" "CMakeFiles/unified_server.dir/src/backend_generic.cpp.o" "gcc" "CMakeFiles/unified_server.dir/src/backend_generic.cpp.o.d"
  "/home/bcloud/1bit-systems/src/backend_npu.cpp" "CMakeFiles/unified_server.dir/src/backend_npu.cpp.o" "gcc" "CMakeFiles/unified_server.dir/src/backend_npu.cpp.o.d"
  "/home/bcloud/1bit-systems/src/model_router.cpp" "CMakeFiles/unified_server.dir/src/model_router.cpp.o" "gcc" "CMakeFiles/unified_server.dir/src/model_router.cpp.o.d"
  "/home/bcloud/1bit-systems/src/strategy_engine.cpp" "CMakeFiles/unified_server.dir/src/strategy_engine.cpp.o" "gcc" "CMakeFiles/unified_server.dir/src/strategy_engine.cpp.o.d"
  "/home/bcloud/1bit-systems/tools/unified_server.cpp" "CMakeFiles/unified_server.dir/tools/unified_server.cpp.o" "gcc" "CMakeFiles/unified_server.dir/tools/unified_server.cpp.o.d"
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_LINKED_INFO_FILES
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_FORWARD_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
