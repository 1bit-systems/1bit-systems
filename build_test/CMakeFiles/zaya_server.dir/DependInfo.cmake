
# Consider dependencies only in project.
set(CMAKE_DEPENDS_IN_PROJECT_ONLY OFF)

# The set of languages for which implicit dependencies are needed:
set(CMAKE_DEPENDS_LANGUAGES
  "HIP"
  )
# The set of files for implicit dependencies of each language:
set(CMAKE_DEPENDS_CHECK_HIP
  "/home/bcloud/1bit-systems/src/backend_hip.cpp" "/home/bcloud/1bit-systems/build_test/CMakeFiles/zaya_server.dir/src/backend_hip.cpp.o"
  "/home/bcloud/1bit-systems/src/zaya_engine.cpp" "/home/bcloud/1bit-systems/build_test/CMakeFiles/zaya_server.dir/src/zaya_engine.cpp.o"
  "/home/bcloud/1bit-systems/tests/backends/backend_hip_adapter.cpp" "/home/bcloud/1bit-systems/build_test/CMakeFiles/zaya_server.dir/tests/backends/backend_hip_adapter.cpp.o"
  "/home/bcloud/1bit-systems/tests/zaya_server.cpp" "/home/bcloud/1bit-systems/build_test/CMakeFiles/zaya_server.dir/tests/zaya_server.cpp.o"
  )
set(CMAKE_HIP_COMPILER_ID "Clang")

# Preprocessor definitions for this target.
set(CMAKE_TARGET_DEFINITIONS_HIP
  "CPPHTTPLIB_BROTLI_SUPPORT"
  "CPPHTTPLIB_OPENSSL_SUPPORT"
  "CPPHTTPLIB_ZLIB_SUPPORT"
  "NDEBUG"
  "USE_PROF_API=1"
  "__HIP_PLATFORM_AMD__=1"
  "__HIP_ROCclr__=1"
  )

# The include file search paths:
set(CMAKE_HIP_TARGET_INCLUDE_PATH
  "/home/bcloud/1bit-systems/tests"
  "/home/bcloud/1bit-systems/tests/backends"
  "/home/bcloud/1bit-systems/include"
  "/home/bcloud/1bit-systems/src"
  "/home/bcloud/1bit-systems/kernels"
  "_deps/nlohmann_json-src/include"
  "_deps/cpp_httplib-src"
  )

# The set of dependency files which are needed:
set(CMAKE_DEPENDS_DEPENDENCY_FILES
  "/home/bcloud/1bit-systems/tests/backends/backend_cpu.cpp" "CMakeFiles/zaya_server.dir/tests/backends/backend_cpu.cpp.o" "gcc" "CMakeFiles/zaya_server.dir/tests/backends/backend_cpu.cpp.o.d"
  "/home/bcloud/1bit-systems/tests/backends/backend_generic.cpp" "CMakeFiles/zaya_server.dir/tests/backends/backend_generic.cpp.o" "gcc" "CMakeFiles/zaya_server.dir/tests/backends/backend_generic.cpp.o.d"
  "/home/bcloud/1bit-systems/tests/backends/backend_npu.cpp" "CMakeFiles/zaya_server.dir/tests/backends/backend_npu.cpp.o" "gcc" "CMakeFiles/zaya_server.dir/tests/backends/backend_npu.cpp.o.d"
  "/home/bcloud/1bit-systems/tests/backends/backend_zinc.cpp" "CMakeFiles/zaya_server.dir/tests/backends/backend_zinc.cpp.o" "gcc" "CMakeFiles/zaya_server.dir/tests/backends/backend_zinc.cpp.o.d"
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_LINKED_INFO_FILES
  )

# Targets to which this target links which contain Fortran sources.
set(CMAKE_Fortran_TARGET_FORWARD_LINKED_INFO_FILES
  )

# Fortran module output directory.
set(CMAKE_Fortran_TARGET_MODULE_DIR "")
