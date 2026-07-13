cmake_minimum_required(VERSION 3.16)
project(npu-infer C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Find XRT
find_library(XRT_COREUTIL xrt_coreutil)
find_library(XRT_XRT_CORE xrt_core)
find_package(Threads REQUIRED)

# Build model library
add_library(npu_model STATIC
    src/model.c
)
target_include_directories(npu_model PUBLIC include)

# Build inference engine library
add_library(npu_engine STATIC
    src/engine.cpp
)
target_include_directories(npu_engine PUBLIC include)
target_link_libraries(npu_engine npu_model ${XRT_COREUTIL} Threads::Threads)

# Test weight packer
add_executable(test_weight_packer
    src/test_weight_packer.c
)
target_link_libraries(test_weight_packer npu_model m)

# Original test model
add_executable(test_model
    src/test_model.c
    src/model.c
)
target_include_directories(test_model PRIVATE include)
target_link_libraries(test_model m)

# Instruction dumper (C++)
add_executable(dump_instrs tools/dump_instrs.cpp)
target_link_libraries(dump_instrs dl)
set_target_properties(dump_instrs PROPERTIES LINKER_LANGUAGE CXX)

# Full inference engine
add_executable(npu_infer
    src/main.cpp
    src/engine.cpp
    src/model.c
)
target_include_directories(npu_infer PRIVATE include)
target_link_libraries(npu_infer npu_model ${XRT_COREUTIL} ${XRT_XRT_CORE} Threads::Threads m dl uuid)
target_compile_options(npu_infer PRIVATE -O3 -mavx2 -mfma -mavx512f -ffast-math -march=native -funroll-loops -fno-math-errno -fvect-cost-model=unlimited -g)
