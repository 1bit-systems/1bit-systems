# LoRA training tool — C++23, MI300X tensor core TRG support
cmake_minimum_required(VERSION 3.20)

# Host-only build (CPU training, for testing)
add_executable(lora_train train.cpp)
target_include_directories(lora_train PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(lora_train PRIVATE -O3 -march=native -std=c++23)

# MI300X tensor core build (add -DUSE_TRG and link hipblaslt)
option(USE_TRG "Enable MI300X tensor core GEMM for LoRA training" OFF)
if(USE_TRG)
    target_compile_definitions(lora_train PRIVATE USE_TRG=1)
    target_link_libraries(lora_train PRIVATE hip::host hip::device hipblaslt)
    target_compile_options(lora_train PRIVATE --offload-arch=gfx942)
endif()
