cmake_minimum_required(VERSION 3.19)
project(hip_npu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_path(HIP_INCLUDE_DIR hip_runtime_api.h PATHS /usr/include)

add_library(hip_npu SHARED hip_npu.cpp)
target_include_directories(hip_npu PRIVATE ${HIP_INCLUDE_DIR})
target_link_libraries(hip_npu PRIVATE dl)

install(TARGETS hip_npu LIBRARY DESTINATION lib)
