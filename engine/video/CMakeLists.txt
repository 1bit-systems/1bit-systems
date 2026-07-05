cmake_minimum_required(VERSION 3.12)
project(video-engine C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_C_STANDARD 11)

# Options
option(SD_CUBLAS "Enable CUDA backend" OFF)
option(SD_VULKAN "Enable Vulkan backend" OFF)
option(SD_METAL "Enable Metal backend" OFF)
option(SD_NPU "Enable 1bit NPU backend" OFF)

# Add stable-diffusion.cpp as a subdirectory
add_subdirectory(stable-diffusion.cpp)

# Create video engine wrapper
add_executable(video_engine
    src/video_wrapper.cpp
)

target_include_directories(video_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp/src
)

target_link_libraries(video_engine PRIVATE
    stable-diffusion
)

if(SD_NPU)
    target_compile_definitions(video_engine PRIVATE USE_NPU)
    target_include_directories(video_engine PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../npu/src
    )
    target_link_libraries(video_engine PRIVATE
        xrt_coreutil uuid m dl
    )
endif()

# Smoke test (no model needed — validates API, cache, and config)
add_executable(smoke_test
    src/smoke_test.cpp
)
target_include_directories(smoke_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stable-diffusion.cpp/src
)
target_link_libraries(smoke_test PRIVATE
    stable-diffusion
)

# Install
install(TARGETS video_engine smoke_test RUNTIME DESTINATION bin)

# Copy GGUF model download helper
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/download_model.sh
    ${CMAKE_CURRENT_BINARY_DIR}/bin/download_model.sh
    COPYONLY
)
