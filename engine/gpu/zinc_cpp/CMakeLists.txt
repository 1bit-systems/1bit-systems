cmake_minimum_required(VERSION 3.21)
project(zinc-cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find Vulkan ──
find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE glslc)

# ── Sources ──
set(ZINC_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(ZINC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ZINC_INCLUDE_DIRS)

set(ZINC_SOURCES
    ${ZINC_SRC_DIR}/vulkan_wrapper.cpp
    ${ZINC_SRC_DIR}/compute_engine.cpp
    ${ZINC_SRC_DIR}/model_loader.cpp
    ${ZINC_SRC_DIR}/main.cpp
)

# gguf_reader is in the repo root's src/ directory.
# When built standalone, CMAKE_CURRENT_SOURCE_DIR = repo-root/engine/gpu/zinc_cpp
# so ../../../ = repo-root. When built as subdirectory from root CMakeLists.txt,
# the root CMake provides the correct include paths directly.
if(NOT ZINC_CPP_AVAILABLE)
    # Standalone build: try to find gguf_reader relative to this project
    # On GitHub: repo-root = CMAKE_CURRENT_SOURCE_DIR/../../..
    set(ZINC_REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
    if(EXISTS ${ZINC_REPO_ROOT}/src/gguf_reader.cpp)
        list(APPEND ZINC_SOURCES ${ZINC_REPO_ROOT}/src/gguf_reader.cpp)
        list(APPEND ZINC_INCLUDE_DIRS ${ZINC_REPO_ROOT}/include ${ZINC_REPO_ROOT}/src)
        message(STATUS "zinc-cpp: using gguf_reader from ${ZINC_REPO_ROOT}")
    else()
        # Try common local paths for development
        foreach(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../.. /home/bcloud/1bit-systems)
            if(EXISTS ${ROOT}/src/gguf_reader.cpp)
                list(APPEND ZINC_SOURCES ${ROOT}/src/gguf_reader.cpp)
                list(APPEND ZINC_INCLUDE_DIRS ${ROOT}/include ${ROOT}/src)
                message(STATUS "zinc-cpp: using gguf_reader from ${ROOT}")
                break()
            endif()
        endforeach()
        if(NOT ZINC_INCLUDE_DIRS)
            message(WARNING "zinc-cpp: gguf_reader not found - model loading disabled")
        endif()
    endif()
endif()

# ── Main executable ──
add_library(zinc_cpp STATIC ${ZINC_SOURCES})

target_include_directories(zinc_cpp PRIVATE
    ${ZINC_INCLUDE_DIR}
    ${Vulkan_INCLUDE_DIRS}
    ${ZINC_INCLUDE_DIRS}
)

target_link_libraries(zinc_cpp PRIVATE
    ${Vulkan_LIBRARIES}
    dl pthread
)

target_compile_options(zinc_cpp PRIVATE
    -O3 -Wall -Wno-unused-result
)

if(ZINC_INCLUDE_DIRS)
    target_compile_definitions(zinc_cpp PRIVATE HAS_GGUF_READER=1)
endif()

# ── Shader compilation: GLSL .comp → SPIR-V .spv ──
# If glslc is available, compile shaders on build
if(GLSLC_EXECUTABLE)
    set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../shaders)
    set(SHADER_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
    
    file(GLOB SHADER_SOURCES ${SHADER_SRC_DIR}/*.comp)
    foreach(SHADER ${SHADER_SOURCES})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        set(SPV_OUTPUT ${SHADER_OUT_DIR}/${SHADER_NAME}.spv)
        add_custom_command(
            OUTPUT ${SPV_OUTPUT}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUT_DIR}
            COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.3 -I${SHADER_SRC_DIR} -O -o ${SPV_OUTPUT} ${SHADER}
            DEPENDS ${SHADER}
            COMMENT "Compiling ${SHADER_NAME}.comp → .spv"
        )
        list(APPEND SPV_FILES ${SPV_OUTPUT})
    endforeach()
    add_custom_target(zinc_shaders ALL DEPENDS ${SPV_FILES})
    add_dependencies(zinc_cpp zinc_shaders)
    
    # Runtime shader directory
    target_compile_definitions(zinc_cpp PRIVATE
        ZINC_SHADER_DIR="${SHADER_OUT_DIR}"
    )
endif()
