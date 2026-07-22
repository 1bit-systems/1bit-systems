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

set(ZINC_SOURCES
    ${ZINC_SRC_DIR}/vulkan_wrapper.cpp
    ${ZINC_SRC_DIR}/compute_engine.cpp
    ${ZINC_SRC_DIR}/model_loader.cpp
    ${ZINC_SRC_DIR}/main.cpp
)

# ── Main executable ──
add_executable(zinc_cpp ${ZINC_SOURCES})

target_include_directories(zinc_cpp PRIVATE
    ${ZINC_INCLUDE_DIR}
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(zinc_cpp PRIVATE
    ${Vulkan_LIBRARIES}
    dl pthread
)

target_compile_options(zinc_cpp PRIVATE
    -O3 -Wall -Wno-unused-result
)

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
            COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O -o ${SPV_OUTPUT} ${SHADER}
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
