if(EXISTS "${CMAKE_SOURCE_DIR}/engine/npu/build/dequant_q4nx.o")
    add_executable(npu_engine_hybrid
        engine/npu/src/npu_engine_hybrid.cpp
        engine/npu/build/dequant_q4nx.o
    )
else()
    message(STATUS "npu_engine_hybrid: dequant_q4nx.o not found — skipping")
endif()
if(TARGET npu_engine_hybrid)
    target_include_directories(npu_engine_hybrid PRIVATE
        ${CMAKE_SOURCE_DIR}/engine/npu
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR})
    target_compile_options(npu_engine_hybrid PRIVATE
        -O3 -march=native -ffast-math -std=c++23 -fopenmp)
    target_link_libraries(npu_engine_hybrid PRIVATE xrt_coreutil uuid rt pthread)
    target_link_options(npu_engine_hybrid PRIVATE
        -L${_RC_ROCM_DEFAULT}/lib/python3.14/site-packages/_rocm_sdk_devel/lib -lamdhip64 -Wl,-rpath,${_RC_ROCM_DEFAULT}/lib/python3.14/site-packages/_rocm_sdk_devel/lib -fopenmp)
    set_target_properties(npu_engine_hybrid PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/engine/npu/build)
endif()