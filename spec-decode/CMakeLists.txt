cmake_minimum_required(VERSION 3.20)
project(1bit-spec-decode VERSION 2.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Feature Options ─────────────────────────────────────────────────────────

option(BUILD_TESTING          "Build test suite"           ON)
option(ENABLE_NPU             "Enable NPU (XRT) backend"   ON)
option(ENABLE_GPU             "Enable GPU (Vulkan) backend" OFF)
option(ENABLE_FUSED_PIPELINE  "Enable NPU+GPU fused pipeline" OFF)
option(ENABLE_INT8_QUANT      "Enable INT8 quantization"   ON)
option(ENABLE_CONTEXT_SCHED   "Enable context scheduler"   ON)
option(ENABLE_CURL            "Enable HTTP daemon"         OFF)
option(ENABLE_PROFILER        "Enable stage profiler"      ON)

# ─── Find Dependencies ───────────────────────────────────────────────────────

find_package(OpenMP REQUIRED)

if(ENABLE_NPU)
    find_package(XRT REQUIRED QUIET)
endif()

if(ENABLE_CURL)
    find_package(CURL REQUIRED)
endif()

# ─── Source paths ────────────────────────────────────────────────────────────

set(NPU_SRC_DIR   "/home/bcloud/engine/npu/src")
set(FUSION_SRC_DIR "/home/bcloud/engine/fusion")

# ─── Header-Only Libraries ───────────────────────────────────────────────────

# INT8 Quantization module
add_library(spec_int8_quant INTERFACE)
target_include_directories(spec_int8_quant INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/engine
)

# Attention kernels module
add_library(spec_attention_kernels INTERFACE)
target_include_directories(spec_attention_kernels INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/engine
)

# Context scheduler module
add_library(spec_context_scheduler INTERFACE)
target_include_directories(spec_context_scheduler INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/engine
)
if(ENABLE_CONTEXT_SCHED)
    target_compile_definitions(spec_context_scheduler INTERFACE ENABLE_CONTEXT_SCHED=1)
endif()

# Model registry module
add_library(spec_model_registry INTERFACE)
target_include_directories(spec_model_registry INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/engine
)

# Fused pipeline module
add_library(spec_fused_pipeline INTERFACE)
target_include_directories(spec_fused_pipeline INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/engine
)

# Draft model libraries (existing)
add_library(mtp_draft INTERFACE)
target_include_directories(mtp_draft INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(dspark_draft INTERFACE)
target_include_directories(dspark_draft INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Legacy spec_decode (enhanced)
add_library(spec_decode INTERFACE)
target_link_libraries(spec_decode INTERFACE
    dspark_draft
    mtp_draft
    spec_int8_quant
    spec_attention_kernels
    spec_context_scheduler
    spec_model_registry
    spec_fused_pipeline
)
target_include_directories(spec_decode INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/engine
)

# ─── Binary: spec-decode-all (merged engine + all features) ──────────────────

if(ENABLE_NPU)
    set(NPU_SOURCES
        engine/npu_single_binary.cpp
        ${NPU_SRC_DIR}/dequant_q4nx.c
    )

    add_executable(npu_spec_decode
        ${NPU_SOURCES}
    )
    target_link_libraries(npu_spec_decode PRIVATE
        spec_decode
        spec_int8_quant
        spec_attention_kernels
        spec_context_scheduler
        spec_model_registry
        spec_fused_pipeline
        XRT::xrt_coreutil
        OpenMP::OpenMP_CXX
    )
    target_include_directories(npu_spec_decode PRIVATE
        ${XRT_INCLUDE_DIR}
        ${NPU_SRC_DIR}
    )
    target_compile_definitions(npu_spec_decode PRIVATE
        ENABLE_INT8_QUANT=1
        ENABLE_CONTEXT_SCHED=1
    )

    install(TARGETS npu_spec_decode RUNTIME DESTINATION bin)
endif()

# ─── Binary: spec-decode-fused (NPU+GPU pipeline) ───────────────────────────

if(ENABLE_FUSED_PIPELINE AND ENABLE_NPU)
    # Needs XRT + Vulkan/ROCm
    add_executable(npu_fused_pipeline
        engine/npu_fused_pipeline_main.cpp
    )
    target_link_libraries(npu_fused_pipeline PRIVATE
        spec_fused_pipeline
        spec_decode
        spec_int8_quant
        spec_attention_kernels
        spec_model_registry
        OpenMP::OpenMP_CXX
    )
    target_include_directories(npu_fused_pipeline PRIVATE
        ${XRT_INCLUDE_DIR}
        ${NPU_SRC_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/engine
    )

    install(TARGETS npu_fused_pipeline RUNTIME DESTINATION bin)
endif()

# ─── Benchmark Tools ─────────────────────────────────────────────────────────

# INT8 quantization benchmark
add_executable(int8_quant_bench
    bench/int8_quant_bench.cpp
)
target_link_libraries(int8_quant_bench PRIVATE
    spec_int8_quant
    OpenMP::OpenMP_CXX
)

# Attention kernel benchmark
add_executable(attn_kernel_bench
    bench/attn_kernel_bench.cpp
)
target_link_libraries(attn_kernel_bench PRIVATE
    spec_attention_kernels
    OpenMP::OpenMP_CXX
)

# Scheduler benchmark
add_executable(sched_bench
    bench/sched_bench.cpp
)
target_link_libraries(sched_bench PRIVATE
    spec_context_scheduler
    OpenMP::OpenMP_CXX
)

# Legacy benchmark (simulated target)
add_executable(spec_decode_bench
    bench/spec_decode_bench.cpp
)
target_link_libraries(spec_decode_bench PRIVATE
    spec_decode
    OpenMP::OpenMP_CXX
)

# Pipeline cost estimator
add_executable(pipeline_bench
    bench/pipeline_bench.cpp
)
target_link_libraries(pipeline_bench PRIVATE
    spec_fused_pipeline
    spec_int8_quant
    spec_attention_kernels
    OpenMP::OpenMP_CXX
)

# ─── Integration Test Suite ──────────────────────────────────────────────────

if(BUILD_TESTING)
    enable_testing()

    # Legacy tests
    add_executable(test_draft_model
        tests/test_draft_model.cpp
    )
    target_link_libraries(test_draft_model PRIVATE mtp_draft)
    add_test(NAME test_draft_model COMMAND test_draft_model)

    add_executable(test_verify_loop
        tests/test_decode_loop.cpp
    )
    target_link_libraries(test_verify_loop PRIVATE spec_decode)
    add_test(NAME test_verify_loop COMMAND test_verify_loop)

    # ── New test: INT8 Quantization ──
    add_executable(test_int8_quant
        engine/tests/test_int8_quant.cpp
    )
    target_link_libraries(test_int8_quant PRIVATE
        spec_int8_quant
        OpenMP::OpenMP_CXX
    )
    add_test(NAME test_int8_quant COMMAND test_int8_quant)

    # ── New test: Attention Kernels ──
    add_executable(test_attention_kernels
        engine/tests/test_attention_kernels.cpp
    )
    target_link_libraries(test_attention_kernels PRIVATE
        spec_attention_kernels
        OpenMP::OpenMP_CXX
    )
    add_test(NAME test_attention_kernels COMMAND test_attention_kernels)

    # ── New test: Context Scheduler ──
    add_executable(test_context_scheduler
        engine/tests/test_context_scheduler.cpp
    )
    target_link_libraries(test_context_scheduler PRIVATE
        spec_context_scheduler
    )
    add_test(NAME test_context_scheduler COMMAND test_context_scheduler)

    # ── New test: Model Registry ──
    add_executable(test_model_registry
        engine/tests/test_model_registry.cpp
    )
    target_link_libraries(test_model_registry PRIVATE
        spec_model_registry
    )
    add_test(NAME test_model_registry COMMAND test_model_registry)

    # ── New test: Fused Pipeline ──
    add_executable(test_fused_pipeline
        engine/tests/test_fused_pipeline.cpp
    )
    target_link_libraries(test_fused_pipeline PRIVATE
        spec_fused_pipeline
        spec_attention_kernels
        spec_int8_quant
        OpenMP::OpenMP_CXX
    )
    add_test(NAME test_fused_pipeline COMMAND test_fused_pipeline)

    # ── New test: Full integration ──
    add_executable(test_spec_decode_integration
        engine/tests/test_spec_decode_integration.cpp
    )
    target_link_libraries(test_spec_decode_integration PRIVATE
        spec_decode
        spec_int8_quant
        spec_attention_kernels
        spec_context_scheduler
        spec_model_registry
        spec_fused_pipeline
        OpenMP::OpenMP_CXX
    )
    add_test(NAME test_spec_decode_integration COMMAND test_spec_decode_integration)

    # ── NPU acceptance test (needs hardware) ──
    if(ENABLE_NPU)
        add_executable(test_acceptance
            tools/test_acceptance.cpp
            ${NPU_SRC_DIR}/dequant_q4nx.c
        )
        target_link_libraries(test_acceptance PRIVATE
            spec_decode
            spec_int8_quant
            XRT::xrt_coreutil
        )
        target_include_directories(test_acceptance PRIVATE
            ${XRT_INCLUDE_DIR}
            ${NPU_SRC_DIR}
        )
        add_test(NAME test_acceptance COMMAND test_acceptance)
    endif()
endif()

# ─── Install Headers ─────────────────────────────────────────────────────────

install(DIRECTORY engine/ draft/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
