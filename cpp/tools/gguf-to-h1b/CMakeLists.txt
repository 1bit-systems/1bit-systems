# gguf-to-h1b — frame a PrismML Bonsai GGUF (Q1_0_g128 / TQ2_0_g128) into a
# .h1b v2 file with the Bonsai flag bits set + emit a sidecar .htok.

add_executable(gguf-to-h1b
    src/main.cpp
)
target_include_directories(gguf-to-h1b PRIVATE include)
target_link_libraries(gguf-to-h1b PRIVATE
    onebit::core
    CLI11::CLI11
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)
target_compile_features(gguf-to-h1b PRIVATE cxx_std_23)
set_target_properties(gguf-to-h1b PROPERTIES OUTPUT_NAME "gguf-to-h1b")

install(TARGETS gguf-to-h1b RUNTIME DESTINATION bin)

if(ONEBIT_BUILD_TESTS)
    add_executable(tools_gguf_to_h1b_test
        tests/gguf_to_h1b_test.cpp
    )
    target_include_directories(tools_gguf_to_h1b_test PRIVATE include)
    target_link_libraries(tools_gguf_to_h1b_test PRIVATE
        onebit::core
        nlohmann_json::nlohmann_json
        doctest::doctest
        spdlog::spdlog
    )
    target_compile_features(tools_gguf_to_h1b_test PRIVATE cxx_std_23)
    add_test(NAME tools_gguf_to_h1b COMMAND tools_gguf_to_h1b_test)
endif()
