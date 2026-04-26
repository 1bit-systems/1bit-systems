# bitnet-to-tq2 — repack microsoft/bitnet-b1.58-2B-4T-bf16 → .h1b v4 TQ2_0_g128.

add_executable(bitnet-to-tq2
    src/main.cpp
)
target_include_directories(bitnet-to-tq2 PRIVATE include)
target_link_libraries(bitnet-to-tq2 PRIVATE
    onebit::core
    CLI11::CLI11
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)
target_compile_features(bitnet-to-tq2 PRIVATE cxx_std_23)
set_target_properties(bitnet-to-tq2 PROPERTIES OUTPUT_NAME "bitnet-to-tq2")

install(TARGETS bitnet-to-tq2 RUNTIME DESTINATION bin)

if(ONEBIT_BUILD_TESTS)
    add_executable(tools_bitnet_to_tq2_test
        tests/bitnet_to_tq2_test.cpp
    )
    target_include_directories(tools_bitnet_to_tq2_test PRIVATE include)
    target_link_libraries(tools_bitnet_to_tq2_test PRIVATE
        onebit::core
        nlohmann_json::nlohmann_json
        doctest::doctest
        spdlog::spdlog
    )
    target_compile_features(tools_bitnet_to_tq2_test PRIVATE cxx_std_23)
    add_test(NAME tools_bitnet_to_tq2 COMMAND tools_bitnet_to_tq2_test)
endif()
