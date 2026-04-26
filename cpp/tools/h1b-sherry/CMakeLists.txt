# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Sherry — see LICENSE-SHERRY.md and SHERRY-FILES.txt at the repo root.
# Commercial use requires a separate license.
#
# h1b-sherry — offline requantizer: TQ1 v4 → Sherry v3 (fp16 dispatch).
#
# Reads a halo .h1b v4 (TQ1, base-3 packing) and emits a .h1b v3 with the
# H1B_FLAG_SHERRY_FP16 bit set. Self-contained: the .h1b payload layout is
# the streaming-style format the Rust onebit-core uses, which is structurally
# distinct from the layer-offset-table format the C++ onebit::core::h1b::File
# parser supports today. We therefore handle .h1b parsing inline and only
# borrow onebit::core for shared types / future cross-cutting wiring.

add_executable(h1b-sherry
    src/main.cpp
)
target_include_directories(h1b-sherry PRIVATE include)
target_link_libraries(h1b-sherry PRIVATE
    onebit::core
    CLI11::CLI11
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)
target_compile_features(h1b-sherry PRIVATE cxx_std_23)
set_target_properties(h1b-sherry PROPERTIES OUTPUT_NAME "h1b-sherry")

install(TARGETS h1b-sherry RUNTIME DESTINATION bin)

if(ONEBIT_BUILD_TESTS)
    add_executable(tools_h1b_sherry_test
        tests/h1b_sherry_test.cpp
    )
    target_include_directories(tools_h1b_sherry_test PRIVATE include)
    target_link_libraries(tools_h1b_sherry_test PRIVATE
        onebit::core
        doctest::doctest
        spdlog::spdlog
    )
    target_compile_features(tools_h1b_sherry_test PRIVATE cxx_std_23)
    add_test(NAME tools_h1b_sherry COMMAND tools_h1b_sherry_test)
endif()
