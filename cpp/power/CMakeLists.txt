# onebit::power — Linux power/thermal control for Strix Halo.
# Mirrors crates/1bit-power. Pure C++20; libryzenadj.so is dlopen()d at
# runtime so we do not link against it at build time and the binary
# remains usable on dev boxes that don't ship the .so.

add_library(onebit_power_lib STATIC
    src/profile.cpp
    src/ec.cpp
    src/ryzen.cpp
    src/metrics.cpp
)
target_include_directories(onebit_power_lib PUBLIC include)
target_link_libraries(onebit_power_lib
    PUBLIC
        nlohmann_json::nlohmann_json
        tomlplusplus::tomlplusplus
        spdlog::spdlog
)
# dlopen lives in libdl on glibc; required for the LibBackend resolver.
target_link_libraries(onebit_power_lib PUBLIC ${CMAKE_DL_LIBS})

add_library(onebit::power ALIAS onebit_power_lib)

add_executable(1bit-power src/main.cpp)
target_link_libraries(1bit-power PRIVATE
    onebit::power
    CLI11::CLI11
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)
set_target_properties(1bit-power PROPERTIES OUTPUT_NAME "1bit-power")

install(TARGETS 1bit-power RUNTIME DESTINATION bin)

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_power_test
        tests/profile_test.cpp
        tests/ec_test.cpp
        tests/ryzen_test.cpp
        tests/metrics_test.cpp
    )
    target_link_libraries(onebit_power_test PRIVATE
        onebit::power
        doctest::doctest
    )
    add_test(NAME onebit_power COMMAND onebit_power_test)
endif()
