add_library(onebit_watchdog_lib STATIC
    src/config.cpp
    src/state.cpp
    src/poller.cpp
)
target_include_directories(onebit_watchdog_lib PUBLIC include)
target_link_libraries(onebit_watchdog_lib
    PUBLIC
        nlohmann_json::nlohmann_json
        tomlplusplus::tomlplusplus
        httplib::httplib
        spdlog::spdlog
)
add_library(onebit::watchdog ALIAS onebit_watchdog_lib)

add_executable(1bit-watchdog src/main.cpp)
target_link_libraries(1bit-watchdog PRIVATE onebit::watchdog CLI11::CLI11)
set_target_properties(1bit-watchdog PROPERTIES OUTPUT_NAME "1bit-watchdog")

install(TARGETS 1bit-watchdog RUNTIME DESTINATION bin)

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_watchdog_test
        tests/config_test.cpp
        tests/state_test.cpp
    )
    target_link_libraries(onebit_watchdog_test
        PRIVATE onebit::watchdog doctest::doctest)
    add_test(NAME onebit_watchdog COMMAND onebit_watchdog_test)
endif()
