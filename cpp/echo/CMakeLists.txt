# 1bit-echo — browser-side WebSocket gateway over 1bit-voice.
# Port of crates/1bit-echo/. Mirrors cpp/mcp/CMakeLists.txt conventions.

add_library(onebit_echo_lib STATIC
    src/codec.cpp
    src/ws.cpp
    src/server.cpp
)
target_include_directories(onebit_echo_lib PUBLIC include)
target_link_libraries(onebit_echo_lib
    PUBLIC
        onebit::voice
        nlohmann_json::nlohmann_json
        spdlog::spdlog
)
add_library(onebit::echo ALIAS onebit_echo_lib)

add_executable(1bit-echo src/main.cpp)
target_link_libraries(1bit-echo PRIVATE onebit::echo CLI11::CLI11)
set_target_properties(1bit-echo PROPERTIES OUTPUT_NAME "1bit-echo")

install(TARGETS 1bit-echo RUNTIME DESTINATION bin)

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_echo_test tests/echo_test.cpp)
    target_link_libraries(onebit_echo_test
        PRIVATE onebit::echo doctest::doctest)
    add_test(NAME onebit_echo COMMAND onebit_echo_test)
endif()
