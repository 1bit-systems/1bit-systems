# 1bit-voice — sentence-boundary streaming voice loop.
# Port of crates/1bit-voice/. Mirrors cpp/mcp/CMakeLists.txt conventions.

add_library(onebit_voice_lib STATIC
    src/splitter.cpp
    src/sse_client.cpp
    src/pipeline.cpp
)
target_include_directories(onebit_voice_lib PUBLIC include)
target_link_libraries(onebit_voice_lib
    PUBLIC
        nlohmann_json::nlohmann_json
        httplib::httplib
        spdlog::spdlog
)
add_library(onebit::voice ALIAS onebit_voice_lib)

add_executable(1bit-voice src/main.cpp)
target_link_libraries(1bit-voice PRIVATE onebit::voice CLI11::CLI11)
set_target_properties(1bit-voice PROPERTIES OUTPUT_NAME "1bit-voice")

install(TARGETS 1bit-voice RUNTIME DESTINATION bin)

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_voice_test tests/voice_test.cpp)
    target_link_libraries(onebit_voice_test
        PRIVATE onebit::voice doctest::doctest)
    add_test(NAME onebit_voice COMMAND onebit_voice_test)
endif()
