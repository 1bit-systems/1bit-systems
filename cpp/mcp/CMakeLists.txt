add_library(onebit_mcp_lib STATIC
    src/server.cpp
)
target_include_directories(onebit_mcp_lib PUBLIC include)
target_link_libraries(onebit_mcp_lib PUBLIC nlohmann_json::nlohmann_json)
add_library(onebit::mcp ALIAS onebit_mcp_lib)

add_executable(1bit-mcp src/main.cpp)
target_link_libraries(1bit-mcp PRIVATE onebit::mcp)
set_target_properties(1bit-mcp PROPERTIES OUTPUT_NAME "1bit-mcp")

install(TARGETS 1bit-mcp RUNTIME DESTINATION bin)

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_mcp_test tests/server_test.cpp)
    target_link_libraries(onebit_mcp_test PRIVATE onebit::mcp doctest::doctest)
    add_test(NAME onebit_mcp COMMAND onebit_mcp_test)
endif()
