add_library(onebit_mcp_linuxgsm_lib STATIC
    src/server.cpp
)
target_include_directories(onebit_mcp_linuxgsm_lib PUBLIC include)
target_link_libraries(onebit_mcp_linuxgsm_lib
    PUBLIC
        onebit::mcp
        nlohmann_json::nlohmann_json
)
add_library(onebit::mcp_linuxgsm ALIAS onebit_mcp_linuxgsm_lib)

add_executable(1bit-mcp-linuxgsm src/main.cpp)
target_link_libraries(1bit-mcp-linuxgsm PRIVATE onebit::mcp_linuxgsm)
set_target_properties(1bit-mcp-linuxgsm PROPERTIES OUTPUT_NAME "1bit-mcp-linuxgsm")

install(TARGETS 1bit-mcp-linuxgsm RUNTIME DESTINATION bin)

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_mcp_linuxgsm_test tests/server_test.cpp)
    target_link_libraries(onebit_mcp_linuxgsm_test
        PRIVATE
            onebit::mcp_linuxgsm
            doctest::doctest
    )
    add_test(NAME onebit_mcp_linuxgsm COMMAND onebit_mcp_linuxgsm_test)
endif()
