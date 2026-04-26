add_library(onebit_mcp_clients_lib STATIC
    src/protocol.cpp
    src/http.cpp
    src/stdio_client.cpp
    src/github.cpp
    src/semgrep.cpp
)
target_include_directories(onebit_mcp_clients_lib PUBLIC include)
target_link_libraries(onebit_mcp_clients_lib
    PUBLIC
        onebit::mcp
        nlohmann_json::nlohmann_json
        httplib::httplib
)
add_library(onebit::mcp_clients ALIAS onebit_mcp_clients_lib)

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_mcp_clients_test
        tests/protocol_test.cpp
        tests/http_test.cpp
        tests/stdio_client_test.cpp
        tests/github_test.cpp
        tests/semgrep_test.cpp
        tests/test_main.cpp
    )
    target_link_libraries(onebit_mcp_clients_test
        PRIVATE
            onebit::mcp_clients
            doctest::doctest
    )
    add_test(NAME onebit_mcp_clients COMMAND onebit_mcp_clients_test)
endif()
