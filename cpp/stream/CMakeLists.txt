# 1bit-stream — .1bl catalog HTTP server.
# Port of crates/1bit-stream/. Lossy tier open, lossless gated on a
# premium HS256 JWT. Reuses ingest's container parser via onebit::ingest.

add_library(onebit_stream_lib STATIC
    src/jwt.cpp
    src/auth.cpp
    src/container.cpp
    src/handlers.cpp
    src/server.cpp
)
target_include_directories(onebit_stream_lib PUBLIC include)
target_link_libraries(onebit_stream_lib
    PUBLIC
        onebit::core
        onebit::ingest
        nlohmann_json::nlohmann_json
        httplib::httplib
)
add_library(onebit::stream ALIAS onebit_stream_lib)

add_executable(1bit-stream src/main.cpp)
target_link_libraries(1bit-stream PRIVATE onebit::stream CLI11::CLI11)
set_target_properties(1bit-stream PROPERTIES OUTPUT_NAME "1bit-stream")

install(TARGETS 1bit-stream RUNTIME DESTINATION bin)

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_stream_test
        tests/auth_test.cpp
        tests/jwt_test.cpp
        tests/container_test.cpp
        tests/handlers_test.cpp
    )
    target_link_libraries(onebit_stream_test PRIVATE
        onebit::stream
        doctest::doctest
    )
    add_test(NAME onebit_stream COMMAND onebit_stream_test)
endif()
