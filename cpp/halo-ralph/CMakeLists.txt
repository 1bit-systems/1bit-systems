add_library(onebit_halo_ralph_lib STATIC
    src/ralph.cpp
)
target_include_directories(onebit_halo_ralph_lib PUBLIC include)
target_link_libraries(onebit_halo_ralph_lib
    PUBLIC
        nlohmann_json::nlohmann_json
        httplib::httplib
)
add_library(onebit::halo_ralph ALIAS onebit_halo_ralph_lib)

add_executable(1bit-halo-ralph src/main.cpp)
target_link_libraries(1bit-halo-ralph PRIVATE onebit::halo_ralph CLI11::CLI11)
set_target_properties(1bit-halo-ralph PROPERTIES OUTPUT_NAME "1bit-halo-ralph")

install(TARGETS 1bit-halo-ralph RUNTIME DESTINATION bin)

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_halo_ralph_test tests/ralph_test.cpp)
    target_link_libraries(onebit_halo_ralph_test
        PRIVATE onebit::halo_ralph doctest::doctest)
    add_test(NAME onebit_halo_ralph COMMAND onebit_halo_ralph_test)
endif()
