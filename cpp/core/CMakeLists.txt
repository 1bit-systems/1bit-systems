add_library(onebit_core_lib STATIC
    src/error.cpp
    src/sampler.cpp
    src/h1b.cpp
    src/htok.cpp
    src/gguf.cpp
)
target_include_directories(onebit_core_lib PUBLIC include)
target_compile_features(onebit_core_lib PUBLIC cxx_std_20)
target_link_libraries(onebit_core_lib PUBLIC nlohmann_json::nlohmann_json)
add_library(onebit::core ALIAS onebit_core_lib)

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_core_test
        tests/sampler_test.cpp
        tests/error_test.cpp
        tests/h1b_test.cpp
        tests/htok_test.cpp
    )
    target_link_libraries(onebit_core_test PRIVATE onebit::core doctest::doctest)
    add_test(NAME onebit_core COMMAND onebit_core_test)
endif()
