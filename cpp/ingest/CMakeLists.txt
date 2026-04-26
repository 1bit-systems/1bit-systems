# 1bit-ingest — source-side packer for .1bl catalogs.
# Port of crates/1bit-ingest/. CLI tool with prepare / pack / validate /
# add-residual subcommands. Header-only deps (sha256, cbor, tar) live
# under src/internal/.

add_library(onebit_ingest_lib STATIC
    src/sha256.cpp
    src/tar.cpp
    src/cbor.cpp
    src/prepare.cpp
    src/pack.cpp
    src/validate.cpp
    src/residual.cpp
)
target_include_directories(onebit_ingest_lib PUBLIC include)
target_include_directories(onebit_ingest_lib PRIVATE src)
target_link_libraries(onebit_ingest_lib
    PUBLIC
        onebit::core
        tomlplusplus::tomlplusplus
)
add_library(onebit::ingest ALIAS onebit_ingest_lib)

add_executable(1bit-ingest src/main.cpp)
target_link_libraries(1bit-ingest PRIVATE onebit::ingest CLI11::CLI11)
set_target_properties(1bit-ingest PROPERTIES OUTPUT_NAME "1bit-ingest")

install(TARGETS 1bit-ingest RUNTIME DESTINATION bin)

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_ingest_test
        tests/sha256_test.cpp
        tests/cbor_test.cpp
        tests/pack_test.cpp
        tests/validate_test.cpp
        tests/residual_test.cpp
    )
    target_include_directories(onebit_ingest_test PRIVATE src)
    target_link_libraries(onebit_ingest_test PRIVATE
        onebit::ingest
        doctest::doctest
    )
    add_test(NAME onebit_ingest COMMAND onebit_ingest_test)
endif()
