# 1bit-landing — marketing page + live status probe on :8190.
# Port of crates/1bit-landing/. Mirrors cpp/mcp/CMakeLists.txt conventions.

# ---- Generate embedded-asset translation unit at configure time -----------
set(_LANDING_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/assets")
set(_LANDING_ASSETS_OUT "${CMAKE_CURRENT_BINARY_DIR}/generated/assets.cpp")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")

execute_process(
    COMMAND ${CMAKE_COMMAND}
            -DASSETS_DIR=${_LANDING_ASSETS_DIR}
            -DOUT_FILE=${_LANDING_ASSETS_OUT}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_assets.cmake
    RESULT_VARIABLE _embed_rc
)
if(NOT _embed_rc EQUAL 0)
    message(FATAL_ERROR "embed_assets.cmake failed (rc=${_embed_rc})")
endif()

# Re-run if any asset changes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_LANDING_ASSETS_DIR}/index.html"
    "${_LANDING_ASSETS_DIR}/style.css"
    "${_LANDING_ASSETS_DIR}/logo.svg"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_assets.cmake"
)

# ---- onebit_landing_lib ---------------------------------------------------
add_library(onebit_landing_lib STATIC
    src/router.cpp
    src/server.cpp
    src/status.cpp
    src/telemetry.cpp
    ${_LANDING_ASSETS_OUT}
)
target_include_directories(onebit_landing_lib PUBLIC include)
target_link_libraries(onebit_landing_lib
    PUBLIC
        nlohmann_json::nlohmann_json
        httplib::httplib
)
add_library(onebit::landing ALIAS onebit_landing_lib)

# ---- 1bit-landing exe -----------------------------------------------------
add_executable(1bit-landing src/main.cpp)
target_link_libraries(1bit-landing PRIVATE onebit::landing CLI11::CLI11)
set_target_properties(1bit-landing PROPERTIES OUTPUT_NAME "1bit-landing")

install(TARGETS 1bit-landing RUNTIME DESTINATION bin)

# ---- tests ----------------------------------------------------------------
if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_landing_test tests/landing_test.cpp)
    target_link_libraries(onebit_landing_test PRIVATE
        onebit::landing
        doctest::doctest
    )
    add_test(NAME onebit_landing COMMAND onebit_landing_test)
endif()
