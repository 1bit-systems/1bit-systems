# 1bit unified operator CLI (C++23 port of crates/1bit-cli).
#
# Mirrors cpp/mcp/CMakeLists.txt + cpp/landing/CMakeLists.txt conventions:
# - onebit_cli_lib STATIC carries every pure module (TOML, registry, sha256,
#   feed parse, burnin analyzer, OOBE error, snapper text parse).
# - 1bit executable wires the CLI11 dispatch.
# - onebit_cli_test exercises the pure layer; subprocess + HTTP paths are
#   exercised through injected probes / fakes so the suite stays hermetic.

add_library(onebit_cli_lib STATIC
    src/package.cpp
    src/registry.cpp
    src/manifest.cpp
    src/sha256.cpp
    src/install.cpp
    src/install_tracker.cpp
    src/install_model.cpp
    src/service.cpp
    src/doctor.cpp
    src/preflight.cpp
    src/oobe_error.cpp
    src/update_feed.cpp
    src/burnin.cpp
    src/budget.cpp
    src/power.cpp
    src/rollback.cpp
    src/version.cpp
    src/proc.cpp
    src/http.cpp
    src/paths.cpp
    src/push.cpp
)
target_include_directories(onebit_cli_lib PUBLIC include)
target_compile_features(onebit_cli_lib PUBLIC cxx_std_23)
target_link_libraries(onebit_cli_lib
    PUBLIC
        nlohmann_json::nlohmann_json
        tomlplusplus::tomlplusplus
        httplib::httplib
        fmt::fmt
        onebit::ingest          # for SHA-256 (used by R2 SigV4 push)
)
add_library(onebit::cli ALIAS onebit_cli_lib)

# ---- 1bit executable ------------------------------------------------------
add_executable(1bit src/main.cpp)
target_link_libraries(1bit PRIVATE
    onebit::cli
    CLI11::CLI11
)
set_target_properties(1bit PROPERTIES OUTPUT_NAME "1bit")
install(TARGETS 1bit RUNTIME DESTINATION bin)

# ---- tests ----------------------------------------------------------------
if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_cli_test
        tests/main_test.cpp
        tests/package_test.cpp
        tests/registry_test.cpp
        tests/manifest_test.cpp
        tests/sha256_test.cpp
        tests/install_tracker_test.cpp
        tests/install_test.cpp
        tests/oobe_error_test.cpp
        tests/preflight_test.cpp
        tests/update_feed_test.cpp
        tests/burnin_test.cpp
        tests/budget_test.cpp
        tests/power_test.cpp
        tests/rollback_test.cpp
        tests/push_test.cpp
    )
    target_link_libraries(onebit_cli_test PRIVATE
        onebit::cli
        doctest::doctest
    )
    add_test(NAME onebit_cli COMMAND onebit_cli_test)
endif()
