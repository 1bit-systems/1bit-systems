# 1bit-tier-mint — BTCPay webhook -> Premium tier JWT issuance.
# Port of crates/1bit-tier-mint/. HTTP server on 127.0.0.1:8151 (one
# port above 1bit-stream on 8150). Reuses the JWT helpers shipped in
# onebit::stream so /v1/catalogs/:slug/lossless verifies the same way.

add_library(onebit_tier_mint_lib STATIC
    src/btcpay.cpp
    src/jwt.cpp
    src/state.cpp
    src/routes.cpp
    src/server.cpp
)
target_include_directories(onebit_tier_mint_lib PUBLIC include)
target_link_libraries(onebit_tier_mint_lib
    PUBLIC
        onebit::core
        onebit::ingest
        onebit::stream
        nlohmann_json::nlohmann_json
        httplib::httplib
)
add_library(onebit::tier_mint ALIAS onebit_tier_mint_lib)

add_executable(1bit-tier-mint src/main.cpp)
target_link_libraries(1bit-tier-mint PRIVATE onebit::tier_mint CLI11::CLI11)
set_target_properties(1bit-tier-mint PROPERTIES OUTPUT_NAME "1bit-tier-mint")

install(TARGETS 1bit-tier-mint RUNTIME DESTINATION bin)

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_tier_mint_test
        tests/btcpay_test.cpp
        tests/jwt_test.cpp
        tests/state_test.cpp
        tests/routes_test.cpp
    )
    target_link_libraries(onebit_tier_mint_test PRIVATE
        onebit::tier_mint
        doctest::doctest
    )
    add_test(NAME onebit_tier_mint COMMAND onebit_tier_mint_test)
endif()
