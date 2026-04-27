# 1bit-agent — autonomous specialist daemon (Discord/Telegram help-desk bot).
#
# Mirrors cpp/cli/CMakeLists.txt + cpp/halo-ralph/CMakeLists.txt conventions:
# - onebit_agent_lib STATIC carries every pure module (config, brain, memory,
#   loop, adapters, tools).
# - halo-agent executable wires it into a long-running daemon.
# - onebit_agent_test exercises pure layers via doctest.
#
# External deps:
# - nlohmann_json   — /v1 chat-completions JSON, tool-call args
# - httplib         — REST + websocket client to the Discord gateway and
#                     /v1 chat-completions on lemond :8180
# - tomlplusplus    — config file parsing
# - OpenSSL         — TLS for httplib (Discord gateway is wss://)
# - SQLite3         — per-instance memory.db
# - onebit::cli     — proc helpers (subprocess, argv quoting)
# - onebit::ingest  — sha256 (used for cache keys + audit-log content hash)
# - onebit::core    — shared error type + logging helpers via onebit::log

find_package(OpenSSL REQUIRED)
find_package(SQLite3 REQUIRED)
# CMake renamed the FindSQLite3 imported target from `SQLite::SQLite3`
# (legacy, deprecated in CMake 4) to `SQLite3::SQLite3` (modern).
# Older CMake (3.14-3.30) only exports the legacy name; newer CMake
# (4+) emits a deprecation warning when the legacy name is used. Pick
# the right one at configure time so we work on both lanes.
if(TARGET SQLite3::SQLite3)
    set(_onebit_sqlite_target SQLite3::SQLite3)
elseif(TARGET SQLite::SQLite3)
    set(_onebit_sqlite_target SQLite::SQLite3)
else()
    message(FATAL_ERROR "SQLite3 found via find_package but no imported target was created")
endif()

add_library(onebit_agent_lib STATIC
    src/error.cpp
    src/config.cpp
    src/brain.cpp
    src/memory.cpp
    src/loop.cpp
    src/factories_default.cpp
    src/adapter_discord.cpp
    src/adapter_discord_shim.cpp
    src/discord_ws.cpp
    src/adapter_telegram.cpp
    src/tools/registry.cpp
    src/tools/repo_search.cpp
    src/tools/bench_lookup.cpp
    src/tools/install_runbook.cpp
    src/tools/gh_issue_create.cpp
    src/tools/agent_consult.cpp
    src/tools/speak_to_echo.cpp
    src/tools/read_file.cpp
    src/tools/list_files.cpp
    src/tools/git_log.cpp
    src/tools/service_status.cpp
)
target_include_directories(onebit_agent_lib PUBLIC include)
target_include_directories(onebit_agent_lib PRIVATE src)
# Belt-and-suspenders: CMake 3.28 (CI) doesn't always propagate
# INTERFACE_INCLUDE_DIRECTORIES from an ALIAS target (onebit::log →
# onebit_log_lib) into a STATIC consumer's compile line. Add the
# core include path explicitly so adapter_discord.cpp + sibling TUs
# can resolve <onebit/log.hpp>.
target_include_directories(onebit_agent_lib PRIVATE
    ${CMAKE_SOURCE_DIR}/core/include
)
target_compile_features(onebit_agent_lib PUBLIC cxx_std_23)
# httplib's websocket + TLS path needs OPENSSL_SUPPORT defined at the call
# site; set it on the public surface so adapter TUs see it consistently.
target_compile_definitions(onebit_agent_lib PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT=1)
# install_runbook.cpp resolves runbook .md files relative to this baked-in
# absolute path (source-tree). Removes the prior /home/bcloud hardcode that
# broke CI runners. Production installs still fall through to
# ${HOME}/.local/share/1bit-agent/runbooks (resolved at runtime).
target_compile_definitions(onebit_agent_lib PRIVATE
    ONEBIT_AGENT_RUNBOOK_DIR="${CMAKE_CURRENT_SOURCE_DIR}/configs/runbooks"
)
target_link_libraries(onebit_agent_lib
    PUBLIC
        onebit::core
        onebit::log
        onebit::cli            # proc helper (subprocess wrappers)
        onebit::ingest         # sha256
        nlohmann_json::nlohmann_json
        httplib::httplib
        tomlplusplus::tomlplusplus
        OpenSSL::SSL
        OpenSSL::Crypto
        ${_onebit_sqlite_target}
        # factories_default.cpp + adapter_*.cpp include <spdlog/spdlog.h>
        # directly. Locally the FetchContent'd spdlog target propagates its
        # include via some sibling link, but on CI's CMake 3.28 the include
        # path doesn't leak — link it explicitly so every TU sees the
        # header. fmt is spdlog's transitive dep but worth being explicit.
        spdlog::spdlog
        fmt::fmt
)
add_library(onebit::agent ALIAS onebit_agent_lib)

# ---- halo-agent executable ------------------------------------------------
add_executable(halo-agent src/main.cpp)
target_link_libraries(halo-agent PRIVATE
    onebit::agent
    CLI11::CLI11
)
set_target_properties(halo-agent PROPERTIES OUTPUT_NAME "halo-agent")
install(TARGETS halo-agent RUNTIME DESTINATION bin)

# ---- tests ----------------------------------------------------------------
if(ONEBIT_BUILD_TESTS)
    file(GLOB ONEBIT_AGENT_TEST_SOURCES CONFIGURE_DEPENDS tests/*.cpp)
    if(ONEBIT_AGENT_TEST_SOURCES)
        add_executable(onebit_agent_test ${ONEBIT_AGENT_TEST_SOURCES})
        target_include_directories(onebit_agent_test PRIVATE src)
        target_link_libraries(onebit_agent_test PRIVATE
            onebit::agent
            doctest::doctest
        )
        add_test(NAME onebit_agent COMMAND onebit_agent_test)
    else()
        message(STATUS "onebit_agent_test: no tests/*.cpp found yet — skipping")
    endif()
endif()
