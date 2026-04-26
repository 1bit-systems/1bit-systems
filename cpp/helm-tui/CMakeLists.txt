# 1bit-helm-tui — interactive operator TUI (FTXUI port of crates/1bit-helm-tui).
# Mirrors cpp/mcp/CMakeLists.txt conventions.

add_library(onebit_helm_tui_lib STATIC
    src/pane.cpp
    src/layout.cpp
    src/state.cpp
    src/widgets.cpp
)
target_include_directories(onebit_helm_tui_lib PUBLIC include)
target_link_libraries(onebit_helm_tui_lib
    PUBLIC
        nlohmann_json::nlohmann_json
        ftxui::dom
        ftxui::screen
)
add_library(onebit::helm_tui ALIAS onebit_helm_tui_lib)

add_executable(1bit-helm-tui src/main.cpp)
target_link_libraries(1bit-helm-tui PRIVATE
    onebit::helm_tui
    CLI11::CLI11
    ftxui::component
    ftxui::dom
    ftxui::screen
)
set_target_properties(1bit-helm-tui PROPERTIES OUTPUT_NAME "1bit-helm-tui")

install(TARGETS 1bit-helm-tui RUNTIME DESTINATION bin)

if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_helm_tui_test tests/helm_tui_test.cpp)
    target_link_libraries(onebit_helm_tui_test PRIVATE
        onebit::helm_tui
        doctest::doctest
    )
    add_test(NAME onebit_helm_tui COMMAND onebit_helm_tui_test)
endif()
