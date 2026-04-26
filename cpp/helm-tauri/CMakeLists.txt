# helm-tauri — opt-in target. The Tauri shell is built via npm/cargo;
# CMake just orchestrates the shell-out so the existing
# `cmake --build --preset release-strix --target helm-tauri` UX works.
#
# Default OFF. Flip with -DONEBIT_BUILD_HELM_TAURI=ON.

option(ONEBIT_BUILD_HELM_TAURI "Build the Tauri+React tiling shell" OFF)

if(ONEBIT_BUILD_HELM_TAURI)
    find_program(NPM_BIN npm REQUIRED)
    find_program(CARGO_BIN cargo REQUIRED)

    add_custom_target(helm-tauri
        COMMAND ${NPM_BIN} install
        COMMAND ${NPM_BIN} run tauri build
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "helm-tauri: npm install + tauri build"
        USES_TERMINAL)
endif()
