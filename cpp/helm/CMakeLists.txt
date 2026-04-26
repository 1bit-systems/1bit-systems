# 1bit-helm — Qt6 Widgets desktop client (port of crates/1bit-helm).
#
# Soft-fails when Qt6 isn't present so the rest of the monorepo builds
# fine on machines without Qt — only the GUI binaries get skipped.
# The pure model layer (and its tests) compiles unconditionally.

# ---- Qt6 probe -----------------------------------------------------
# DBus fuels the bearer keyring path + the SNI tray host. Network is
# the QtNetwork path the live SSE worker will use (post-MVP).
find_package(Qt6 6.5 QUIET COMPONENTS Widgets DBus Network)

set(_HELM_HAS_QT FALSE)
if(Qt6_FOUND AND Qt6Widgets_FOUND AND Qt6DBus_FOUND AND Qt6Network_FOUND)
    set(_HELM_HAS_QT TRUE)
else()
    message(WARNING
        "1bit-helm: Qt6 (Widgets+DBus+Network) not found — desktop "
        "binaries will be skipped. Pure model+tray library still "
        "builds and tests still run.")
endif()

# ---- Pure model library (no Qt dep) -------------------------------
add_library(onebit_helm_lib STATIC
    src/conversation.cpp
    src/session.cpp
    src/stream.cpp
    src/models.cpp
    src/telemetry.cpp
    src/conv_log.cpp
    src/tray.cpp
    src/app_model.cpp
    src/lemonade_bundle.cpp
)
target_include_directories(onebit_helm_lib PUBLIC include)
target_link_libraries(onebit_helm_lib PUBLIC nlohmann_json::nlohmann_json httplib::httplib)
add_library(onebit::helm ALIAS onebit_helm_lib)

# ---- Bearer (Qt-aware: keyring path lit up only when QtDBus present) ----
# The bearer file-fallback path compiles without Qt. When Qt6 is
# present we light up the secret-service backend in the same TU
# (gated on ONEBIT_HELM_HAS_QT).
add_library(onebit_helm_bearer STATIC src/bearer.cpp)
target_include_directories(onebit_helm_bearer PUBLIC include)
target_link_libraries(onebit_helm_bearer PUBLIC onebit::helm)
if(_HELM_HAS_QT)
    target_compile_definitions(onebit_helm_bearer PUBLIC ONEBIT_HELM_HAS_QT=1)
    target_link_libraries(onebit_helm_bearer PUBLIC Qt6::Core Qt6::DBus)
    set_target_properties(onebit_helm_bearer PROPERTIES AUTOMOC ON)
endif()
add_library(onebit::helm_bearer ALIAS onebit_helm_bearer)

# ---- Tests run on the pure model + tray + bearer-file path -------
if(ONEBIT_BUILD_TESTS)
    add_executable(onebit_helm_test tests/helm_test.cpp)
    target_link_libraries(onebit_helm_test PRIVATE
        onebit::helm
        onebit::helm_bearer
        doctest::doctest
    )
    add_test(NAME onebit_helm COMMAND onebit_helm_test)
endif()

# ---- Embeddable Lemonade bundle ----------------------------------
# When ON, fetch lemonade-embeddable v10.2.0 ubuntu-x64 tarball at configure
# time and stage it under build/helm-bundle/lemonade/. Install rule drops it
# into the AppImage at share/1bit-helm/lemonade/. Helm spawns this lemond
# as a child on :8180 when no system service is detected.
option(HELM_BUNDLE_LEMONADE "Bundle lemonade-embeddable into 1bit-helm install" OFF)

if(HELM_BUNDLE_LEMONADE)
    set(_LEMONADE_VER 10.2.0)
    set(_LEMONADE_URL "https://github.com/lemonade-sdk/lemonade/releases/download/v${_LEMONADE_VER}/lemonade-embeddable-${_LEMONADE_VER}-ubuntu-x64.tar.gz")
    set(_LEMONADE_STAGE "${CMAKE_BINARY_DIR}/helm-bundle/lemonade")
    set(_LEMONADE_TGZ   "${CMAKE_BINARY_DIR}/helm-bundle/lemonade-embeddable-${_LEMONADE_VER}.tar.gz")

    if(NOT EXISTS "${_LEMONADE_STAGE}/bin/lemond")
        message(STATUS "1bit-helm: fetching Embeddable Lemonade v${_LEMONADE_VER} (Apache-2.0, no telemetry)")
        file(MAKE_DIRECTORY "${_LEMONADE_STAGE}")
        file(DOWNLOAD "${_LEMONADE_URL}" "${_LEMONADE_TGZ}"
             SHOW_PROGRESS
             STATUS  _LEMONADE_DL_STATUS
             TIMEOUT 600
             TLS_VERIFY ON)
        list(GET _LEMONADE_DL_STATUS 0 _LEMONADE_DL_RC)
        if(NOT _LEMONADE_DL_RC EQUAL 0)
            message(WARNING "1bit-helm: lemonade-embeddable download failed (${_LEMONADE_DL_STATUS}); skipping bundle.")
        else()
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E tar xzf "${_LEMONADE_TGZ}"
                WORKING_DIRECTORY "${_LEMONADE_STAGE}"
                RESULT_VARIABLE _LEMONADE_TAR_RC)
            file(REMOVE "${_LEMONADE_TGZ}")
            if(NOT _LEMONADE_TAR_RC EQUAL 0)
                message(WARNING "1bit-helm: lemonade-embeddable extract failed; skipping bundle.")
            endif()
        endif()
    endif()

    if(EXISTS "${_LEMONADE_STAGE}/bin/lemond")
        # Install the entire staged tree under share/1bit-helm/lemonade/ so the
        # AppImage / pkg ships it as a sibling of the helm binary.
        install(DIRECTORY "${_LEMONADE_STAGE}/"
                DESTINATION share/1bit-helm/lemonade
                USE_SOURCE_PERMISSIONS)
        # Compile-time path so the C++ side knows where to find lemond at runtime.
        # (C++ side reads HELM_BUNDLED_LEMOND_PATH; falls back to PATH lookup.)
        add_compile_definitions(HELM_BUNDLE_LEMONADE=1)
    endif()
endif()

# ---- Qt-only artifacts: MainWindow + tray binary ------------------
if(_HELM_HAS_QT)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)

    # AUTOMOC pairs each .cpp with a same-basename header for Q_OBJECT scanning.
    # Our main_window.hpp lives under include/onebit/helm/, not next to the
    # .cpp, so AUTOMOC misses it. List the header explicitly so MOC sees Q_OBJECT.
    add_executable(1bit-helm
        src/main.cpp
        src/main_window.cpp
        include/onebit/helm/main_window.hpp
    )
    target_link_libraries(1bit-helm PRIVATE
        onebit::helm
        onebit::helm_bearer
        Qt6::Widgets
        Qt6::DBus
        Qt6::Network
    )
    target_compile_definitions(1bit-helm PRIVATE ONEBIT_HELM_HAS_QT=1)
    set_target_properties(1bit-helm PROPERTIES
        OUTPUT_NAME "1bit-helm"
        AUTOMOC ON
    )
    install(TARGETS 1bit-helm RUNTIME DESTINATION bin)

    add_executable(1bit-halo-helm-tray src/main_tray.cpp)
    target_link_libraries(1bit-halo-helm-tray PRIVATE
        onebit::helm
        Qt6::Widgets
        Qt6::DBus
    )
    target_compile_definitions(1bit-halo-helm-tray PRIVATE ONEBIT_HELM_HAS_QT=1)
    set_target_properties(1bit-halo-helm-tray PROPERTIES
        OUTPUT_NAME "1bit-halo-helm-tray"
        AUTOMOC ON
    )
    install(TARGETS 1bit-halo-helm-tray RUNTIME DESTINATION bin)
endif()
