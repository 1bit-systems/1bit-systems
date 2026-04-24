# 1bit.cpp — Phase 2 router-cpp scaffold.
#
# Opt-in subdirectory; wired from rocm-cpp/CMakeLists.txt only when the
# consumer sets -DBUILD_ROUTER_CPP=ON. Keeps the default Rust-backed
# build from paying extra compile cost on the iGPU host.
#
# Requires C++20. The rocm-cpp top-level pins CXX_STANDARD 17 because CK
# headers don't tolerate 20; we override locally so router TU's can use
# std::span / concepts / ranges without affecting librocm_cpp's
# compilation. The rcpp_ headers are plain C, so this mix compiles cleanly.

cmake_minimum_required(VERSION 3.21)

add_library(onebit_router_cpp STATIC
    src/chat_template.cpp
    src/sampler.cpp
    src/model.cpp
    src/backend_hip.cpp
    src/router.cpp)

target_include_directories(onebit_router_cpp PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# C++20 override — see comment above.
set_target_properties(onebit_router_cpp PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON)

# Keep warnings loud for the new surface — the Rust bar is already high
# and we don't want silent C++ regressions to creep in.
target_compile_options(onebit_router_cpp PRIVATE
    -Wall -Wextra -Wpedantic
    # Suppressed for the same reason rocm-cpp suppresses them:
    # kernel-adjacent code has legitimate unused params for abstraction
    # hooks; warnings should not block scaffolding commits.
    -Wno-unused-parameter
    # TopKSampler / TopPSampler ctor fields are stashed for the follow-up
    # `sampler/cpu.rs` port; the fields are read there, not in this pass.
    -Wno-unused-private-field)

# Depend on librocm_cpp for the rcpp_bitnet_load_h1b / rcpp_bitnet_free
# entry points (+ future kernel launchers). That's the only runtime dep.
target_link_libraries(onebit_router_cpp PUBLIC rocm_cpp)

# -- tests/test_router_smoke : host-only parity test --------------------------
# No HIP sources; no GPU needed. Asserts ChatTemplate byte-exact parity
# with the Rust reference + the Router scaffolding contract (forward_token
# throws `not yet wired`, load_model rejects missing files, etc).
add_executable(test_router_smoke tests/test_router_smoke.cpp)
set_target_properties(test_router_smoke PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
target_compile_options(test_router_smoke PRIVATE
    -Wall -Wextra -Wpedantic)
target_link_libraries(test_router_smoke PRIVATE onebit_router_cpp)
