# Rust workspace — DEPRECATED

All Rust crates in this directory have been ported to pure C++:

| Rust Crate | C++ Replacement | Location |
|------------|----------------|----------|
| `onebitd` (daemon) | `onebitd` | `tools/onebitd.cpp` |
| `onebit-cli` (CLI agent) | `onebit` / `1bit` | `tools/onebit.cpp` |
| `onebit-tui` (terminal UI) | `bitnet_tui` | `tools/bitnet_tui.cpp` (already existed) |
| `daemon/npu-cppd` | `onebitd` | `tools/onebitd.cpp` |
| `daemon/npu-gpu-cpud` | `onebitd` + `unified_router` | `tools/onebitd.cpp` + `tools/unified_router.cpp` |
| `daemon/1bit-npu-server` | `onebitd` | `tools/onebitd.cpp` |

The C++ versions compile under the main `CMakeLists.txt` and require no Rust
toolchain. They use the same dependencies already fetched by CMake:

- `cpp-httplib` (HTTP server/client — replaces axum + reqwest)
- `nlohmann/json` (JSON — replaces serde_json)
- `FTXUI` (TUI — replaces ratatui + crossterm)

This directory is kept for reference only. All CI/CD (`.github/workflows/ci.yml`,
`.github/workflows/release.yml`) has been updated to build C++ targets exclusively.

To build: `cmake -B build && ninja -C build onebitd onebit onebit_bin bitnet_tui`
