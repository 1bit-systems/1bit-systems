# cpp/ — 1bit-systems C++ port

C++20 replacements for the legacy `crates/` Rust workspace. Phase-2 of
`feedback_stack_architecture.md` ("rewrite Rust → C++ for engine-language
monoculture") triggered 2026-04-26.

## Build

```sh
cmake --preset release-strix
cmake --build --preset release-strix
ctest --preset release-strix
```

## Layout

| dir              | replaces (Rust)        | LOC (Rust) |
|------------------|------------------------|------------|
| core/            | crates/1bit-core       | 4667       |
| mcp/             | crates/1bit-mcp        | 182        |
| cli/             | crates/1bit-cli        | 6208       |
| landing/         | crates/1bit-landing    | 1164       |
| power/           | crates/1bit-power      | 851        |
| voice/           | crates/1bit-voice      | 667        |
| echo/            | crates/1bit-echo       | 744        |
| helm/            | crates/1bit-helm       | 2742       |
| helm-tui/        | crates/1bit-helm-tui   | 392        |
| watchdog/        | crates/1bit-watchdog   | 612        |
| stream/          | crates/1bit-stream     | 1095       |
| tier-mint/       | crates/1bit-tier-mint  | 661        |
| ingest/          | crates/1bit-ingest     | 1132       |
| retrieval/       | crates/1bit-retrieval  | 763        |
| halo-ralph/      | crates/1bit-halo-ralph | 247        |
| mcp-clients/     | crates/1bit-mcp-clients| 659        |
| mcp-linuxgsm/    | crates/1bit-mcp-linuxgsm| 257       |
| onnx/            | crates/1bit-onnx       | 996        |
| kokoro/          | crates/1bit-kokoro     | 668 (wraps halo-kokoro upstream) |
| aie/             | crates/1bit-aie        | 1161       |
| tools/h1b-sherry | tools/h1b-sherry       | -          |
| tools/gguf-to-h1b| tools/gguf-to-h1b      | -          |
| tools/bitnet-to-tq2| tools/bitnet-to-tq2  | -          |

Dropped (no port):
- `1bit-mlx` — Apple Silicon, irrelevant on Linux fleet.
- `1bit-agents-legacy` — already excluded from workspace.
- `1bit-mcp-discord-orphan` — already excluded; depends on retired onebit-agents.

## Conventions

- C++20, `-Wall -Wextra -Wpedantic`, no exceptions in hot paths.
- Header-only deps via FetchContent (json, httplib, CLI11, tomlplusplus, FTXUI, spdlog, fmt, doctest).
- Shared targets exported as `onebit::core`, `onebit::mcp`, etc.
- One `CMakeLists.txt` per crate dir.
- Tests live alongside sources as `*_test.cpp`, registered via `add_test`.
