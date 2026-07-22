# TypeScript NPU Server — DEPRECATED

This TypeScript server (`npu-infer/1bit/src/server.ts`, 218 lines) has been
functionally replaced by the pure C++ inference server:

- **`zaya_server`** (`tests/zaya_server.cpp`) — HIP inference engine with
  OpenAI-compatible HTTP API, model auto-detection, and all backends (NPU,
  GPU, CPU). Built as part of the main CMake build.

- **`onebitd`** (`tools/onebitd.cpp`) — C++ daemon that spawns the inference
  backend as a subprocess and reverse-proxies OpenAI-compatible HTTP requests.

The TypeScript version used Fastify + npx tsx and required a Node.js runtime.
The C++ replacements use `cpp-httplib` (already a CMake FetchContent dependency)
and require zero external runtimes.

This file is kept for reference only. All runtime paths now go through the
CMake-built C++ binaries.
