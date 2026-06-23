> **Start here:** Read `docs/wiki/README.md` before any work on this project.

# CLAUDE.md — conventions for 1bit-systems

1-bit inference engine for AMD Strix Halo. Pure Rust HTTP server wrapping
rocm-cpp HIP kernels. Zero Python, zero C++ at the server layer.

## Hard rules

- **Zero Python at runtime.** Build-time tools allowed. Hot path is Rust + HIP.
- **Only Rust for the server.** axum HTTP server. No C++ in the server layer.
  Kernels live in `bong-water-water-bong/rocm-cpp` (C++/HIP).
- **No proxy layer.** The Rust server IS the proxy. No Node.js, no shell-script
  orchestration.
- **OpenAI-compatible.** `/v1/chat/completions`, `/v1/completions`,
  `/v1/models`, `/v1/embeddings`. Standard SDK clients work.
- **Rust 1.88+, edition 2024.**

## Architecture

```
onebit (:13305)  →  axum (Rust)
  └── bitnet_decode --server  →  rocm-cpp (C++/HIP)
       └── librocm_cpp.so  →  ternary GEMV/GEMV, 4.9-7.2x rocBLAS
```

## Repos

| Repo | Role |
|---|---|
| `bong-water-water-bong/1bit-engine` | Rust HTTP server (the runtime) |
| `bong-water-water-bong/rocm-cpp` | C++/HIP kernels (the engine) |
| `bong-water-water-bong/1bit-systems` | Website, docs, benchmarks (this repo) |

## Layout

```
.
├── 1bit-site/     # CF Pages site for 1bit.systems
├── benchmarks/    # benchmark scripts + results
├── docs/          # architecture notes, wiki
```

## What NOT to do

- **Don't add Python to the hot path.**
- **Don't add C++ to the server.** Kernels stay in rocm-cpp.
- **Don't add Node.js or shell-script orchestration.**
- **Don't add a separate proxy — the Rust server IS the proxy.**
- **Don't commit secrets.**
