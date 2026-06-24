# Project Wiki: 1bit-systems

## Mission
To provide the fastest local 1-bit inference on AMD Strix Halo. The runtime is a pure Rust HTTP server (1bit-engine) wrapping rocm-cpp HIP kernels. Zero Python, zero C++ at the server layer.

## Architecture
- **Runtime:** `onebit` (axum HTTP server, Rust) on port `:13305`
- **Engine:** `bitnet_decode --server` (rocm-cpp, C++/HIP) — spawns as subprocess
- **Kernels:** `librocm_cpp.so` — ternary GEMV (decode) at 7.8× rocBLAS, prefill GEMM at 21.9 TFlops
- **API Surface:** OpenAI-compatible `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`

## Current Performance (ROCm 7.2.4, gfx1151, June 2026)

### Decode GEMV (batch=1, memory-bound)
| Shape | rocm-cpp (µs) | rocBLAS (µs) | Speedup |
|---|---|---|---|
| 2560×2560 | ~30 | 212 | 7.1× |
| 4096×4096 | ~100 | 814 | 8.1× |
| 6912×2560 | 27.0 | ~700 | 7.8× |

### Prefill GEMM (compute-bound)
| Shape | rocm-cpp (TF) | rocBLAS (TF) | B Memory |
|---|---|---|---|
| 2560×6912×2560 | 21.94 | 29.99 | 1/4 |
| 2560×2560×6912 | 20.91 | — | 1/4 |
| 4096×4096×4096 | 19.73 | 28.77 | 1/4 |

Effective throughput per byte: **2.9× rocBLAS**

## Agent Handoff
- **Install:** `git clone 1bit-engine && cargo build --release`
- **Run:** `./target/release/onebit --model model.h1b --port 13305`
- **Tune:** `onebit --model model.h1b --tune-prefill` (auto-selects best kernel)
- **Test:** `curl http://127.0.0.1:13305/v1/models`

## Repos
| Repo | Role | Version |
|---|---|---|
| [rocm-cpp](https://github.com/bong-water-water-bong/rocm-cpp) | HIP kernels (engine) | v0.2.0 |
| [1bit-engine](https://github.com/bong-water-water-bong/1bit-engine) | Rust server (runtime) | 0.1.0 |
| [1bit-systems](https://github.com/bong-water-water-bong/1bit-systems) | Docs, website | — |

## Hard Rules
- **Zero Python at runtime.** Hot path is Rust + HIP.
- **Rust-only server.** No C++ in the server layer.
- **OpenAI-compatible.** `/v1/` endpoints, standard SDK clients work.
- **ROCm 7.2.4+** required for gfx1151.
