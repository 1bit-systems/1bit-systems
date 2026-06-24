<div align="center">

<img src="1bit-site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# Local 1-bit inference, wired for Strix Halo.

### Pure Rust. Zero Python.

**[→ Project Wiki](docs/wiki/README.md)** — architecture, decisions, and agent onboarding.

`1bit.systems` is a 1-bit inference engine for AMD Strix Halo (gfx1151). The
runtime is a Rust HTTP server that wraps [rocm-cpp](https://github.com/bong-water-water-bong/rocm-cpp)
HIP kernels, delivering 4.9–7.2× faster decode than rocBLAS FP16 at 1/4 the memory.

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.systems-12a0ed.svg)](https://1bit.systems)
[![Discord](https://img.shields.io/badge/discord-1bit.systems-f00fd2.svg?logo=discord&logoColor=white)](https://discord.gg/dSyV646eBs)
[![Endpoint](https://img.shields.io/badge/endpoint-:13305%2Fv1-00ff00.svg)](#connect-apps)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)

</div>

---

## Architecture

```
onebit (:13305)   axum (Rust)
  └── bitnet_decode --server   rocm-cpp (C++/HIP)
       └── librocm_cpp.so      ternary GEMV/GEMV
            └── gfx1151        Strix Halo iGPU
```

**Zero Python. Zero C++ at the server layer.** One Rust binary spawns one
C++/HIP subprocess. Streaming passthrough, health checks, CORS — minimal.

## Install

```bash
# Prerequisites: ROCm 7.x, Rust 1.88+
git clone https://github.com/bong-water-water-bong/1bit-engine
cd 1bit-engine
cargo build --release

# Run
./target/release/onebit --model path/to/model.h1b --port 13305
```

## Connect Apps

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:13305/v1", api_key="any")
print(client.chat.completions.create(
    model="bitnet",
    messages=[{"role":"user","content":"Say hello in one word."}],
    max_tokens=20,
).choices[0].message.content)
```

| App | Base URL |
|---|---|
| OpenAI SDK (Python, Node, Go) | `http://127.0.0.1:13305/v1` |
| Open WebUI, AnythingLLM, n8n, Dify | `http://127.0.0.1:13305/v1` |
| Continue.dev, Aider, Cline | `http://127.0.0.1:13305/v1` |

## Verified Benchmarks

On Strix Halo (Ryzen AI MAX+ 395, 128 GB unified), ROCm 7.2.4, via rocm-cpp:

| Model | Prompt tok/s | Gen tok/s |
|---|---|---|
| BitNet-2B-4T | — | **82** (bit-match vs PyTorch) |
| Bonsai 1.7B IQ1_S | 5,001 | 231 |
| Bonsai 4B IQ1_S | 2,125 | 126 |
| Bonsai 8B IQ1_S | 1,325 | 96 |

**Decode GEMV:** 7.6× faster than rocBLAS FP16 (27.6 µs vs ~700 µs)  
**Prefill GEMM (FFN up):** 21.8 TFlops our ternary vs 32.9 TFlops rocBLAS —  
  but at **1/4 the B memory bandwidth** = 2.6× effective throughput  
**Square prefill (4096³):** 20.2 TFlops (61% of rocBLAS at 1/4 memory)

## Repos

| Repo | Role |
|---|---|
| [1bit-engine](https://github.com/bong-water-water-bong/1bit-engine) | Rust HTTP server (the runtime) |
| [rocm-cpp](https://github.com/bong-water-water-bong/rocm-cpp) | C++/HIP kernels (the engine) |
| [1bit-systems](https://github.com/bong-water-water-bong/1bit-systems) | Website, docs, benchmarks (this repo) |

## License

MIT
