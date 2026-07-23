> ## ⚠️ ARCHIVED — historical document, does not reflect the current project
>
> This describes an **early, abandoned architecture** (a Rust HTTP server runtime).
> The current project is the opposite of what's described below: **pure C++,
> zero Rust at runtime, zero Python at runtime.** See [README.md](../README.md)
> and [CONTRIBUTING.md](../CONTRIBUTING.md) for the current architecture. Kept
> only for historical reference — do not copy claims or numbers from this file.

---

<div align="center">

<img src="../brand-lockup.svg" alt="1bit" width="540">

# Local 1-bit inference, wired for Strix Halo.

### Pure Rust. Zero Python. *(ARCHIVED — see banner above)*

**[→ Project Wiki](docs/wiki/README.md)** — architecture, decisions, and agent onboarding.

`1bit.systems` is a 1-bit inference engine for AMD Strix Halo (gfx1151). The
runtime is a Rust HTTP server that wraps [rocm-cpp](https://github.com/bong-water-water-bong/rocm-cpp)
HIP kernels, delivering 4.9–7.2× faster decode than rocBLAS FP16 at 1/4 the memory.

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.systems-12a0ed.svg)](https://1bit.systems)
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
# One command — handles Ubuntu, Arch, Fedora
curl -fsSL https://raw.githubusercontent.com/bong-water-water-bong/1bit-engine/main/install.sh | bash
```

The installer handles everything: ROCm build deps, Rust, rocm-cpp kernels, and the Rust server. Then download a .h1b model and run:

```bash
source ~/.cargo/env
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HSA_ENABLE_SDMA=0
~/1bit/engine/target/release/onebit --model model.h1b --port 13305 --tune-prefill --fp16-weights
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

## Verified Benchmarks — TheRock 7.15.0a, gfx1151, June 2026

### Prefill GEMM (our ternary 4h kernel vs rocBLAS FP16)

| Shape | rocm-cpp (TFlops) | rocBLAS (TFlops) | Ratio | B Memory |
|---|---|---|---|---|
| FFN up (2560×6912×2560) | **21.94** | 29.99 | 0.73× | **1/4** |
| FFN down (2560×2560×6912) | **20.91** | — | — | **1/4** |
| Square (4096×4096×4096) | **19.73** | 28.77 | 0.69× | **1/4** |

Effective throughput per byte: **2.9× rocBLAS**

### Decode GEMV (batch=1, memory-bound)

| Shape | rocm-cpp halo (µs) | rocBLAS FP16 (µs) | Speedup | B Memory |
|---|---|---|---|---|
| 2560×2560 | ~30 | 212 | **7.1×** | 1/16 |
| 4096×4096 | ~100 | 814 | **8.1×** | 1/16 |
| 6912×2560 (LM head) | **27.0** | ~700 | **7.8×** | 1/16 |
| 4096×11008 | ~200 | 1468 | **7.3×** | 1/16 |

**sherry** (3:4 N:M sparse, PolyForm NC): 18.7 µs = 1.45× halo  
**tq1** (PolyForm NC): 18.7 µs = 1.44× halo

### llama.cpp Q1_0 Full Burn (7 models)

| Model | Quant | Size | pp512 t/s | tg128 t/s |
|---|---|---|---|---|
| Bonsai-1.7B | Q1_0 | 231 MB | 5,001 | 231 |
| BitNet-2B-4T | Q1_0 | 538 MB | 3,652 | 120 |
| Bonsai-4B | Q1_0 | 540 MB | 2,125 | 126 |
| Bonsai-8B | Q1_0 | 1.07 GB | 1,325 | 96 |
| Qwen3-Coder-Next 80B | IQ1_S | 17.6 GB | 662 | 51 |
| Llama-4-Scout 17Bx16E | IQ1_S | 27.2 GB | 326 | 21 |
| BitNet-2B-4T | TQ1_0 | 1.02 GB | 282 | 50 |

Full data: [rocm-cpp results/BENCHMARK-20260623.md](https://github.com/bong-water-water-bong/rocm-cpp/blob/main/results/BENCHMARK-20260623.md)

## Repos

| Repo | Role |
|---|---|
| [1bit-engine](https://github.com/bong-water-water-bong/1bit-engine) | Rust HTTP server (the runtime) |
| [rocm-cpp](https://github.com/bong-water-water-bong/rocm-cpp) | C++/HIP kernels (the engine) |
| [1bit-systems](https://github.com/bong-water-water-bong/1bit-systems) | Website, docs, benchmarks (this repo) |

## License

MIT
