<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# One Binary to rule them all.

### Pure C++ inference server · 207 KB · No Python · No Rust

Single binary that runs every model on every backend — NPU fused, GPU ternary, ROCm HIP, Vulkan, CPU. Auto-detects architecture from the model header. Zero configuration files.

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.systems-12a0ed.svg)](https://1bit.systems)
[![ROCm 7.2.4](https://img.shields.io/badge/rocm-7.2.4-blue.svg)](https://rocm.docs.amd.com)
[![Binary](https://img.shields.io/badge/binary-207%20KB-f00fd2.svg)](tests/zaya_server.cpp)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)
[![NPU fused](https://img.shields.io/badge/npu%20fused-291%20tok%2Fs-00ff00.svg)](https://github.com/bong-water-water-bong/1bit-systems)

</div>

---

## Hardware-Validated Performance

Measured on **AMD Strix Halo** (Ryzen AI Max+ 395) — 32 XDNA 2 NPU tiles + Radeon 8060S (gfx1151) GPU, 128 GB unified LPDDR5X memory.

| Engine | Backend | Hardware | tok/s | Status |
|--------|---------|----------|:-----:|--------|
| **GPU 1-bit** (llama.cpp ROCm) 🏆 | ROCm HIP | Radeon 8060S | **383** | ✅ measured |
| **NPU fused** | XDNA 2 xclbin | XDNA 2 · 32 tiles | **291** | ✅ coherent |
| **GPU ternary** (Vulkan) | Vulkan GLSL | Radeon 8060S | **307** | ✅ coherent |
| **NPU FLM** (production) | XDNA 2 xclbin | XDNA 2 · 32 tiles | **94** | ✅ validated |
| **C++ all-5** (auto-detect) | Q4NX header parse | XDNA 2 · 32 tiles | **28** | ⚙️ raw |
| **GPU ZINC** (Vulkan) | Vulkan GLSL | Radeon 8060S | **22** | ✅ coherent |
| **GPU Zaya** (ROCm HIP) | HIP kernels | Radeon 8060S | **10.6** | ✅ pure C++ server |

See [full benchmark data](benchmarks/RESULTS-stack-2026-04-28.md) and the [live benchmarks](https://1bit.systems/live.html) page.

---

## The Token Router

The **token router** is the intelligence layer that dispatches every token to the highest-performance backend available, in priority order:

```
┌─ Request ─────────────────────────────────────────────┐
│  POST /v1/chat/completions {"model":"zaya", ...}      │
└──────────────────────┬────────────────────────────────┘
                       ▼
┌─ Token Router (9.7 MB) ───────────────────────────────┐
│                                                        │
│  1. NPU fused  ──► XDNA 2 xclbin   ──► 291 tok/s     │
│                    (32 tiles, INT8)                    │
│                                                        │
│  2. GPU ternary ──► Vulkan compute  ──► 307 tok/s     │
│                    (1.58-bit, GLSL)                    │
│                                                        │
│  3. ROCm HIP   ──► HIP kernels      ──► 113 tok/s     │
│                    (ternary GEMV/GEMM)                 │
│                                                        │
│  4. CPU        ──► OpenMP fallback  ──► ~5 tok/s      │
│                    (Q4NX, any x86)                     │
└──────────────────────┬────────────────────────────────┘
                       ▼
┌─ Response ────────────────────────────────────────────┐
│  {"choices":[{"text":"Hello! ..."}]}                   │
└────────────────────────────────────────────────────────┘
```

The router profiles each backend at startup, then selects the fastest path per-layer. When NPU dispatch overhead outweighs the benefit (e.g. attention for contexts < 128 tokens), it gracefully falls back to the GPU or CPU. No configuration files. No manual tuning.

---

## Architecture

```
1bit/
├── tests/                  Pure C++ inference server
│   └── zaya_server.cpp    ← THE ONE BINARY — 207 KB
├── src/                    HIP C++ kernels → ternary GEMV/GEMM
├── include/                C API headers
├── kernels/                HIP implementations → zaya, bonsai, sherry, kv-cache
├── engine/
│   ├── npu/                C++23 INT8 engine — NPU (XDNA 2)
│   └── gpu/                Zig engine — GPU (Vulkan/CUDA/Metal)
├── site/                   1bit.systems website
│   ├── assets/             Brand assets, favicon
│   ├── blog/               Engineering blog
│   └── docs/               Architecture & deployment docs
├── tools/                  Model converters, benchmarks, video-lora
├── docs/                   Architecture, build guide, roadmap
├── packaging/              deb, snap, tarball, Docker
└── benchmarks/             Historical benchmark data
```

**One repo, one build, one binary.** `cmake` builds the kernels and the server. No language runtimes required.

---

## Quick Start

### Build zaya_server

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build --target zaya_server -j8
```

### Run

```bash
./build/zaya_server                           # Starts on port 8088
# Optional flags:
#   --port 13305                              # Custom port
#   --model /path/to/model.h1b                # Model path
#   --tune-prefill                            # Enable prefill autotuning
#   --fp16-weights                            # Use FP16 weight cache
```

### OpenAI-compatible API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8088/v1",
    api_key="any"  # required by spec, not checked
)

response = client.chat.completions.create(
    model="zaya",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=20,
)
print(response.choices[0].message.content)
```

Any OpenAI-compatible client works — Open WebUI, AnythingLLM, Continue, Aider, Cline, n8n, Dify.

---

## How It Works

### One Binary, All Models

`zaya_server.cpp` (19 KB of source, 207 KB compiled binary) reads the **Q4NX header** of any supported model at startup, auto-detects architecture dimensions (layers, heads, hidden size), allocates the correct buffers, and dispatches to the right backend. No config files. No model registry.

> **73+ models. 6 backends. One 207 KB binary.** The header knows what it is. The binary figures out the rest.
>
> — [Read the full blog post →](site/blog/one-binary-to-rule-them-all.html)

### No Rust at Runtime

The server is **pure C++** (zaya_server.cpp, compiled via CMake with HIP). Zero Rust dependencies. Zero Rust runtime. Zero Rust build tools required.

### No Python at Runtime

Zero Python dependencies at inference time. No pip, no conda, no virtualenv. The binary links against HIP and libc — nothing else. (Python is used only in `tools/` for model conversion and benchmarking.)

### Verified Model Families

| Model | Size | Hidden | Layers | NPU speed |
|-------|------|--------|:------:|:---------:|
| Qwen3-0.6B | 0.6B | 1,536 | 28 | **291 tok/s** |
| Qwen3-8B | 8B | 4,096 | 32 | 215 ms/tok |
| Qwen3-VL-4B | 4B | 2,560 | 36 | 141 ms/tok |
| Llama-3.1-8B | 8B | 4,096 | 32 | 185 ms/tok |
| Gemma4-E2B | 2B | 2,304 | 26 | 117 ms/tok |

### 6 Backends

| Backend | API | Speed (0.6B) |
|---------|-----|:------------:|
| NPU fused | XDNA 2 xclbin | **291 tok/s** |
| NPU v12 (FLM) | XDNA 2 xclbin | 97 tok/s |
| ROCm HIP | HIP kernels | 113 tok/s |
| Vulkan | GLSL compute | 22 tok/s |
| CUDA | llama.cpp | ~30 tok/s |
| Metal | llama.cpp | ~15 tok/s |

---

## Documentation

| Resource | Description |
|----------|-------------|
| [1bit.systems](https://1bit.systems) | Full website with benchmarks, install, and blog |
| [One Binary Blog Post](site/blog/one-binary-to-rule-them-all.html) | How auto-detection works |
| [Kernel Architecture](site/docs/architecture.html) | RDNA 3.5, ternary theory, packing |
| [Build Guide](docs/) | Full build instructions |
| [Benchmark Data](benchmarks/) | Historical benchmark reports |
| [Packaging](packaging/) | deb, snap, tarball, Docker |

---

## License

MIT. See `LICENSE`.

Sherry-specific kernels (3:4 N:M sparse ternary GEMV, TQ1 packer): PolyForm Noncommercial 1.0.0. See `LICENSE-SHERRY.md`.
