<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# One Binary to rule them all

### 1bit.systems · ~400 KB exe + ~1.1 MB kernel lib · NPU + GPU + CPU · Zero Python

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.systems-12a0ed.svg)](https://1bit.systems)
[![ROCm 7.15.0a](https://img.shields.io/badge/rocm-7.15.0a-f00fd2.svg)](https://rocm.docs.amd.com/en/7.13.0-preview/)
[![TheRock nightly](https://img.shields.io/badge/therock-nightly-00ffaa.svg)](https://rocm.nightlies.amd.com/whl-multi-arch/)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)
[![GPU Kernels](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/bong-water-water-bong/1bit-systems/main/site/badge_gpu.json)](site/benchmarks.json)
[![NPU Engine](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/bong-water-water-bong/1bit-systems/main/site/badge_npu.json)](site/benchmarks.json)
[![GGUF](https://img.shields.io/badge/GGUF-Qwen2%20%7C%20Qwen3%20layout-00ff00)](src/gguf_loader.cpp)
[![ONNX](https://img.shields.io/badge/ONNX%20weight%20extraction-ff9900)](src/onnx_loader.cpp)
[![Tests](https://img.shields.io/badge/tests-9%2F11-yellow)](tests/)  <!-- 2 e2e tests need model files (issue #233) -->

**One server binary (zaya_server) unifies GPU + CPU inference.**

The default NPU path delegates to **[FastFlowLM](https://github.com/amd/fastflowlm)** (external subprocess, runs standard Qwen3 Q4NX models, not 1-bit). The project's own NPU engine (`engine/npu/`) is a standalone C++23 INT8 engine that works on Strix Halo NPU but is not yet integrated into the unified server's cascade. See [`engine/npu/README.md`](engine/npu/) for details.

Auto-detects model architecture from the model header — no config files, no model registry.

Reverse-engineered AMD's XDNA 2 NPU in 4 days with no documentation. The project's in-process NPU engine (npu_engine_universal, XRT-based) is available for direct integration; the server's default NPU path uses the FastFlowLM subprocess for model dispatch and xclbin management. 1800+ hours of engineering across 28 layers of GEMM kernels, Vulkan flash attention, and a self-healing agent watchdog.

**[Read the full journey &rarr;](docs/journey.md)**

</div>

---

## Benchmarks

*Numbers auto-update from [`site/benchmarks.json`](site/benchmarks.json) on every push.*

| Benchmark | Value | Backend |
|-----------|:-----:|---------|
| Q1 GEMV † | **417 tok/s** | ROCm HIP (fused kernel) |
| Fused TQ2 † | **415 tok/s** | ROCm HIP (QKV+GU fused) |
| GPU ternary † | **318 tok/s** | Vulkan ZINC |
| TQ2 GEMV † | **355 tok/s** | ROCm HIP |
| NPU v12 | **97 tok/s** | XDNA 2 (32 tiles) |
| Prefill | **42.21 TFLOPS** | INT8 WMMA |
| ROCm HIP † | **64 tok/s** | ROCm HIP (kernels) |
| llama.cpp ROCm | **229 tok/s** | PrismML on same hardware |

> † **Kernel-level tok/s-equivalent** on a synthetic 28-layer weight buffer (excludes KV-cache attention, softmax, RoPE, FFN non-GEMM ops, sampler, tokenizer, and host↔device transfers). The llama.cpp ROCm row is *end-to-end decode of a real model* and is not comparable without adjustment. See [issue #235](https://github.com/bong-water-water-bong/1bit-systems/issues/235).

---

## Quick Start

**Note:** You need a model file (`.h1b` or `--manifest`). Without one, the server starts in no-weights mode and returns an error on chat requests.

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
# Download a model, e.g. from [1bit.systems/models](https://1bit.systems/models)
cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build --target zaya_server -j8
./build/zaya_server --model /path/to/model.h1b
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8088/v1", api_key="any")
print(client.chat.completions.create(model="zaya", messages=[{"role":"user","content":"Hello"}]).choices[0].message.content)
```

---

## Architecture

```
1bit/
  tests/zaya_server.cpp    ~400 KB exe + ~1.1 MB kernel lib
  src/                     HIP/C++ kernels (GEMV, prefill, attention)
  include/                 C API headers
  kernels/                 GPU kernels: bonsai, sherry, MoE
  engine/
    npu/                   C++17 INT8 engine (XDNA 2)
    gpu/                   Zig engine (Vulkan/CUDA/Metal)
  tools/                   Converters, benchmarks, training
  site/                    1bit.systems website
  packaging/               deb, snap, Docker
  benchmarks/              Historical data
```

### Loaders

- **GGUF** — Qwen2 / Qwen3 layout (header+embedding read; single transformer weight path; per-architecture attention/FFN not validated for Llama/Mistral/DeepSeek)
- **ONNX** — Protobuf wire format (F32/F16/BF16/INT8/INT32)
- **Q4NX** — FLM native format (311 tensors)
- **H1B** — Legacy ternary format

### Backends

- **NPU** — XDNA 2 (32 tiles). The server's default NPU path delegates to the **FastFlowLM** external subprocess (`/opt/fastflowlm/bin/flm`), which runs a standard Qwen3 model (not 1-bit). The project's in-process XRT-based engine (`npu_engine_universal`) is available for direct integration. See [issue #231](https://github.com/bong-water-water-bong/1bit-systems/issues/231).
- **GPU** — Radeon 8060S via Vulkan SPIR-V + ROCm HIP
- **CPU** — Fallback (scalar / AVX-512)

---

## License

MIT. Sherry-specific kernels: PolyForm Noncommercial 1.0.0.

---

<div align="center">
<a href="https://1bit.systems">Website</a> · <a href="site/blog/one-binary-to-rule-them-all.html">Blog</a> · <a href="site/demo/">Demo</a>
</div>
