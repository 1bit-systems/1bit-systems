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
[![Q4NX](https://img.shields.io/badge/Q4NX-fully%20decoded-00ff00)](docs/fastflowlm-decode/SUMMARY.md)
[![1BP](https://img.shields.io/badge/1BP-native%20format-00ffaa)](include/onebp_format.h)
[![Tests](https://img.shields.io/badge/tests-9%2F11-yellow)](tests/)  <!-- 2 e2e tests need model files (issue #233) -->

**One server binary (zaya_server) unifies NPU + GPU + CPU inference — no external subprocess, no proprietary runtime.**

FastFlowLM, AMD's closed-source NPU inference engine, has been fully reverse-engineered and replaced: all 22 proprietary `.so` libraries disassembled, all 209 xclbin bitstreams traced back to their AIE generators, and the whole stack rebuilt from source (87.8MB closed binary → 17.5MB open one). The project's own NPU engine (`engine/npu/`, `npu_engine_universal`) now dispatches directly via XRT — see [`docs/fastflowlm-decode/SUMMARY.md`](docs/fastflowlm-decode/SUMMARY.md) for the full decode report.

Model-agnostic end to end: the engine auto-detects architecture and quantization from the model header — no config files, no model registry, no per-model glue code. It reads **GGUF** and **ONNX** directly, speaks FastFlowLM's own **Q4NX** tiled layout natively, and ships **1BP** — this project's own single-file format (256-byte header + tensor index + memory-mappable Q4NX-tiled weights, zero external config.json).

Reverse-engineered AMD's XDNA 2 NPU in 4 days with no documentation. 1800+ hours of engineering across 28 layers of GEMM kernels, Vulkan flash attention, and a self-healing agent watchdog.

**[Read the full journey &rarr;](docs/journey.md)**

</div>

---

## Benchmarks

*Numbers auto-update from [`site/benchmarks.json`](site/benchmarks.json) on every push.*

> ⚠️ **The table below mixes kernel-level synthetic microbenchmarks with end-to-end inference.** Rows in the first table are kernel-level only — they exclude KV-cache attention, softmax, RoPE, FFN non-GEMM ops, sampler, tokenizer, and host↔device transfers. **Real end-to-end throughput is substantially lower** — see the second table. See [issue #235](https://github.com/bong-water-water-bong/1bit-systems/issues/235) for discussion.

### 🧪 Kernel-Level Microbenchmarks (synthetic 28-layer weight buffer)

| Benchmark | Value | Backend |
|-----------|:-----:|---------|
| Q1 GEMV | **417 tok/s** | ROCm HIP (fused kernel) |
| Fused TQ2 | **415 tok/s** | ROCm HIP (QKV+GU fused) |
| GPU ternary | **318 tok/s** | Vulkan ZINC |
| TQ2 GEMV | **355 tok/s** | ROCm HIP |
| NPU v12 | **97 tok/s** | XDNA 2 (32 tiles) |
| Prefill | **42.21 TFLOPS** | INT8 WMMA |
| ROCm HIP | **64 tok/s** | ROCm HIP (kernels) |

### 🏁 End-to-End Inference (real model, real prompts)

| Benchmark | Value | Backend | Notes |
|-----------|:-----:|---------|-------|
| zaya_server (Qwen 27B Q4_K) | **30 tok/s** | ROCm HIP | Full decode, speculative MTP, Strix Halo |
| zaya_server (Qwen 35B MoE Q4_K) | **20 tok/s** | ROCm HIP | Full decode, speculative MTP, Strix Halo |
| llama.cpp ROCm (PrismML) | **229 tok/s** | PrismML on same hardware | See [issue #235](https://github.com/bong-water-water-bong/1bit-systems/issues/235) |

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
    npu/                   C++23 INT8 engine (XDNA 2)
    gpu/                   Zig engine (Vulkan/CUDA/Metal)
  tools/                   Converters, benchmarks, training
  site/                    1bit.systems website
  packaging/               deb, snap, Docker
  benchmarks/              Historical data
```

### Loaders

- **GGUF** — Qwen2 / Qwen3 layout (header+embedding read; single transformer weight path; per-architecture attention/FFN not validated for Llama/Mistral/DeepSeek)
- **ONNX** — Protobuf wire format (F32/F16/BF16/INT8/INT32)
- **Q4NX** — FastFlowLM's native tiled format, fully decoded (311 tensors, 4-bit groups of 32 with bf16 scales, 32×256 NPU tile layout) — see [`Q4NX_FORMAT.md`](fastflowlm_analysis/Q4NX_FORMAT.md)
- **1BP** — this project's native format: single self-contained file, Q4NX-tiled weights, no external metadata. `tools/gguf_to_onebp.py` converts any GGUF model in place.
- **H1B** — Legacy ternary format

### Backends

- **NPU** — XDNA 2 (32 tiles), fully in-process via `npu_engine_universal` (XRT-based, C++23). Runs GGUF/Q4NX/1BP models directly — no FastFlowLM subprocess, no closed-source dependency. Instruction sequences and GEMM/MHA dispatch were reverse-engineered from FLM's 22 `.so` libraries; xclbin bitstreams are rebuilt from AIE generators via `aiecc`/Peano. See [`docs/fastflowlm-decode/SUMMARY.md`](docs/fastflowlm-decode/SUMMARY.md).
- **GPU** — Radeon 8060S via Vulkan SPIR-V + ROCm HIP
- **CPU** — Fallback (scalar / AVX-512)

---

## FastFlowLM Decode

FastFlowLM (AMD's closed-source XDNA 2 inference engine) is fully reverse-engineered and replaced as of 2026-07-19.

| Component | Before (closed) | After (open) |
|-----------|:----------------:|:------------:|
| CLI + server | `flm`, 87.8 MB | Rebuilt, 17.5 MB |
| NPU sequence gen | 22 proprietary `.so` files | `libnpu_engine_universal.so` (173 KB) |
| FPGA bitstreams | 209 `.xclbin` files | 63 rebuilt from AIE generators |
| Toolchain | AMD Xilinx IP | `aiecc` + Peano/LLVM-AIE |

Build pipeline: Python AIE kernel generator → MLIR → `aiecc` + Peano → `.xclbin`.

The key finding: the `.so` files were NPU instruction **sequence generators**, not compute kernels — the actual computation lives entirely in the `.xclbin` FPGA bitstreams. Both layers are now fully rebuildable from source. Full writeup: [`docs/fastflowlm-decode/SUMMARY.md`](docs/fastflowlm-decode/SUMMARY.md) · reverse-engineering detail: [`fastflowlm_analysis/`](fastflowlm_analysis/).

---

## License

MIT. Sherry-specific kernels: PolyForm Noncommercial 1.0.0.

---

<div align="center">
<a href="https://1bit.systems">Website</a> · <a href="site/blog/one-binary-to-rule-them-all.html">Blog</a> · <a href="site/demo/">Demo</a>
</div>
