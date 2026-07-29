<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# One Binary to rule them all

### Pure C++23 inference engine · NPU + GPU + CPU in a single binary · Zero Python · Zero Rust · Zero config files

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.systems-12a0ed.svg)](https://1bit.systems)
[![ROCm](https://img.shields.io/badge/rocm-7.15.0a-f00fd2.svg)](https://github.com/bong-water-water-bong/TheRock)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Metal](https://img.shields.io/badge/Metal-Apple%20Silicon-ff9500.svg)](https://developer.apple.com/metal/)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)
[![GGUF](https://img.shields.io/badge/GGUF-Qwen2%20%7C%20Qwen3%20%7C%20Mamba-00ff00)](src/gguf_loader.cpp)
[![1BP](https://img.shields.io/badge/1BP-single%20file%2C%20zero%20config-00ffaa)](include/onebp_format.h)
[![Tests](https://img.shields.io/github/actions/workflow/status/bong-water-water-bong/1bit-systems/ci.yml?branch=main&label=tests)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)

**[🌐 Website](https://1bit.systems)** · **[🤗 1BP Models](https://huggingface.co/bong-water-water-bong)** · **[📚 Docs](docs/README.md)** · **[🛠️ Journey](docs/journey.md)** · **[📊 Benchmarks](docs/wiki/performance.md)** · **[🗺️ Roadmap](docs/guides/roadmap.md)**

**1bit** is an open-source, model-agnostic C++23 inference engine for running large language models on **AMD Strix Halo** (XDNA 2 NPU, RDNA 3.5 GPU), NVIDIA GPUs (CUDA), Apple Silicon (Metal), and any Vulkan 1.2+ device — all from a **single binary with zero Python at runtime**. It reads **GGUF**, **ONNX**, and the native **1BP** ternary format (TQ2 2-bit quantization) with automatic architecture detection — no config files, no model registry, no per-model glue code.

We reverse-engineered AMD's closed-source NPU stack (FastFlowLM) in 4 days — turning 22 proprietary `.so` files into a 207 KB open-source binary. We then extracted 37 pre-built FLM models with 209 NPU xclbins, and created our own 1BP format to transform AMD's open-source models into high-performance ternary binaries. Fully open-source under **MIT license**. 18 model architectures supported, 46+ 1BP models.

**Platform support:**
- **AMD Strix Halo** — XDNA 2 NPU + ROCm HIP GPU (79 tok/s BlackMamba 1.5B)
- **NVIDIA GPU** — CUDA backend (sm_70+)
- **Apple Silicon** — Metal GPU backend
- **Any Vulkan 1.2+ GPU** — ZINC engine
- **x86 CPU** — OpenMP fallback

**Key numbers:**
- 18 model architectures · 46+ 1BP models · 4 backends
- 433 tok/s peak kernel (Q1 GEMV, ROCm HIP)
- 79.4 tok/s end-to-end (BlackMamba 1.5B, Strix Halo)
- 37 FLM models extracted (209 NPU xclbins)

</div>

---

## Quick Start

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems && cmake -B build && cmake --build build
./build/zaya_server -m model.1bp -p "Hello world"
```

See the [Installation Guide](docs/wiki/Installation.md) for full instructions.

## Model Families

| Family | Type | Best Backend | Peak tok/s | Status |
|--------|------|-------------|-----------:|--------|
| Zaya1 | MoE + CCA attn | GPU HIP | 64 | ✅ |
| BlackMamba | Mamba1 + MoE | GPU HIP | 79.4 | ✅ |
| Zamba2 | Mamba2 hybrid | GPU Vulkan | ~30 | ✅ |
| Qwen2/Qwen3 | Dense / VL | GPU HIP / NPU | — | ✅ |
| Llama 3.1/3.2 | Dense | GPU HIP / NPU | — | ✅ |
| DeepSeek V2/V3/R1 | MoE + MLA | GPU HIP | — | ✅ |
| Mistral / Pixtral | Dense | GPU HIP | — | ✅ |
| Gemma 3/4 | Dense | NPU / GPU | — | ✅ |
| Phi4-Mini | Dense | NPU | — | ✅ |
| Bonsai (Deepgrove) | Ternary-native | GPU HIP | 21.9 | ✅ |
| Laguna | Dense | GPU HIP | — | ✅ |
| Falcon | Dense + MQA | GPU HIP | — | ✅ |
| OLMo | Dense (no RoPE) | GPU HIP | — | ✅ |
| ZR1 | Dense reasoning | GPU Vulkan | 26 | ✅ |
| Qwen2-VL / Qwen3-VL | Vision-Language | GPU HIP | — | ✅ |
| Whisper | Speech-to-text | NPU / GPU | — | ✅ |
| Nanbeige4.1 | Dense | NPU | — | ✅ |

**[→ Full model details and per-model benchmarks](docs/wiki/models.md)**

## Benchmarks

| Benchmark | tok/s | Backend | Status |
|-----------|:-----:|---------|--------|
| Q1 GEMV kernel | 433 | ROCm HIP | ✅ |
| Fused TQ2 kernel | 420 | ROCm HIP | ✅ |
| GPU ternary (Vulkan) | 318 | Vulkan ZINC | ✅ |
| BlackMamba 1.5B e2e | 79.4 | ROCm HIP | ✅ |
| BlackMamba 2.8B e2e | 46.0 | ROCm HIP | ✅ |

**[→ Full benchmarks](docs/wiki/performance.md)**

## Architecture

```
gguf / 1bp ──▶ [model loader: auto-detect 18 architectures]
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   NPU (XDNA 2)   GPU (ROCm)    GPU (Vulkan)
   32 tiles,       Radeon 8060S   Radeon 8060S
   50 TOPS
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                 CPU (OpenMP)
```

**[→ Architecture deep-dive](docs/guides/architecture.md)** · **[→ NPU reverse-engineering story](docs/journey.md)**

## Backends

| Backend | Hardware | What It Runs |
|---------|----------|-------------|
| NPU (npu_engine_universal) | XDNA 2, 32 tiles | INT8 GEMM, FLM models |
| GPU HIP (ROCm) | Radeon 8060S | Ternary GEMV, MoE, SSM |
| GPU Vulkan (ZINC) | Radeon 8060S | Dense models, 1BP |
| GPU CUDA | NVIDIA | Ternary kernels |
| GPU Metal | Apple Silicon | Ternary kernels |
| CPU (OpenMP) | x86 | Q4NX fallback |

## 📜 How We Got Here — Reverse Engineering the XDNA 2 NPU

This project started with a laptop, a disassembler, and no docs. AMD shipped the Ryzen AI Max+ 395 with a 50 TOPS XDNA 2 NPU locked behind a closed-source runtime (FastFlowLM) — 22 proprietary `.so` files, 209 xclbin bitstreams, zero documentation. **We reverse-engineered the entire stack in 4 days and replaced it with open-source code.**

| Component | Before (closed) | After (open) |
|-----------|:----------------:|:------------:|
| CLI + server | `flm`, 87.8 MB | Rebuilt, 17.5 MB |
| NPU sequence gen | 22 proprietary `.so` files | `libnpu_engine_universal.so` (173 KB) |
| FPGA bitstreams | 209 `.xclbin` files | 63 rebuilt from AIE generators |
| Toolchain | AMD Xilinx IP | `aiecc` + Chess/AMD Xilinx IP |

The key finding: the `.so` files were NPU instruction **sequence generators**, not compute kernels — the actual computation lives entirely in the `.xclbin` FPGA bitstreams. Both layers are now fully rebuildable from source.

> **Read the full 1800+ line journey** → [`docs/journey.md`](docs/journey.md) — every crash, breakthrough, and bug documented in real-time.
>
> **Technical reverse-engineering report** → [`docs/research/fastflowlm-decode/SUMMARY.md`](docs/research/fastflowlm-decode/SUMMARY.md)
>
> **Raw analysis** → [`docs/research/fastflowlm-analysis/`](docs/research/fastflowlm-analysis/) — binary analysis, xclbin captures, instruction traces

Since then: Mamba1 GPU backend (79.4 tok/s), Vulkan flash attention, model-agnostic GGUF routing, TQ2 ternary format, vision-language support, and a self-healing agent watchdog — **1800+ hours of engineering, all open source, MIT.**

---

## License

MIT — do whatever you want.

## Links

[Website](https://1bit.systems) · [Docs](docs/) · [Models](docs/wiki/models.md) · [Benchmarks](docs/wiki/performance.md) · [Journey](docs/journey.md) · [Roadmap](docs/guides/roadmap.md)
