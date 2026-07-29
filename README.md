# 1bit.systems — One Binary. All Backends.

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![ROCm](https://img.shields.io/badge/rocm-7.15.0a-f00fd2.svg)](https://github.com/bong-water-water-bong/TheRock)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Metal](https://img.shields.io/badge/Metal-Apple%20Silicon-ff9500.svg)](https://developer.apple.com/metal/)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)
[![GGUF](https://img.shields.io/badge/GGUF-Qwen2%20%7C%20Qwen3%20%7C%20Mamba-00ff00)](src/gguf_loader.cpp)
[![1BP](https://img.shields.io/badge/1BP-single%20file%2C%20zero%20config-00ffaa)](include/onebp_format.h)

## What is this?

A single C++23 binary (~207 KB) that runs LLMs and VLMs on AMD NPU (XDNA 2),
ROCm/CUDA/Metal/Vulkan GPU, and CPU — zero Python, zero config files.
Auto-detects 18 model architectures from GGUF/1BP headers.

We reverse-engineered AMD's closed-source NPU stack (FastFlowLM), extracted 37
pre-built models with 209 NPU xclbins, and created our own 1BP ternary weight
format to transform AMD's open-source models into high-performance binaries.
MIT licensed.

**Key numbers:**
- 18 model architectures · 46+ 1BP models · 4 backends
- 433 tok/s peak kernel (Q1 GEMV, ROCm HIP)
- 79.4 tok/s end-to-end (BlackMamba 1.5B, Strix Halo)
- 37 FLM models extracted (209 NPU xclbins)

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

## License

MIT — do whatever you want.

## Links

[Website](https://1bit.systems) · [Docs](docs/) · [Models](docs/wiki/models.md) · [Benchmarks](docs/wiki/performance.md) · [Journey](docs/journey.md) · [Roadmap](docs/guides/roadmap.md)
