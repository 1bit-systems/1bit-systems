<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# One Binary to rule them all

### Pure C++23 · one in-process engine, NPU + GPU + CPU in a single address space · thin server / CLI / daemon binaries share it · Zero Python · Zero Rust · Zero Node.js

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

**[🌐 Website](https://1bit.systems)** · **[🤗 1BP Models](https://huggingface.co/bong-water-water-bong)** · **[📚 Docs](docs/README.md)** · **[🛠️ Journey](docs/journey.md)** · **[📊 Benchmarks](docs/wiki/performance.md)** · **[🗺️ Roadmap](ROADMAP.md)**

**1bit** is an open-source, model-agnostic C++23 inference engine for running large language models on **AMD Strix Halo** (XDNA 2 NPU, RDNA 3.5 GPU), NVIDIA GPUs (CUDA), Apple Silicon (Metal), and any Vulkan 1.2+ device — all from a **single in-process binary with zero Python at runtime**. It reads **GGUF**, **ONNX**, and the native **1BP** ternary format (TQ2 2-bit quantization) with automatic architecture detection — no config files, no model registry, no per-model glue code. Fully open-source under **MIT license**. 17 model architectures supported, 40+ models.

---

**A single in-process engine unifies NPU + GPU + CPU inference — no external inference subprocess, no proprietary runtime. The `zaya_server`, `unified_server`, `onebit` CLI and `onebitd` daemon are thin front-ends over that one shared engine. C++23, zero Python at runtime.**

**Platform support:**
- **AMD Strix Halo** — XDNA 2 NPU + ROCm HIP GPU (79 tok/s BlackMamba 1.5B)
- **NVIDIA GPU** — CUDA backend (sm_70+, compute capability 7.0+)
- **Apple Silicon** — Metal GPU backend (M1/M2/M3/M4, macOS 14+)
- **Any Vulkan 1.2+** — Portable GPU backend
- **CPU** — Generic C++23 fallback (no GPU required)

> **Data note**: Per-backend tok/s figures are published with sources and honesty flags in [`docs/wiki/performance.md`](docs/wiki/performance.md). The NPU v12 engine currently measures **69 tok/s** (re-measured 2026-07-12 on Qwen3-0.6B). Mamba1 GPU numbers (79.4 tok/s) are current and re-validated 2026-07-26.

**17 architectures · 40+ models** — see [`models/catalog/README.md`](models/catalog/README.md) for the full list.

### 🚀 Flagship 1BP models — built, quantized & hosted by us

| Model | Family | Arch | Measured | Download |
|-------|--------|------|:--------:|:--------:|
| **Zaya1-8B** | Zyphra | MoE (16-expert) | **~64 tok/s** ✅ | [🤗 HF](https://huggingface.co/bong-water-water-bong/ZAYA1-8B-1BP) |
| **ZAYA1-VL-8B** | Zyphra | Vision-Language · ViT-L + MoE | **Vision + Text** 🆕 | [🤗 HF](https://huggingface.co/bong-water-water-bong/ZAYA1-8B-1BP) |
| **BlackMamba-1.5B** | Zyphra | Mamba1 · MoE | **79.4 tok/s** ✅ | [🤗 HF](https://huggingface.co/bong-water-water-bong/BlackMamba-1.5B-1BP) |
| **BlackMamba-2.8B** | Zyphra | Mamba1 · MoE | **46.0 tok/s** ✅ | [🤗 HF](https://huggingface.co/bong-water-water-bong/BlackMamba-2.8B-1BP) |
| **Zamba2-2.7B** | Zyphra | Mamba2-hybrid | **~30 tok/s** ✅ | [🤗 HF](https://huggingface.co/bong-water-water-bong/Zamba2-2.7B-Instruct-v2-1BP) |
| **ZR1-1.5B** | Zyphra | Dense · reasoning | **26 tok/s** (ZINC GPU) ✅ | [🤗 HF](https://huggingface.co/bong-water-water-bong/ZR1-1.5B-1BP) |
| **Bonsai-1.7B** | Deepgrove | Ternary TQ2 (2-bit) | 21.9 tok/s | [🤗 HF](https://huggingface.co/bong-water-water-bong/Bonsai-1.7B-TQ2-1BP) |

> **Zyphra is AMD's open-source AI lab** — the first model family trained end-to-end on AMD ROCm, no NVIDIA CUDA required. From training through inference, every Zyphra model runs on the AMD open ecosystem: ROCm for training, this engine for inference on Strix Halo (NPU + GPU + CPU).
>
> This engine supports the **complete Zyphra family** across every architecture — dense (ZR1), MoE (Zaya1), Mamba1 SSM (BlackMamba), Mamba2 hybrid (Zamba2), and now **vision-language (ZAYA1-VL-8B)** — all running on the same binary, zero Python, zero proprietary code. All models are converted with a pure-C++ toolchain, hosted on Hugging Face with open weights under permissive licenses. **[Browse all on Hugging Face →](https://huggingface.co/bong-water-water-bong)**

### Why 1BP?

1BP is this project's native model format, designed to eliminate the config-file tax that every other format imposes:

- **256-byte header** — magic, version, quantization type (Q4NX/TQ2/F16/F32), model architecture enum, tokenizer config — **zero external config.json or tokenizer.json**
- **Memory-mappable weight data** — Q4NX-tiled arrays laid out exactly as the NPU DMA expects them, no load-time reshape or transpose. TQ2 ternary packs 2-bit codes at exactly half the size of Q4NX (2560 bytes/tile vs 5120)
- **Tensor index** — named tensors with native dims, byte offset, and size — no safetensors index file needed

The format exists because every model format the project ingests (GGUF, ONNX, safetensors) has a different indexing scheme, padding convention, and metadata layout. 1BP is the **normalization layer**: converters write 1BP once, the engine reads 1BP everywhere, and the translation cost is paid at conversion time rather than on every inference startup.

**Find pre-converted 1BP models at [1bit.systems/models](https://1bit.systems/models)** — Zamba2, ZR1, BlackMamba, and community-submitted conversions.

**AMD shipped the NPU locked. We unlocked it in 4 days** — no docs, no NDAs, just a laptop and a disassembler. FastFlowLM, AMD's closed-source NPU inference engine, was fully reverse-engineered and replaced with a native open-source stack; the project's own NPU engine (`engine/npu/`, `npu_engine_universal`) now dispatches directly via XRT. Full numbers in [How We Got Here](#-how-we-got-here--reverse-engineering-the-xdna-2-npu) below. Then we kept going: Mamba1 GPU kernels (79.4 tok/s on BlackMamba), Vulkan flash attention, model-agnostic GGUF routing, and a self-healing agent watchdog — 1800+ hours of engineering across all of it. One binary, all backends, zero Python.

Model-agnostic end to end: the engine auto-detects architecture and quantization from the model header — no config files, no model registry, no per-model glue code. It reads **GGUF** and **ONNX** directly, speaks FastFlowLM's own **Q4NX** tiled layout natively, and ships **1BP** — this project's own single-file format (256-byte header + tensor index + memory-mappable Q4NX-tiled weights, zero external config.json).

**[Read the full journey &rarr;](docs/journey.md)** — 1800+ lines, every crash and breakthrough documented.

</div>

---

## Benchmarks

*Full breakdown, including NPU hardware validation and CPU fallback numbers: [`docs/wiki/performance.md`](docs/wiki/performance.md). Source of truth: [`site/benchmarks.json`](site/benchmarks.json).*

> ⚠️ **The kernel-level table below measures single-GEMM-kernel throughput on a synthetic 28-layer weight buffer.** These are "tok/s-equivalent" numbers — they exclude KV-cache attention, softmax, RoPE, FFN non-GEMM ops, sampler, tokenizer, and host↔device transfers. **Real end-to-end throughput is substantially lower** — see the second table. See [issue #235](https://github.com/bong-water-water-bong/1bit-systems/issues/235) for full discussion.

### 🧪 Kernel-Level Microbenchmarks (synthetic 28-layer weight buffer)

| Benchmark | Value | Backend |
|-----------|:-----:|---------|
| Q1 GEMV | **433 tok/s** | ROCm HIP (fused kernel) |
| Fused TQ2 | **420 tok/s** | ROCm HIP (QKV+GU fused) |
| GPU ternary | **318 tok/s** | Vulkan ZINC |
| TQ2 GEMV | **367 tok/s** | ROCm HIP |
| NPU v12 | **69 tok/s** | XDNA 2 (32 tiles) |
| Prefill | **39.4 TFLOPS** | INT8 WMMA |
| ROCm HIP | **64 tok/s** | ROCm HIP (kernels) |

### 🏁 End-to-End Inference (real model, real prompts)

| Benchmark | Value | Backend | Notes |
|-----------|:-----:|---------|-------|
| BlackMamba 1.5B | **79.4 tok/s** | Mamba1 HIP (Strix Halo) | Full decode, ROCm HIP, alternating SSM/MoE dispatch |
| BlackMamba 2.8B | **46.0 tok/s** | Mamba1 HIP (Strix Halo) | Full decode, ROCm HIP, alternating SSM/MoE dispatch |
| zaya_server (Qwen 27B Q4_K) | **30 tok/s** | ROCm HIP | Full decode, speculative MTP, Strix Halo |
| zaya_server (Qwen 35B MoE Q4_K) | **20 tok/s** | ROCm HIP | Full decode, speculative MTP, Strix Halo |
| llama.cpp ROCm (PrismML) | **229 tok/s** | PrismML on same hardware | See [issue #235](https://github.com/bong-water-water-bong/1bit-systems/issues/235) |

---

## Quick Start

### Try it now (GGUF model)

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
source env.sh
cmake -B build -G Ninja
cmake --build build --target onebit_bin unified_server zaya_server -j$(nproc)

# Interactive chat (REPL with NPU stack management):
./build/1bit chat

# Start the NPU stack (daemon + server):
./build/1bit up

# Or run with a GGUF model directly:
./build/unified_server -w /path/to/models/ -p 8088

# Mamba1 GPU backend:
./build/test_mamba1_backend /path/to/blackmamba-1.5b.gguf 64
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8088/v1", api_key="any")
print(client.chat.completions.create(model="blackmamba-1.5b",
      messages=[{"role":"user","content":"Hello"}]).choices[0].message.content)
```

---

## Architecture

```
1bit-systems/
  src/                     C++23 engine core: backend manager, HIP/CUDA/Metal kernels, loaders, vision encoder
  include/                 Public C API headers + vision_encoder.h (ViT + projector)
  kernels/                 GPU kernels: bonsai, sherry, MoE, Mamba1, fused QKV
  engine/
    gpu/zinc_cpp/          Vulkan SPIR-V GPU engine (ZINC)
    fusion/                Fused NPU+GPU dispatch, zero-copy paths
    npu/                   C++23 INT8 NPU engine (XDNA 2) — see build note below
  tools/                   CLI agent, daemon, converters, benchmarks:
    onebit.cpp             CLI agent (chat, up, down, status, build, config)
    onebitd.cpp            Daemon (spawns backend, proxies HTTP)
    unified_router.cpp     NPU+GPU routing proxy
    jarvis/                Voice agent — STT/TTS, C++ port
    gguf_to_onebp.cpp      GGUF → 1BP converter
  tests/                   Test/bench executables, ctest
  models/                  Model catalog + conversion manifests
  docs/                    Documentation (wiki/, archive/, marketing/)
  site/                    1bit.systems website
  packaging/               deb, snap, Docker, systemd services
  benchmarks/              Historical benchmark data
  scripts/                 Build & dev scripts
```

Build note: the root `CMakeLists.txt` builds `src/`, `kernels/`, `include/`, `tools/`, `tests/`, and `engine/gpu/zinc_cpp/` as one graph. `engine/npu/`, `npu-infer/`, and `spec-decode/` each ship their own `CMakeLists.txt` and build separately — not yet unified into the root build graph.

### Loaders

- **GGUF** — Qwen2 / Qwen3 layout (header+embedding read; single transformer weight path; per-architecture attention/FFN not validated for Llama/Mistral/DeepSeek)
- **ONNX** — Protobuf wire format (F32/F16/BF16/INT8/INT32)
- **Q4NX** — FastFlowLM's native tiled format, fully decoded (311 tensors, 4-bit groups of 32 with bf16 scales, 32×256 NPU tile layout) — see [`Q4NX_FORMAT.md`](docs/research/fastflowlm-analysis/Q4NX_FORMAT.md)
- **1BP** — this project's native format: single self-contained file, Q4NX-tiled weights, no external metadata. The `gguf_to_onebp` tool (pure C++, `tools/gguf_to_onebp.cpp`) converts any GGUF model in place — no Python.
- **H1B** — Legacy ternary format

### Backends

- **Mamba1 GPU** — Radeon 8060S via ROCm HIP. Alternating SSM + MoE layers (BlackMamba architecture). **79.4 tok/s** (1.5B).
- **NPU** — XDNA 2 (32 tiles), fully in-process via `npu_engine_universal` (XRT-based, C++23). Runs GGUF/Q4NX/1BP models directly — no FastFlowLM subprocess, no closed-source dependency. Instruction sequences and GEMM/MHA dispatch were reverse-engineered from FLM's 22 `.so` libraries; xclbin bitstreams are rebuilt from AIE generators via `aiecc`/Chess (AMD Xilinx IP). See [`docs/research/fastflowlm-decode/SUMMARY.md`](docs/research/fastflowlm-decode/SUMMARY.md).
- **GPU (ZINC)** — Radeon 8060S via Vulkan SPIR-V (GGUF/H1B models, multi-arch)
- **GPU (HIP)** — ROCm HIP for Zaya-style models
- **CPU** — Fallback (scalar / AVX-512 / generic GGUF)

---

## Model Coverage

Model-agnostic isn't just a claim about the loader — it's been exercised across genuinely different architectures: dense transformer (Qwen3, Llama, ZR1), mixture-of-experts (Zaya1-74B-A4B, Qwen 35B MoE), vision-language (Qwen2-VL), Mamba2-hybrid state-space (Zamba2), and genuinely ternary/1-bit-native weights (Bonsai, stored via 1BP's TQ2 quant, not just upsampled to 4-bit) — same engine, same auto-detect path, no per-architecture fork.

### Zaya1 — the flagship family

| Model | Params | Format | Performance | Status |
|-------|:------:|--------|-------------|:------:|
| **Zaya1-8B** | 8.84B | Q4NX / **1BP** | ~64 tok/s decode (GPU) | ✅ Primary — extensively tested, native 1BP support |
| Zaya1 Preview 74B-A4B (MoE) | 74.79B | Q4NX / **1BP** | 17.9 tok/s (iGPU, llama.cpp fork, 2026-07-03 — historical, no longer runs on current hardware) | ✅ 1BP conversion complete — [HF](https://huggingface.co/bong-water-water-bong/ZAYA1-74B-preview-1BP) |

Zaya1-8B is the model this project was built around: it's the one validated end-to-end through Q4NX, GGUF, and 1BP, and the one the `gguf_to_onebp` converter targets first when converting into the native format. Both sizes are published complete on Hugging Face — [**ZAYA1-8B-1BP**](https://huggingface.co/bong-water-water-bong/ZAYA1-8B-1BP) (1283 tensors, 16-expert MoE FFN weights) and [**ZAYA1-74B-preview-1BP**](https://huggingface.co/bong-water-water-bong/ZAYA1-74B-preview-1BP) (1923 tensors, 24-expert MoE FFN weights) — every tensor structurally verified against the source GGUF (exact parameter-count match) and numerically verified (dequantized values within expected 4-bit quantization tolerance).

### Zyphra family — beyond Zaya

Zaya1's maker, Zyphra, publishes several other architecturally distinct model lines. Converted the ones this engine can actually run end to end — dense transformer and Mamba2-hybrid — through the same 1BP pipeline:

| Model | Params | Architecture | Format | Status |
|-------|:------:|--------------|--------|:------:|
| [Zamba2-1.2B-Instruct-v2](https://huggingface.co/bong-water-water-bong/Zamba2-1.2B-Instruct-v2-1BP) | 1.2B | Mamba2-hybrid (attention every 6th layer) | **1BP** | ✅ |
| [Zamba2-2.7B-Instruct-v2](https://huggingface.co/bong-water-water-bong/Zamba2-2.7B-Instruct-v2-1BP) | 2.7B | Mamba2-hybrid | **1BP** | ✅ |
| [Zamba2-7B-Instruct-v2](https://huggingface.co/bong-water-water-bong/Zamba2-7B-Instruct-v2-1BP) | 7B | Mamba2-hybrid | **1BP** | ✅ |
| [ZR1-1.5B](https://huggingface.co/bong-water-water-bong/ZR1-1.5B-1BP) | 1.5B | Dense transformer (Qwen2 arch), reasoning-tuned | **1BP** | ✅ **26 tok/s (ZINC GPU)** |
| [BlackMamba-1.5B](https://huggingface.co/bong-water-water-bong/BlackMamba-1.5B-1BP) | 1.5B | Mamba1 + top-1 MoE (no attention at all) | **1BP** | ✅ **79.4 tok/s** |
| [BlackMamba-2.8B](https://huggingface.co/bong-water-water-bong/BlackMamba-2.8B-1BP) | 2.8B | Mamba1 + top-1 MoE | **1BP** | ✅ **46.0 tok/s** |

Each converted from a Q8_0/BF16 source (not a 4-bit GGUF) to avoid compounding quantization error through a second 4-bit pass, then structurally and numerically verified the same way as the Zaya1 conversions.

> **On-device validation — Strix Halo (Radeon 8060S, gfx1151):** **BlackMamba-1.5B 79.4 tok/s** / **2.8B 46.0 tok/s** on the Mamba1 HIP backend (re-validated 2026-07-26, 3-run average). **Dense GPU inference is live**: **ZR1-1.5B (Qwen2) runs on the native C++ ZINC Vulkan backend at ~26 tok/s** and matches the CPU reference **token-for-token** ([#844](https://github.com/bong-water-water-bong/1bit-systems/issues/844) — closed). ZINC is enabled by default for the architectures it computes correctly (llama/mistral/qwen2) and falls back to the exact `cpu_generic` path otherwise; `ZINC_DISABLE=1` forces HIP/CPU. The engine is also **crash-hardened** — a backend that fails to initialize fails over to HIP/CPU instead of taking the server down.

**BlackMamba required a from-scratch converter** — no upstream GGUF export exists for this architecture, and it predates the architecture support standard converters have. The one-time bootstrap conversion shipped with three real correctness bugs on the first pass (wrong Q4_0 nibble encoding, a conv1d weight reshape that silently scrambled channel/kernel-tap pairing, and a dropped MoE router bias), all found and fixed by cross-checking against the in-tree C++ reference (`tools/blackmamba_cpu_reference.cpp`) and the official Zyphra implementation — see the model cards on Hugging Face for the full writeup. The resulting weights are what `gguf_to_onebp` now ingests directly.

**Fast inference is now wired**: `src/mamba1_engine.hip` kernels are compiled into `librocm_cpp.so` and the `Mamba1Backend` (HIP GPU) is registered as a first-class backend in `BackendManager`. Both BlackMamba sizes load end-to-end through the Mamba1 GPU backend: alternating SSM layers (rmsnorm → in_proj → conv1d/silu → selective_scan → gate → out_proj) and MoE FFN layers (router → top-1 expert dispatch → SiLU → scale-add residual). The diagnostic tool `tools/test_mamba1_backend.cpp` loads a Mamba1 GGUF directly into the HIP backend for testing without the HTTP server. PR [#579](https://github.com/bong-water-water-bong/1bit-systems/pull/579) shipped the build linkage, conv state fix, and A_log exponentiation fix.

**Vision-language is now supported**: ZAYA1-VL-8B — a real vision-language model combining a SigLIP ViT vision encoder with the Zaya1-8B MoE text decoder. The vision tower (24-layer ViT, fused QKV, Q8_0 quant) and connector (QWEN2_MERGER-style projector) are handled by the new `vision_encoder` library — pure C++, no Python, no OpenCV. The converter now preserves vision weights in 1BP files with full tensor metadata. See [`include/vision_encoder.h`](include/vision_encoder.h) and [`tools/zaya1_vl_demo.cpp`](tools/zaya1_vl_demo.cpp).

### ⚠️ Data Clarity

> **Current (re-validated 2026-07-26):** BlackMamba-1.5B 79.4 tok/s, BlackMamba-2.8B 46.0 tok/s, ZR1-1.5B 26 tok/s (ZINC GPU) — all verified end-to-end. See model cards above.
>
> **Directional (needs re-measure):** NPU numbers below were measured before the 2026-07-19 GGUF dequant correctness fixes. They're directionally right but not re-verified. See [`docs/wiki/performance.md`](docs/wiki/performance.md).
>
> **Disproven:** 572 tok/s DSpark speculative decoding — the checkpoint was undertrained. See [issue discussion](https://github.com/bong-water-water-bong/1bit-systems/issues/235).

### 🏆 Top 5 — Raw NPU Engine, No FLM (single binary, auto-detected)

*From [`engine/npu/BENCHMARKS.md`](engine/npu/BENCHMARKS.md), measured 2026-07-03/07-12 — predates the 2026-07-19 GGUF dequant correctness fixes (Q2_K/Q3_K/Q5_K, RoPE, dtype enums), so treat as directionally right pending re-measurement, not re-verified today.*

| Model | Family | Decode | Tok/s | Correctness |
|-------|--------|:------:|:-----:|:-----------:|
| Qwen3-0.6B | Qwen3 | 36 ms/tok | **28** | 28/28 ✅ |
| Gemma4-E2B | Gemma | 62 ms/tok | **16** | 35/35 ✅ |
| Qwen3-VL-4B | Qwen3 (vision) | 93 ms/tok | **11** | 36/36 ✅ |
| Llama-3.1-8B | Llama | 100 ms/tok | **10** | 32/32 ✅ |
| Qwen3-8B | Qwen3 | 127 ms/tok | **8** | 36/36 ✅ |

Same binary, same auto-detect path, no per-model glue — the loader reads architecture off the model header for all 40 models.

### Also validated

| Model | Architecture | Backend | Note |
|-------|--------------|---------|------|
| Bonsai-1.7B | Ternary (IQ1_S mixed quant) | Vulkan/ZINC | 21.6-21.9 tok/s, 99.6% of theoretical memory bandwidth |
| Zamba2 (1.2B / 2.7B / 7B) | Mamba2 SSD hybrid | ROCm (fallback) | Mamba2 lacks tuned ROCm kernels — PyTorch fallback, ~73× slower than attention models; see [`models/catalog/README.md`](models/catalog/README.md) |
| Qwen2-VL | Vision-language | GPU | Minimal POC — real image-to-text, stops at EOS |

### TQ2 — the actual 1-bit/ternary storage path

Every model above is stored via 1BP's default Q4NX quant (4-bit, works for any source precision). `ONEBP_TQ1`/`ONEBP_TQ2` have been defined in the format since it was designed but were never implemented — meaning even genuinely ternary-trained models were getting upsampled to 4-bit on the way in. Fixed for TQ2: symmetric 2-bit quantization (every value is exactly `-scale`, `0`, or `+scale`, one BF16 scale per 32-group, no zero-point needed), exactly half of Q4NX's tile size.

| Model | Params | Format | Verification | HF |
|-------|:------:|--------|---------------|-----|
| [Bonsai-1.7B](https://huggingface.co/bong-water-water-bong/Bonsai-1.7B-TQ2-1BP) | 1.72B | **1BP (TQ2)** | 100% of dequantized values match source within BF16 scale-rounding (mean rel. error rounds to 0.000000) — lossless repack, not requantization | [link](https://huggingface.co/bong-water-water-bong/Bonsai-1.7B-TQ2-1BP) |

Convert another ternary-native model the same way: `./build/gguf_to_onebp model.gguf output.1bp --tq2` (pure C++, no Python). `ONEBP_TQ1` (1.58-bit, base-3 packing) is still unimplemented — 256 isn't evenly divisible by its 5-values-per-byte scheme, so it needs more careful boundary handling than TQ2 did.

---

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

MIT. Sherry-specific kernels: PolyForm Noncommercial 1.0.0.

---

<div align="center">
<a href="https://1bit.systems">Website</a> · <a href="site/blog/one-binary-to-rule-them-all.html">Blog</a> · <a href="site/demo/">Demo</a>
<br><br>
<a href="https://github.com/bong-water-water-bong/1bit-systems">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/github/stars/bong-water-water-bong/1bit-systems?style=social&label=Star">
    <img src="https://img.shields.io/github/stars/bong-water-water-bong/1bit-systems?style=social&label=Star" alt="Star on GitHub">
  </picture>
</a>
<br>
<i>If this project saved you time or inspired you, <b>star the repo</b> — it tells GitHub this matters, and helps others find it.</i>
<br><br>

[![Star History Chart](https://api.star-history.com/svg?repos=bong-water-water-bong/1bit-systems&type=Date)](https://star-history.com/#bong-water-water-bong/1bit-systems&Date)

</div>