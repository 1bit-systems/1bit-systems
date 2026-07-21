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
[![1BP](https://img.shields.io/badge/1BP-single%20file%2C%20zero%20config-00ffaa)](include/onebp_format.h)
[![Tests](https://img.shields.io/badge/tests-9%2F11-yellow)](tests/)  <!-- 2 e2e tests need model files (issue #233) -->

**One server binary (zaya_server) unifies NPU + GPU + CPU inference — no external subprocess, no proprietary runtime.**

### Why 1BP?

1BP is this project's native model format, designed to eliminate the config-file tax that every other format imposes:

- **256-byte header** — magic, version, quantization type (Q4NX/TQ2/F16/F32), model architecture enum, tokenizer config — **zero external config.json or tokenizer.json**
- **Memory-mappable weight data** — Q4NX-tiled arrays laid out exactly as the NPU DMA expects them, no load-time reshape or transpose. TQ2 ternary packs 2-bit codes at exactly half the size of Q4NX (2560 bytes/tile vs 5120)
- **Tensor index** — named tensors with native dims, byte offset, and size — no safetensors index file needed

The format exists because every model format the project ingests (GGUF, ONNX, safetensors) has a different indexing scheme, padding convention, and metadata layout. 1BP is the **normalization layer**: converters write 1BP once, the engine reads 1BP everywhere, and the translation cost is paid at conversion time rather than on every inference startup.

**Find pre-converted 1BP models at [1bit.systems/models](https://1bit.systems/models)** — Zamba2, ZR1, BlackMamba, and community-submitted conversions.

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

- **NPU** — XDNA 2 (32 tiles), fully in-process via `npu_engine_universal` (XRT-based, C++23). Runs GGUF/Q4NX/1BP models directly — no FastFlowLM subprocess, no closed-source dependency. Instruction sequences and GEMM/MHA dispatch were reverse-engineered from FLM's 22 `.so` libraries; xclbin bitstreams are rebuilt from AIE generators via `aiecc`/Chess (AMD Xilinx IP). See [`docs/fastflowlm-decode/SUMMARY.md`](docs/fastflowlm-decode/SUMMARY.md).
- **GPU** — Radeon 8060S via Vulkan SPIR-V + ROCm HIP
- **CPU** — Fallback (scalar / AVX-512)

---

## Model Coverage

Model-agnostic isn't just a claim about the loader — it's been exercised across genuinely different architectures: dense transformer (Qwen3, Llama, ZR1), mixture-of-experts (Zaya1-74B-A4B, Qwen 35B MoE), vision-language (Qwen2-VL), Mamba2-hybrid state-space (Zamba2), and genuinely ternary/1-bit-native weights (Bonsai, stored via 1BP's TQ2 quant, not just upsampled to 4-bit) — same engine, same auto-detect path, no per-architecture fork.

### Zaya1 — the flagship family

| Model | Params | Format | Performance | Status |
|-------|:------:|--------|-------------|:------:|
| **Zaya1-8B** | 8.84B | Q4NX / **1BP** | ~64 tok/s decode (GPU) | ✅ Primary — extensively tested, native 1BP support |
| Zaya1 Preview 74B-A4B (MoE) | 74.79B | Q4NX / **1BP** | 17.9 tok/s (iGPU, llama.cpp fork, 2026-07-03 — historical, no longer runs on current hardware) | ✅ 1BP conversion complete — [HF](https://huggingface.co/bong-water-water-bong/ZAYA1-74B-preview-1BP) |

Zaya1-8B is the model this project was built around: it's the one validated end-to-end through Q4NX, GGUF, and 1BP, and the one `tools/gguf_to_onebp.py` targets first when converting into the native format. Both sizes are published complete on Hugging Face — [**ZAYA1-8B-1BP**](https://huggingface.co/bong-water-water-bong/ZAYA1-8B-1BP) (1283 tensors, 16-expert MoE FFN weights) and [**ZAYA1-74B-preview-1BP**](https://huggingface.co/bong-water-water-bong/ZAYA1-74B-preview-1BP) (1923 tensors, 24-expert MoE FFN weights) — every tensor structurally verified against the source GGUF (exact parameter-count match) and numerically verified (dequantized values within expected 4-bit quantization tolerance).

### Zyphra family — beyond Zaya

Zaya1's maker, Zyphra, publishes several other architecturally distinct model lines. Converted the ones this engine can actually run end to end — dense transformer and Mamba2-hybrid — through the same 1BP pipeline:

| Model | Params | Architecture | Format | Status |
|-------|:------:|--------------|--------|:------:|
| [Zamba2-1.2B-Instruct-v2](https://huggingface.co/bong-water-water-bong/Zamba2-1.2B-Instruct-v2-1BP) | 1.2B | Mamba2-hybrid (attention every 6th layer) | **1BP** | ✅ |
| [Zamba2-2.7B-Instruct-v2](https://huggingface.co/bong-water-water-bong/Zamba2-2.7B-Instruct-v2-1BP) | 2.7B | Mamba2-hybrid | **1BP** | ✅ |
| [Zamba2-7B-Instruct-v2](https://huggingface.co/bong-water-water-bong/Zamba2-7B-Instruct-v2-1BP) | 7B | Mamba2-hybrid | **1BP** | ✅ |
| [ZR1-1.5B](https://huggingface.co/bong-water-water-bong/ZR1-1.5B-1BP) | 1.5B | Dense transformer (Qwen2 arch), reasoning-tuned | **1BP** | ✅ |

Each converted from a Q8_0/BF16 source (not a 4-bit GGUF) to avoid compounding quantization error through a second 4-bit pass, then structurally and numerically verified the same way as the Zaya1 conversions.

**Deliberately not converted**: BlackMamba-1.5B/2.8B (Zyphra's older pure-Mamba, no-attention line) — no GGUF exists for it anywhere and it predates the architecture support standard converters have, so producing one would mean writing an architecture-specific converter from scratch, not running existing tooling. ZAYA1-VL-8B — a real vision-language model; this converter only handles text weights, so a "conversion" would silently drop the vision tower and misrepresent it as the full model. Both skipped rather than shipped half-working.

### 🏆 Top 5 — Raw NPU Engine, No FLM (single binary, auto-detected)

*From [`engine/npu/BENCHMARKS.md`](engine/npu/BENCHMARKS.md), measured 2026-07-03/07-12 — predates the 2026-07-19 GGUF dequant correctness fixes (Q2_K/Q3_K/Q5_K, RoPE, dtype enums), so treat as directionally right pending re-measurement, not re-verified today.*

| Model | Family | Decode | Tok/s | Correctness |
|-------|--------|:------:|:-----:|:-----------:|
| Qwen3-0.6B | Qwen3 | 36 ms/tok | **28** | 28/28 ✅ |
| Gemma4-E2B | Gemma | 62 ms/tok | **16** | 35/35 ✅ |
| Qwen3-VL-4B | Qwen3 (vision) | 93 ms/tok | **11** | 36/36 ✅ |
| Llama-3.1-8B | Llama | 100 ms/tok | **10** | 32/32 ✅ |
| Qwen3-8B | Qwen3 | 127 ms/tok | **8** | 36/36 ✅ |

Same binary, same auto-detect path, no per-model glue — the loader reads architecture off the model header for all five.

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

Convert another ternary-native model the same way: `python3 tools/gguf_to_onebp.py model.gguf output.1bp --tq2`. `ONEBP_TQ1` (1.58-bit, base-3 packing) is still unimplemented — 256 isn't evenly divisible by its 5-values-per-byte scheme, so it needs more careful boundary handling than TQ2 did.

---

## FastFlowLM Decode

FastFlowLM (AMD's closed-source XDNA 2 inference engine) is fully reverse-engineered and replaced as of 2026-07-19.

| Component | Before (closed) | After (open) |
|-----------|:----------------:|:------------:|
| CLI + server | `flm`, 87.8 MB | Rebuilt, 17.5 MB |
| NPU sequence gen | 22 proprietary `.so` files | `libnpu_engine_universal.so` (173 KB) |
| FPGA bitstreams | 209 `.xclbin` files | 63 rebuilt from AIE generators |
| Toolchain | AMD Xilinx IP | `aiecc` + Chess/AMD Xilinx IP |

Build pipeline: Python AIE kernel generator → MLIR → `aiecc` + Chess → `.xclbin`.

The key finding: the `.so` files were NPU instruction **sequence generators**, not compute kernels — the actual computation lives entirely in the `.xclbin` FPGA bitstreams. Both layers are now fully rebuildable from source. Full writeup: [`docs/fastflowlm-decode/SUMMARY.md`](docs/fastflowlm-decode/SUMMARY.md) · reverse-engineering detail: [`fastflowlm_analysis/`](fastflowlm_analysis/).

---

## License

MIT. Sherry-specific kernels: PolyForm Noncommercial 1.0.0.

---

<div align="center">
<a href="https://1bit.systems">Website</a> · <a href="site/blog/one-binary-to-rule-them-all.html">Blog</a> · <a href="site/demo/">Demo</a>
</div>
