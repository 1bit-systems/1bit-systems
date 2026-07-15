<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# 1bit.systems

### 398 KB single binary · NPU + GPU + CPU · Zero Python

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.systems-12a0ed.svg)](https://1bit.systems)
[![ROCm 7.2.4](https://img.shields.io/badge/rocm-7.2.4-blue.svg)](https://rocm.docs.amd.com)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)
[![GPU Kernels](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/bong-water-water-bong/1bit-systems/main/site/badge_gpu.json)](site/benchmarks.json)
[![NPU Engine](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/bong-water-water-bong/1bit-systems/main/site/badge_npu.json)](site/benchmarks.json)
[![GGUF](https://img.shields.io/badge/GGUF-Qwen2%20%7C%20Llama%20%7C%20Mistral-00ff00)](src/gguf_loader.cpp)
[![ONNX](https://img.shields.io/badge/ONNX-supported-00ff00)](src/onnx_loader.cpp)
[![Tests](https://img.shields.io/badge/tests-11%2F11-00ff00)](tests/)

**One binary, all backends, all models.** Auto-detects model architecture from the model header — no config files, no model registry.

</div>

---

## Benchmarks

*Numbers auto-update from [`site/benchmarks.json`](site/benchmarks.json) on every push.*

| Benchmark | tok/s | Backend |
|-----------|:-----:|---------|
| Q1 GEMV | **417** | ROCm HIP (fused kernel) |
| Fused TQ2 | **415** | ROCm HIP (QKV+GU fused) |
| GPU ternary | **318** | Vulkan ZINC |
| TQ2 GEMV | **355** | ROCm HIP |
| NPU v12 | **69** | XDNA 2 (32 tiles) |
| Prefill | **42.2 TFLOPS** | INT8 WMMA |
| ROCm HIP | **64** | ROCm HIP (kernels) |
| llama.cpp ROCm | **229** | PrismML on same hardware |

---

## Quick Start

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build --target zaya_server -j8
./build/zaya_server
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
  tests/zaya_server.cpp    398 KB binary
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

- **GGUF** — Qwen2, Llama, Mistral, DeepSeek (F32/F16/Q8_0/Q4_0/Q5_1/Q5_K/Q8_K)
- **ONNX** — Protobuf wire format (F32/F16/BF16/INT8/INT32)
- **Q4NX** — FLM native format (311 tensors)
- **H1B** — Legacy ternary format

### Backends

- **NPU** — XDNA 2 (32 tiles) via subprocess protocol
- **GPU** — Radeon 8060S via Vulkan SPIR-V + ROCm HIP
- **CPU** — Fallback (scalar / AVX-512)

---

## License

MIT. Sherry-specific kernels: PolyForm Noncommercial 1.0.0.

---

<div align="center">
<a href="https://1bit.systems">Website</a> · <a href="site/blog/one-binary-to-rule-them-all.html">Blog</a> · <a href="site/demo/">Demo</a>
</div>
