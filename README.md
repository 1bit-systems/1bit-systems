<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# 1bit.systems

### 398 KB single binary · NPU + GPU + CPU · Zero Python

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.systems-12a0ed.svg)](https://1bit.systems)
[![ROCm 7.2.4](https://img.shields.io/badge/rocm-7.2.4-blue.svg)](https://rocm.docs.amd.com)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)

**One binary, all backends, all models.** Auto-detects model architecture from the `.h1b` header — no config files, no model registry.

</div>

---

## Benchmarks

Measured **2026-07-15** on **AMD Strix Halo** (Ryzen AI Max+ 395, Radeon 8060S, 128 GB LPDDR5X). All numbers are **decode** (generation) tok/s unless noted.

### 1bit.systems — Kernel Throughput (our HIP/Vulkan kernels)

| Benchmark | tok/s | TFLOPS | Notes |
|-----------|:-----:|:------:|-------|
| Q1 GEMV (28-layer model) | **417** | — | Q1_0 128B blocks |
| Fused TQ2 (QKV+GU) | **415** | — | 1.15× over individual |
| TQ2 GEMV (28-layer model) | **355** | — | TQ2_0 g128 |
| Sherry 3:4 sparse | **1.57×** over Halo | — | 37.5% bytes-reduction |
| Prefill I8 WMMA | — | **42.2** | Fastest variant (APRE) |

### 1bit.systems — Full Inference (end-to-end decode)

| Engine | Backend | tok/s | Measured |
|--------|---------|:-----:|:--------:|
| Q1 GEMV (Bonsai) | ROCm HIP | **417** | this build |
| GPU ternary (ZINC) | Vulkan | **318** | bench_zinc_vulkan.sh |
| NPU v12 | XDNA 2 xclbin | **69** | validated |
| GPU ROCm HIP (kernels) | ROCm HIP | **64** | bench_rocm_hip.sh |
| NPU FLM | XDNA 2 xclbin | **57** | validated |
| C++ all-5 (auto-detect) | CPU/NPU | **42** | verified |
| GPU Zaya (ROCm HIP) | ROCm HIP | **10.6** | validated |

### llama.cpp ROCm on Same Hardware (for comparison)

| Model | Size | Backend | Quant | tok/s | Source |
|-------|:----:|---------|:-----:|:-----:|--------|
| Ternary-Bonsai-1.7B | 1.7B | PrismML ROCm | Q2_0 | **229** | bench_gpu_1bit.sh (corrected) |
| Qwen3.5-35B-A3B | 35B | llama.cpp ROCm | MXFP4 | **47.3** | community |
| Qwen3.6-35B-A3B | 35B | llama.cpp ROCm | BF16 | **23.7** | community |
| Qwen3.6-35B-A3B | 35B | llama.cpp Vulkan | Q4_K_M | **60.4** | community |

> Why 1bit.systems beats llama.cpp on small models: Our 1.7B ternary model does **417 tok/s** via native HIP kernels vs **229 tok/s** via llama.cpp ROCm running the same model. The gap comes from hand-tuned WMMA GEMV kernels vs llama.cpp general-purpose GEMM path.

---

## Quick Start

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build --target zaya_server -j8
./build/zaya_server  # Starts on port 8088
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
  tests/zaya_server.cpp    398 KB binary (stripped)
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

---

## License

MIT. Sherry-specific kernels: PolyForm Noncommercial 1.0.0.

---

<div align="center">
<a href="https://1bit.systems">Website</a>  <a href="https://github.com/bong-water-water-bong/1bit-systems/issues">Issues</a>  <a href="site/blog/one-binary-to-rule-them-all.html">Blog</a>
</div>