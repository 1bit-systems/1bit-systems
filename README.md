<div align="center">

# 1bit.systems

### A completely open-source NPU + GPU inference engine

[![GPU Kernels](https://img.shields.io/badge/GPU-426%20tok/s-00ff00?style=flat-square)](site/blog/one-binary-to-rule-them-all.html)
[![GGUF](https://img.shields.io/badge/GGUF-supported-00ff00?style=flat-square)](src/gguf_loader.cpp)
[![ONNX](https://img.shields.io/badge/ONNX-supported-00ff00?style=flat-square)](src/onnx_loader.cpp)
[![Tests](https://img.shields.io/badge/tests-11%2F11-00ff00?style=flat-square)](tests/)
[![License](https://img.shields.io/badge/license-MIT-00ff00?style=flat-square)](LICENSE)
[![Build](https://img.shields.io/badge/build-passing-00ff00?style=flat-square)](.github/workflows/ci.yml)
[![ROCm](https://img.shields.io/badge/ROCm-7.2.4-blue?style=flat-square)](https://rocm.docs.amd.com)
[![Strix Halo](https://img.shields.io/badge/Strix%20Halo-gfx1151%20%2B%20XDNA%202-12a0ed?style=flat-square)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)

</div>

Reverse-engineered AMD's XDNA 2 NPU in 4 days. Built the first open-source fused NPU+GPU inference engine. Runs on **AMD Strix Halo** (Ryzen AI Max+ 395, Radeon 8060S). Pure C++ — zero Python at runtime.

## Numbers

| What | How fast | Where |
|------|:--------:|-------|
| GPU ternary GEMV | **426 tok/s** | `bench_fused_tq2_1024` |
| GPU 1-bit GEMV | **417 tok/s** | `bench_bonsai_q1_1024` |
| Prefill | **42.2 TFLOPS** | `bench_prefill_variants` |
| GGUF model load | **290 tensors** | `rcpp_bitnet_load_gguf()` |
| Q4NX model load | **311 tensors** | `fused-engine` |

## Quick Start

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build --target zaya_server -j8
./build/zaya_server
```

## What's Inside

- **3 model loaders** — GGUF, ONNX, Q4NX, H1B
- **3 backends** — NPU (XDNA 2), GPU (ROCm/Vulkan), CPU
- **Fused engine** — NPU GEMM + GPU flash attention
- **Agent watchdog** — self-healing thermal/latency/failover
- **Zero Python** — pure C++ from load to inference

## The Story

- [Day 1-4: Reverse-engineered AMD's NPU](site/blog/reverse-engineered-amd-npu-4-days.html)
- [NPU is 132x more efficient than GPU](site/blog/npu-132x-more-efficient-than-gpu.html)
- [72x NPU speedup sprint](site/blog/npu-optimization-sprint-72x-speedup.html)
- [One binary to rule them all](site/blog/one-binary-to-rule-them-all.html)

## Quick Demo

```bash
bash scripts/demo.sh
```

MIT License. Go build something.
