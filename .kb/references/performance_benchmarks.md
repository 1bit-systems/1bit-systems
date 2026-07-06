---
type: Reference
title: Performance Benchmarks (July 2026)
description: Source-of-truth benchmark results across all engines — NPU C++ v12, FLM fallback, GPU Vulkan, ROCm — from the July 6, 2026 refresh.
tags: [benchmarks, performance, npu, gpu, rocm]
resource: https://github.com/1bit-systems/1bit/blob/main/docs/wiki/performance.md
timestamp: 2026-07-06T00:00:00Z
---

# Overview

Live-verified benchmark results from the July 6, 2026 refresh. Full source at the [performance wiki](https://github.com/1bit-systems/1bit/blob/main/docs/wiki/performance.md).

## Engine Head-to-Head (Qwen3-0.6B)

| Engine | tok/s | ms/tok | Notes |
|--------|-------|--------|-------|
| NPU fused layer | **291** | 3.4 | Production — one xclbin call/layer, 38 KB binary |
| NPU C++ v12 | **97** | 10.3 | Fallback — M=32 batched dispatch |
| FLM proxy | **94** | 10.6 | Fallback v2 — AMD proprietary |
| C++ all-5 | **28** | 35.7 | All 5 models auto-detected |
| GPU (ROCm) | **22** | 45.5 | Via zaya-llama.cpp ggml-rocm |

## Benchmark — Fused Layer Engine

| Model | Engine | tok/s | Binary |
|-------|--------|-------|--------|
| Qwen3-0.6B | Fused layer | **291** | 38 KB |
| Qwen3-0.6B | C++ v12 (fallback) | 97 | 104 KB |
| Qwen3-0.6B (all-5) | All-5 | 28 | 117 KB |

## Full Benchmark Source

See the [full performance wiki](https://github.com/1bit-systems/1bit/blob/main/docs/wiki/performance.md) for:
- Complete 1-bit model tables
- All engine head-to-head comparisons
- ROCm custom kernel benchmarks
- Eagle3 speculative decoding benchmarks
- H2O KV cache eviction benchmarks
- Wave32 GPU optimization benchmarks

## Citations

[1] [Performance wiki](https://github.com/1bit-systems/1bit/blob/main/docs/wiki/performance.md)
[2] [Engine bench deep-dive](https://github.com/1bit-systems/1bit/blob/main/engine/npu/BENCHMARKS.md)
[3] [GPU 1-Bit Benchmarks](/references/gpu_benchmarks.md)
