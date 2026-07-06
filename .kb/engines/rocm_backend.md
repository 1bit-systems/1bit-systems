---
type: Engine
title: ROCm Custom Kernel Backend
description: Custom HIP kernels for 1-bit and ternary inference on AMD Strix Halo (gfx1151), folded into zaya-llama.cpp as ggml-rocm backend.
tags: [rocm, hip, gpu, ternary, 1-bit, ggml, kernels]
resource: https://github.com/1bit-systems/zaya-llama.cpp/tree/main/ggml/src/ggml-rocm/
timestamp: 2026-07-06T00:00:00Z
---

# Overview

The ROCm backend lives inside the `zaya-llama.cpp/` fork as `ggml/src/ggml-rocm/`. It provides custom HIP kernels for 1-bit and ternary inference on AMD Strix Halo (gfx1151).

## Exported Symbols

100 exported C API symbols covering:

| Category | Kernels |
|----------|---------|
| Ternary GEMV | Matrix-vector for ternary weights |
| Bonsai Q1/TQ2 | 1-bit and ternary-quantized Bonsai models |
| Sherry | Sherry model inference kernels |
| KV Cache Attention | Prefill, decode, FD, I8, PQ3 |
| Prefill GEMM | 28.4 TFlops prefill GEMM |
| Medusa | Medusa tree attention for speculative decoding |
| Model Loader | Loader for GGUF models |
| Tokenizer | Tokenizer integration |

## Build

Built standalone at `1bit/build-rocm/librocm_cpp.so` or as a ggml backend.

## Toolchain

Uses **TheRock 7.12** ROCm toolchain via nightly pip (ROCm 7.14.0a):

```bash
# ROCm venv at /tmp/rocm-venv/
# System ROCm at /opt/rocm
```

## Supported Models

| Model | BPW | Config |
|-------|-----|--------|
| Bonsai-1.7B TQ2 | 2.0 | 113 tok/s decode (8.8 ms/tok) |
| Zaya | Ternary | Hybrid CCA + MoE with EDA router |

## Performance

**Bonsai-1.7B TQ2**: 113 tok/s decode at 8.8 ms/tok — the fastest 1-bit model on ROCm.

## Citations

[1] [GPU Benchmarks](/references/gpu_benchmarks.md)
[2] [Zaya Architecture](/structures/zaya_architecture.md)
