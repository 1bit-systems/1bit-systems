---
type: Reference
title: GPU 1-Bit Model Benchmarks
description: All 1-bit/ternary models tested on Radeon 8060S (Vulkan) at ≤1.5625 bpw. Measured via llama.cpp/ZINC, 3 repetitions.
tags: [benchmarks, gpu, vulkan, 1-bit, ternary, radeon]
timestamp: 2026-07-06T00:00:00Z
---

# Overview

All models at ≤1.5625 bpw, measured via Vulkan on Radeon 8060S, 3 repetitions.

## Benchmarks

| Model | BPW | Size | Engine | Decode (tok/s) |
|-------|-----|------|--------|-----------------|
| Qwen2 0.5B | 1.06 (IQ1_S) | 296 MB | llama.cpp | **381** |
| Qwen3.5-0.8B | 1.25 (Q1_0) | 268 MB | llama.cpp | **312** |
| Hy-MT2 1.8B | 1.3125 (STQ1_0) | 441 MB | ZINC (Sherry) | **267** |
| gemma-2-2b | 1.06 (IQ1_S) | 788 MB | llama.cpp | **158** |
| gemma3 4B | 1.06 (IQ1_S) | 1.05 GB | llama.cpp | **122** |
| Nemo 8B | 1.06 (IQ1_S) | 1.97 GB | llama.cpp | **79** |
| Qwen3.5-9B | 1.25 (Q1_0) | 1.82 GB | llama.cpp | **70** |

## Notes

- All models quantized to ≤1.5625 bpw (bits per weight)
- **Radeon 8060S** integrated GPU on AMD Strix Halo APU
- llama.cpp backend for IQ1_S and Q1_0 formats
- ZINC (custom Zig engine) for STQ1_0 (Sherry format)

## Full Reference

See `docs/wiki/performance.md` for the complete live-verified table.

## Citations

[1] [Performance wiki](https://github.com/1bit-systems/1bit/blob/main/docs/wiki/performance.md)
