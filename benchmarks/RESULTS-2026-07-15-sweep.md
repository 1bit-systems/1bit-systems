# Full Benchmark Sweep — 2026-07-15

**Focus:** 1-bit ternary performance on Strix Halo
**Host:** AMD Ryzen AI MAX+ 395 w/ Radeon 8060S
**GPU:** gfx1151 (RDNA 3.5, Wave32) — TheRock 7.15.0a
**NPU:** XDNA 2 — 32 tiles (8 cols × 4 core rows)

## Kernel Microbenchmarks

| Benchmark | This sweep | Prior | Δ |
|-----------|:----------:|:-----:|:-:|
| Sherry GEMV | **143.7 GB/s** | 153.0 | -6% |
| TQ1 GEMV | **183.2 GB/s** | 191.6 | -4% |
| Halo GEMV | **150.4 GB/s** | 162.8 | -8% |
| Prefill 4h | **21.35 TFLOPS** | 21.77 | -2% |
| Prefill I8-APRE | **37.05 TFLOPS** | 38.89 | -5% |

*Note: Variation within expected thermal/DVFS range. All within 15% tolerance.*

## 1-bit Ternary Full-Model Decode (Qwen3/Bonsai, 28-layer)

| Format | tok/s | ms/tok | BW | Notes |
|--------|:-----:|:------:|:--:|-------|
| Q1_0 tile8 | **75.4** | 13.27 ms | 15 GB/s | Baseline, tile-interleaved |
| Q1_0 1024-block | **429** | 2.33 ms | 75 GB/s | **Fastest 1-bit format** |
| TQ2_1024 | **366** | 2.73 ms | 126 GB/s | Ternary quantized |
| Fused TQ2 (QKV+GU) | **421** | 2.38 ms | — | 1.17× over individual |

### Key takeaway
Q1_0 1024-block is the **fastest 1-bit format** on this hardware at **429 tok/s**, but TQ2_1024 has **higher effective bandwidth** (126 GB/s vs 75 GB/s). The fused QKV+GU kernel recovers most of the TQ2 gap by reducing launch overhead (1.17× speedup over individual launches).

## KV Cache Benchmarks (seq_len=2048)

| Kernel | tok/s | vs FP16 |
|--------|:-----:|:-------:|
| FP16 attention | 853 | 1.00× |
| INT8 attention | 814 | 0.95× |
| Flash-Decoding | **10,731** | **12.58×** |
| FD + PlanarQuant-3 | **8,679** | 10.17× |

## Zaya Engine — ROCm HIP (gfx1151)

| Metric | Value |
|--------|:-----:|
| Forward (40 layers) | 808 ms |
| Layers/sec | 49.5 |
| Precision | FP16 |
| MoE experts | 16 × 40 (enabled) |

## Core Validated Numbers (from upstream)

| Engine | tok/s | Source |
|--------|:-----:|--------|
| GPU 1-bit (llama.cpp ROCm) | 373 | Upstream validated |
| GPU ternary (Vulkan ZINC) | 369 | Upstream validated |
| ROCm HIP (Bonsai 1.7B) | 65 | Upstream validated |
| NPU FLM (Qwen3-0.6B) | 57 | Upstream validated |

## Binary Sizes

| Binary | Size |
|--------|:----:|
| `zaya_server` | 282 KB |
| `unified_server` | 1.2 MB |
| `bitnet_decode` | 688 KB |
| `librocm_cpp.so` | 1.0 MB |
