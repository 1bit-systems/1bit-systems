# Performance & Benchmarks

**Single source of truth for 1bit.systems performance claims.** Every number quoted in
`README.md`, `CONTRIBUTING.md`, `ROADMAP.md`, and `site/index.html` should trace back to
this file. If you change a number here, update `site/benchmarks.json` and
`site/numbers.json` to match — those are consumed programmatically and must never drift
from this document again.

**Verified on-device — AMD Ryzen AI Max+ 395 (Strix Halo)**

| Component | Spec |
|-----------|------|
| NPU | XDNA 2, 32 AIE2P tiles, 50 TOPS INT8 |
| GPU | Radeon 8060S (gfx1151), 32 CUs, ~800 GB/s, HIP + Vulkan |
| CPU | Zen 5, 16C/32T |
| RAM | 128 GB unified |
| Binary | `zaya_server` — 288,712 bytes (≈282 KB), Release build, gfx1151 |

---

## At a Glance

| Engine | Hardware | Speed | Status | Power | Model |
|--------|----------|:-----:|--------|:-----:|-------|
| **GPU 1-bit** (llama.cpp) 🏆 | Radeon 8060S | **383 tok/s** | ✅ measured via third-party tool | ~45W | Qwen2-0.5B IQ1_S |
| **GPU ternary** (native kernels) | Radeon 8060S | **433 tok/s** | ✅ validated — Q1 GEMV fused | ~45W | 28-layer synthetic |
| **GPU TQ2 fused** | Radeon 8060S | **420 tok/s** | ✅ validated — QKV+GU fused | ~45W | 28-layer synthetic |
| **GPU TQ2 GEMV** | Radeon 8060S | **367 tok/s** | ✅ validated | ~45W | 28-layer synthetic |
| **GPU BitNet TQ2_0** | Radeon 8060S | **420 tok/s** | ✅ validated — GGUF native | ~45W | 28-layer synthetic |
| **GPU BitNet TQ1_0** | Radeon 8060S | **202 GB/s** | ✅ validated — base-3 LUT | ~45W | 28-layer synthetic |
| **GPU Q1_0 binary** | Radeon 8060S | **380 tok/s** | ✅ validated — 1-bit | ~45W | 28-layer synthetic |
| **NPU FLM** (production) | XDNA 2 · 32 tiles | **94 tok/s** | ✅ validated, coherent | ~15W | Qwen3-0.6B |
| **GPU ternary** (Vulkan) | Radeon 8060S | **307 tok/s** | ✅ validated on-device (3.3 ms/tok) | ~45W | Bonsai-1.7B Q2_0 (1.58-bit) |
| **NPU v12** (fallback) | XDNA 2 · 32 tiles | **69 tok/s** | ⚙️ raw, re-measured 2026-07-12 | ~15W | Qwen3-0.6B |
| **ROCm** (HIP) | Radeon 8060S | **113 tok/s** | reported | ~45W | Bonsai TQ2 ternary |
| **GPU ZINC** (Vulkan F16) | Radeon 8060S | **22 tok/s** | ✅ validated | ~45W | Bonsai-1.7B F16 |

**Status legend:** ✅ *validated* = measured on-device with coherent output · ⚙️ *raw* = kernel runs at this speed, engine output WIP · *reported* = not independently re-measured this pass.

---

## Binary/Ternary GPU Kernels (HIP — Radeon 8060S)

Every new kernel verified bit-exact against CPU reference on real Strix Halo hardware (gfx1151).

| Kernel | Format | Bits/W | Latency (4K×4K) | Apparent BW | Correctness |
|--------|--------|:------:|:---------------:|:-----------:|:-----------:|
| **Q1 GEMV fused** | 128-block Q1_0 | 1.0 | — | — | ✅ exact |
| **Fused TQ2** | QKV+GU fused | 2.0 | — | — | ✅ exact |
| **TQ2 GEMV** | Group-scaled ternary | 2.0 | — | — | ✅ exact |
| **BitNet TQ2_0** | GGML_TYPE_TQ2_0 (llama.cpp native) | 2.06 | 2.0 µs | 16,280 GB/s | ✅ exact |
| **BitNet TQ1_0** | GGML_TYPE_TQ1_0 (base-3 decode) | 1.69 | — | 202 GB/s | ✅ exact |
| **Q1_0 binary** | 128-block sign bits | 1.0 | 1.1 µs | 3,710 GB/s | ✅ exact |
| **TQ1 halo** | Base-3 H1B v4 | 1.58 | 17.5 µs | 202 GB/s | ✅ exact |

Cache-hot benchmarks — real-world throughput depends on model size and weight residency.

---

## Binary/Ternary NPU Kernels (XDNA 2 — AIE2P)

Three on-tile LUT-decode kernels, all compiling via `xchesscc_wrapper aie2p`:

| Phase | Format | Bits/W | Decode Method | Object Size | DDR Savings |
|:-----:|--------|:------:|:-------------:|:-----------:|:-----------:|
| **2** | TQ2 | 2.0 | LUT[256] → byte→4×int8 | 9532 B | 4× vs INT8 |
| **3** | TQ1 | 1.58 | LUT[243] → byte→5×int8 (base-3) | 9624 B | 4.9× vs INT8 |
| **4** | Q1_0 | 1.0 | 64-bit sign mask → ±scale | 11984 B | 3.6× vs INT8 |

NPU bridge: `tq2_to_q4nx` converts any 1BP TQ2 model to Q4NX format for existing NPU engine
(~3.5s for 112 tensors). All three Chess microkernels verified to compile; scalar MAC fallback
in kernel (block-vectorized `mac_8x8_8x8T` path documented as optimization target).

---

## GPU Ternary (Vulkan) — Bonsai-1.7B Q2_0 (1.58-bit)

| Metric | Value |
|--------|-------|
| Decode | **307 tok/s** (3.3 ms/tok) |
| Model | Bonsai-1.7B Q2_0 (1.58-bit ternary) |
| Backend | llama.cpp (Vulkan) |
| Prefill | 3,118 tok/s |

---

## 1-Bit Model Benchmarks (GPU — Radeon 8060S)

Measured via llama.cpp on Radeon 8060S via Vulkan.

| Model | BPW | Size | Params | Prefill | Decode | ms/tok |
|-------|:---:|:----:|:------:|:-------:|:------:|:------:|
| Qwen2 0.5B | **1.06** (IQ1_S) | 296 MB | 494M | 4,188 tok/s | **383 tok/s** | 2.6 |
| Qwen3.5-0.8B | **1.25** (Q1_0) | 268 MB | 752M | 3,883 tok/s | **312 tok/s** | 3.3 |
| Hy-MT2 1.8B | **1.3125** (STQ1_0) | 441 MB | 1.8B | 238 tok/s | **267 tok/s** | 3.7 |
| gemma-2-2b | **1.06** (IQ1_S) | 788 MB | 2.6B | 1,773 tok/s | **158 tok/s** | 6.3 |

---

## DDR Bandwidth Savings — Binary/Ternary Formats

| Format | Bytes per K=64 col | vs INT8 |
|--------|:------------------:|:-------:|
| INT8 (baseline) | 64 | 1× |
| **TQ2** (2-bit) | **16** | **4×** |
| **TQ1** (1.58-bit) | **13** | **4.9× (best)** |
| **Q1_0** (1-bit) | **18** | **3.6×** (block overhead) |

---

## Raw C++ Engine — All 5 Models (M=32 batch, OpenMP)

| Model | Hidden | Size | Prefill | Decode | Tok/s | Layers |
|-------|:------:|:----:|:-------:|:------:|:-----:|:------:|
| **Qwen3-0.6B** | 1,536 | 610 MB | 14 ms/tok | **36 ms/tok** | **28** | 28/28 |
| **Gemma4-E2B** | 2,304 | 4.7 GB | 20 ms/tok | **62 ms/tok** | **16** | 35/35 |
| **Llama-3.1-8B** | 4,096 | 5.7 GB | 47 ms/tok | **100 ms/tok** | **10** | 32/32 |
| **Qwen3-8B** | 4,096 | 6.0 GB | 49 ms/tok | **127 ms/tok** | **8** | 36/36 |

---

## GPU (ROCm HIP Kernels) — Bonsai TQ2

| Metric | Value |
|--------|-------|
| Decode | **113 tok/s** (8.8 ms/tok) |
| Model | Bonsai-1.7B TQ2 ternary |
| Backend | ggml-rocm |

---

## Engine Evolution (July 2026)

| Date | Engine | Decode | Breakthrough |
|------|--------|:------:|--------------|
| Jun 28 | v7 BFP16 | 1930 ms/tok | First working decode |
| Jul 1 | i8 swap | 244 ms/tok | K-interleaving fixed |
| Jul 2 | v9/v12, M=32 batch | 10 ms/tok | M=32 + OpenMP attention |
| Jul 6 | Fused layer | 3.4 ms/tok | One xclbin/transformer layer |
| **Jul 24** | **Binary/ternary GPU kernels** | **1–2 µs** | **Q1_0, BitNet, IQ GPU kernels verified exact** |
| **Jul 24** | **NPU ternary LUT decode** | **3 kernels** | **TQ2/TQ1/Q1_0 on-tile decode via Chess** |
| **Jul 24** | **1BP TQ1 converter** | `--tq1` | **Base-3 1.58-bit format end-to-end** |

---

*All kernel-level numbers verified bit-exact on real Strix Halo hardware (gfx1151). 
End-to-end model benchmark numbers from llama.cpp for cross-validation.*
