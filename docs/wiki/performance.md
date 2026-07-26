# Performance & Benchmarks

**Single source of truth for 1bit.systems performance claims.** Every number here is
pulled directly from [`site/benchmarks.json`](../../site/benchmarks.json)
(`"_authoritative": true`). `README.md` and `site/index.html` link here instead of
restating tables — if you change a number, change it in `benchmarks.json` first, then
regenerate this page and `site/numbers.json`/`site/badge_*.json` from it. Do not hand-edit
numbers into more than one file again — that's what caused this page to drift roughly two
weeks out of date the last time it was hand-maintained (see git history).

**Verified on-device — AMD Ryzen AI Max+ 395 (Strix Halo)**

| Component | Spec |
|-----------|------|
| NPU | XDNA 2, 32 AIE2P tiles, 51 TOPS INT8 (measured via `xrt-smi validate`) |
| GPU | Radeon 8060S (gfx1151), 32 CUs, HIP + Vulkan |
| CPU | Zen 5, 16C/32T |
| RAM | 128 GB unified |

---

## Kernel-Level Microbenchmarks (synthetic 28-layer weight buffer)

> ⚠️ These measure single-GEMM-kernel throughput, isolated and correctness-verified
> bit-exact against a CPU reference. They exclude KV-cache attention, softmax, RoPE,
> non-GEMM FFN ops, sampler, tokenizer, and host↔device transfers — **not** an
> end-to-end decode number. See the End-to-End table below and
> [issue #235](https://github.com/bong-water-water-bong/1bit-systems/issues/235).

| Kernel | Value | Backend | Status |
|--------|:-----:|---------|--------|
| Q1 GEMV (fused) | **433 tok/s** | ROCm HIP | ✅ validated, re-measured 2026-07-24 |
| Fused TQ2 (QKV+GU) | **420 tok/s** | ROCm HIP | ✅ validated, re-measured 2026-07-24 |
| BitNet TQ2_0 (GGML native) | **420 tok/s** | ROCm HIP | ✅ validated, re-measured 2026-07-24 |
| Q1_0 binary | **380 tok/s** | ROCm HIP | ✅ validated, re-measured 2026-07-24 |
| TQ2 GEMV | **367 tok/s** | ROCm HIP | ✅ validated, re-measured 2026-07-24 |
| GPU ternary (Vulkan) | **318 tok/s** | Vulkan ZINC | ✅ validated |
| BitNet TQ1_0 (base-3 LUT) | **202 tok/s** | ROCm HIP | ✅ validated, re-measured 2026-07-24 |
| Prefill INT8 WMMA (I8-APRE) | **39.4 TFLOPS** | INT8 WMMA | ✅ re-validated 2026-07-26 (was 40.5, colder-GPU pass) |
| ROCm HIP (kernels) | **64 tok/s** | ROCm HIP | ✅ validated |
| IQ1_S dequant+GEMV | **45 tok/s** | ROCm HIP | ✅ validated — correctness pending full IQ1_M port |
| NPU v12 | **69 tok/s** | XDNA 2 (32 tiles), Qwen3-0.6B | ⚙️ optimized, re-measured 2026-07-12 — not re-verified since |

**NPU raw hardware validation** (`xrt-smi validate`, 2026-07-25): 51 TOPS INT8 GEMM,
50µs avg latency, 74,735–75,404 op/s — confirms the NPU/driver/firmware stack is healthy.
This is a device-level number, not a model-inference tok/s figure.

---

## End-to-End Inference (real model, real prompts)

| Model | Value | Backend | Notes |
|-------|:-----:|---------|-------|
| BlackMamba 1.5B | **79.4 tok/s** | Mamba1 HIP (Strix Halo) | Full decode, alternating SSM/MoE dispatch. Re-validated 2026-07-26 after `__shfl_xor_sync` kernel fixes. |
| llama.cpp ROCm (PrismML, third-party) | **229 tok/s** | Same hardware | Comparison point, not our engine. See [issue #235](https://github.com/bong-water-water-bong/1bit-systems/issues/235). |
| BlackMamba 2.8B | **46.0 tok/s** | Mamba1 HIP (Strix Halo) | Full decode. Re-validated 2026-07-26. Reachable today only via the server's internal benchmark thread — `POST /v1/completions` hangs, see [issue #922](https://github.com/bong-water-water-bong/1bit-systems/issues/922). |
| zaya_server (Qwen 27B Q4_K) | **30 tok/s** | ROCm HIP | Full decode, speculative MTP, Strix Halo |
| ZR1-1.5B (Zyphra) | **26 tok/s** | Vulkan ZINC | Reasoning-tuned dense transformer, Qwen2 arch |
| zaya_server (Qwen 35B MoE Q4_K) | **20 tok/s** | ROCm HIP | Full decode, speculative MTP, Strix Halo |
| CPU (generic backend), ZAYA1-8B-shaped | **2.5 tok/s** | AVX-512 CPU, portable path | Real `forward()`+`generate()` loop, not a synthetic kernel. Steady-state 5-token average; single-pass first-token latency was 4.37 tok/s. |

---

## DDR Bandwidth Savings — Binary/Ternary Formats

| Format | Bytes per K=64 col | vs INT8 |
|--------|:------------------:|:-------:|
| INT8 (baseline) | 64 | 1× |
| **TQ2** (2-bit) | **16** | **4×** |
| **TQ1** (1.58-bit) | **13** | **4.9× (best)** |
| **Q1_0** (1-bit) | **18** | **3.6×** (block overhead) |

---

## Engine Evolution (July 2026)

| Date | Engine | Decode | Breakthrough |
|------|--------|:------:|--------------|
| Jul 1 | i8 swap | 244 ms/tok | K-interleaving fixed |
| Jul 2 | v9/v12, M=32 batch | 10 ms/tok | M=32 + OpenMP attention |
| Jul 6 | Fused layer | 3.4 ms/tok | One xclbin/transformer layer |
| Jul 24 | Binary/ternary GPU kernels | 1–2 µs | Q1_0, BitNet, IQ GPU kernels verified exact |
| Jul 24 | NPU ternary LUT decode | 3 kernels | TQ2/TQ1/Q1_0 on-tile decode via Chess |
| Jul 25 | NPU HW re-validated + zero-copy fusion fix | — | `xrt-smi validate` clean; fixed buffer-overflow segfault in fusion pipeline test |
| Jul 26 | Mamba1 HIP re-measure | 79.4 / 46.0 tok/s | Fixed `__shfl_xor_sync` correctness bug, numbers went *up* |

---

*All kernel-level numbers verified bit-exact on real Strix Halo hardware (gfx1151), median
of 3 runs. Status legend: ✅ validated · ⚙️ optimized (kernel runs at this speed, engine
integration in progress).*
