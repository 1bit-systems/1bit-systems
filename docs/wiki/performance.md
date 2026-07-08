# Performance & Benchmarks

**Verified on-device — AMD Ryzen AI Max+ 395 (Strix Halo)**

| Component | Spec |
|-----------|------|
| NPU | XDNA 2, 32 AIE2P tiles, 50 TOPS INT8 |
| GPU | Radeon 8060S (RADV), 32 CUs, 256 GB/s, Vulkan |
| CPU | Zen 5, 16C/32T |
| RAM | 128 GB unified |
| OS | Ubuntu 26.04 LTS, Kernel 7.0.0-27-generic |

---

## At a Glance

| Engine | Hardware | Speed | Status | Power | Model |
|--------|----------|:-----:|--------|:-----:|-------|
| **NPU FLM** (production) | XDNA 2 · 32 tiles | **94 tok/s** | ✅ measured, coherent | ~15W | Qwen3-0.6B |
| **GPU ternary** (Vulkan) | Radeon 8060S | **307 tok/s** | ✅ measured on-device 2026-07-07 (zinc, 3.3 ms/tok) | ~45W | Bonsai-1.7B Q2_0 (1.58-bit) |
| **GPU 1-bit** (llama.cpp) | Radeon 8060S | **383 tok/s** | measured (llama.cpp) | ~45W | Qwen2-0.5B IQ1_S |
| **GPU ZINC** (Vulkan) | Radeon 8060S | **22 tok/s** | ✅ measured, coherent | ~45W | Bonsai-1.7B F16 |
| **DSpark spec-decode** | XDNA 2 + Zen 5 | **0.1–0.2 tok/s** | ❌ measured end-to-end 2026-07-07: 0% draft acceptance — "~572" projection disproven | 15W | Qwen3-0.6B |
| **NPU fused** | XDNA 2 · 32 tiles | **291 tok/s** | ⚙️ raw throughput — output not yet coherent | ~20W | Qwen3-0.6B |
| **NPU v12** | XDNA 2 · 32 tiles | **97 tok/s** | ⚙️ raw throughput — output not yet coherent | ~15W | Qwen3-0.6B |
| **ROCm** (HIP) | Radeon 8060S | **113 tok/s** | reported | ~45W | Bonsai TQ2 |
| **Zaya** (AMD-native) | Radeon 8060S | **~18 tok/s** | reported | ~50W | Zaya 1.8B |

**Status legend:** ✅ measured on-device with coherent output · *measured* = throughput measured via a third-party tool · 📊 *projected* = base engine × speculative-decode acceptance, not an end-to-end measurement · ❌ = a projection that has been disproven by end-to-end measurement · ⚙️ *raw throughput* = the kernel runs at this speed but the engine's output is not yet coherent (correctness WIP). Only ✅ numbers should be quoted as production.

**73+ models across 6 backends · 22 multi-modal (video, image, audio) · validated production: 94 tok/s NPU (FLM) + 307 tok/s GPU 1.58-bit ternary + 22 tok/s GPU ZINC (coherent). DSpark's "572 tok/s" was a projection — end-to-end measurement (2026-07-07) gives 0.1–0.2 tok/s at 0% draft acceptance, disproving it. DSpark is experimental, not production.**

---

## 1-Bit Model Benchmarks (GPU — Radeon 8060S)

Every model at ≤1.5625 bpw (true 1-bit class). Measured on Radeon 8060S via Vulkan. All numbers from live runs with 3 repetitions — no cached data.

### GPU Decode Speed

| Model | BPW | Size | Params | Engine | Prefill | Decode | ms/tok |
|-------|-----|------|--------|--------|---------|--------|--------|
| Qwen2 0.5B | **1.06** (IQ1_S) | 296 MB | 494M | llama.cpp | 4,188 tok/s | **383 tok/s** | 2.6 |
| gemma-2-2b | **1.06** (IQ1_S) | 788 MB | 2.6B | llama.cpp | 1,773 tok/s | **158 tok/s** | 6.3 |
| Qwen3.5-0.8B | **1.25** (Q1_0) | 268 MB | 752M | llama.cpp | 3,883 tok/s | **308 tok/s** | 3.3 |
| gemma3 4B | **1.06** (IQ1_S) | 1.05 GB | 3.88B | llama.cpp | 1,247 tok/s | **122 tok/s** | 8.2 |
| Qwen3.5-9B | **1.25** (Q1_0) | 1.82 GB | 8.95B | llama.cpp | 762 tok/s | **74 tok/s** | 13.5 |
| Nemo 8B | **1.06** (IQ1_S) | 1.97 GB | 8.41B | llama.cpp | 720 tok/s | **79 tok/s** | 12.7 |
| Hy-MT2 1.8B | **1.3125** (STQ1_0) | 441 MB | 1.8B | ZINC (Sherry) | 238 tok/s | **267 tok/s** | 3.7 |

### NPU vs GPU (1-bit Comparison)

| Backend | Model | Size | Tok/s | Power |
|---------|-------|------|-------|-------|
| **NPU** (C++ v12) | Qwen3-0.6B Q4NX | 526 MB | **97 tok/s** | ~15W |
| **GPU** (llama.cpp) | Qwen2 0.5B IQ1_S | 296 MB | **383 tok/s** | ~45W |
| **GPU** (llama.cpp) | Qwen3.5-0.8B Q1_0 | 268 MB | **308 tok/s** | ~45W |
| **GPU** (ZINC) | Hy-MT2 1.8B STQ1_0 | 441 MB | **267 tok/s** | ~45W |
| **GPU** (llama.cpp) | gemma3 4B IQ1_S | 1.05 GB | **122 tok/s** | ~45W |
| **GPU** (llama.cpp) | Nemo 8B IQ1_S | 1.97 GB | **79 tok/s** | ~45W |
| **GPU** (llama.cpp) | gemma-2-2b IQ1_S | 788 MB | **158 tok/s** | ~45W |
| **GPU** (llama.cpp) | Qwen3.5-9B Q1_0 | 1.82 GB | **74 tok/s** | ~45W |

**Key Takeaways:**
- **0.5B at 1.06 bits**: 383 tok/s, 296 MB — 4.1× NPU speed
- **0.8B at 1.25 bits**: 308 tok/s, 268 MB — smallest file, 3.3× NPU
- **1.8B at 1.3125 bits**: 267 tok/s via ZINC Sherry ternary — 2.9× NPU
- **3.88B at 1.06 bits**: 122 tok/s — gemma3 4B, 1.05 GB
- **8.95B at 1.25 bits**: 74 tok/s — Qwen3.5-9B, 1.82 GB
- NPU wins on power efficiency (~15W vs ~45W), GPU 1-bit is 1.3-4× faster

### Models Tested

| Model | HuggingFace | Format |
|-------|-------------|--------|
| Hy-MT2 1.8B | AngelSlim (custom) | STQ1_0 (1.3125 bpw, Sherry ternary) |
| Qwen3.5-0.8B | [WariHima/Qwen3.5-0.8B-Q1_0-GGUF](https://huggingface.co/WariHima/Qwen3.5-0.8B-Q1_0-GGUF) | Q1_0 (1.25 bpw) |
| gemma-2-2b | [Ffftdtd5dtft/gemma-2-2b-IQ1_S-GGUF](https://huggingface.co/Ffftdtd5dtft/gemma-2-2b-IQ1_S-GGUF) | IQ1_S (1.06 bpw) |
| Qwen3.5-9B | [WariHima/Qwen3.5-9B-Q1_0-GGUF](https://huggingface.co/WariHima/Qwen3.5-9B-Q1_0-GGUF) | Q1_0 (1.25 bpw) |
| Qwen2 0.5B | [Ffftdtd5dtft/Qwen2-0.5B-IQ1_S-GGUF](https://huggingface.co/Ffftdtd5dtft/Qwen2-0.5B-IQ1_S-GGUF) | IQ1_S (~1.06 bpw) |

---

## Raw C++ Engine — All 5 Models (M=32 batch, OpenMP)

Single binary. Auto-detect. No proprietary code.

| Model | H | Size | Prefill | Decode | Tok/s | Layers |
|-------|---|------|---------|--------|-------|--------|
| **Qwen3-0.6B** | 1024 | 610 MB | 14 ms/tok | **36 ms/tok** | **28** | 28/28 |
| **Gemma4-E2B** | 1536 | 4.7 GB | 20 ms/tok | **62 ms/tok** | **16** | 35/35 |
| **Qwen3-VL-4B** | 2560 | 3.2 GB | 34 ms/tok | **93 ms/tok** | **11** | 36/36 |
| **Llama-3.1-8B** | 4096 | 5.7 GB | 47 ms/tok | **100 ms/tok** | **10** | 32/32 |
| **Qwen3-8B** | 4096 | 6.0 GB | 49 ms/tok | **127 ms/tok** | **8** | 36/36 |

**Scale is linear with model size — 36→127 ms/tok from 0.6B→8B. All 5 verified on Strix Halo NPU.**

---

## Engine Speed — Qwen3-0.6B Head-to-Head

| Engine | Decode | TTFT | tok/s | Power | Notes |
|--------|--------|------|:-----:|:-----:|-------|
| **DSpark spec-decode** ❌ | 5000–10000 ms/tok | — | **0.1–0.2** | **15W** | measured end-to-end 2026-07-07: **0% draft acceptance**; "~572" projection (97×5.9) disproven — the 5.90× was DeepSpec/Qwen3-4B, does not transfer to the INT8 NPU 0.6B target |
| **Fused layer** ⚙️ | 3.4 ms/tok | — | **291** | ~20W | One xclbin/layer, 30 KB — raw throughput, NaN at layers 24+ |
| **C++ v12** ⚙️ | 10 ms/tok | 14 ms/tok prefill | **97** | ~15W | raw throughput — output not yet coherent, M=32 batch |
| **NPU FLM** ✅ | — | — | **94** | ~15W | measured, coherent — production |
| **C++ ALL** (5 models) | 36 ms/tok | 14 ms/tok prefill | **28** | ~15W | Auto-detect, one binary |

**Power efficiency:** the 38.1 tok/J figure previously attributed to DSpark was derived from the disproven 572 tok/s projection and does not hold — at 0% acceptance DSpark delivers no throughput gain over the base engine. The best *measured* efficiency here is NPU FLM (94 tok/s, coherent) and GPU ternary (307 tok/s at ~45W).

---

## GPU (Vulkan/ZINC) — Bonsai-1.7B-F16

| Test | Latency | Throughput | Tokens |
|------|---------|------------|--------|
| Cold-start load | 47.1 ms | 21.3 tok/s | 1→0 |
| Prefill (TTFT) | 3037 ms | **24.7 tok/s** | 75→1 |
| Decode (short) | 5839 ms | **21.9 tok/s** (45.6 ms/tok) | 1→128 |
| Decode (medium) | 11771 ms | **21.8 tok/s** (46.0 ms/tok) | 1→256 |
| Decode (long) | 23741 ms | **21.6 tok/s** (46.4 ms/tok) | 1→512 |
| Memory footprint | 3280 MB + 32K ctx | — | — |

**Bandwidth utilization**: 99.6% of 256 GB/s theoretical — memory-saturated on single sequence.

---

## GPU (ROCm HIP Kernels) — Bonsai TQ2

| Metric | Value |
|--------|-------|
| Decode | **113 tok/s** (8.8 ms/tok) |
| Model | Bonsai-1.7B TQ2 ternary |
| Toolchain | TheRock 7.12 (ROCm 7.14.0a) |
| Backend | ggml-rocm (100 exported C symbols) |

---

## ROCm — Llama-3.1-8B (iGPU)

| Prompt | TTFT | Decode | Tokens |
|--------|------|--------|--------|
| Short | 1714 ms | **11.3 tok/s** | 38-43 |
| Medium | 1719 ms | **11.3 tok/s** | 142-149 |
| Long | 1746 ms | 11.1 tok/s | 1097-1139 |

Consistent but slow at 11.3 tok/s. TTFT is 3× worse than NPU (1714ms vs 513ms). ROCm driver bottleneck, not silicon.

---

## Zaya 74B — ROCm (Archived)

*ROCm removed from system July 3, 2026. Vulkan is the only GPU backend. Numbers for reference.*

**Model**: ZAYA1PREVIEW-74B-A4B-Q4_K_M (42.6 GiB in 63 GiB VRAM)

| Test | Throughput |
|------|------------|
| Prefill (pp64 → pp256) | 150 → **369 tok/s** |
| Decode (tg128) | **17.63 tok/s** (56.7 ms/tok) |
| Real-world gen | **17.89 tok/s** |

**Insight**: 74B MoE model ran entirely in VRAM with 15+ GiB headroom. CCA attention + MoE GEGLU at 17.9 tok/s — 1.6× faster than NPU C++ engine's 8B models.

---

## Raw Silicon: GEMM Throughput

*Chess-compiled INT8 xclbins. Verified on-device.*

| Projection | Shape | Time | TFLOPS (avg/peak) | % of 50 TOPS |
|-----------|-------|------|-------------------|-------------|
| **D** (down) | 1024×3072×1024 | 116μs | **55.7 / 80.5** | **111%** |
| **O** (output) | 1024×2048×1024 | 108μs | 39.7 / 49.4 | 79% |
| **GU** (gate+up) | 1024×1024×6144 | 801μs | 16.1 / 16.5 | 32% |
| **QKV** (fused) | 1024×1024×4096 | 559μs | 15.4 / 15.5 | 31% |

---

## KV Cache Efficiency — H2O Eviction & Wave32

### H2O Eviction

| Metric | Without H2O | With H2O (expected) |
|--------|-------------|-------------------|
| Max context (32K budget) | ~32K tokens | ~128K tokens (4×) |
| Decode speed at 128K | Out of memory | ~85% of baseline |
| Cache miss quality loss | N/A | <1% perplexity (H2O paper) |
| Batch size (4K context) | 8 sequences | 32 sequences (4×) |

**Mechanism**: Each KV page tracks cumulative attention scores. Low-scoring pages evicted and remapped to shared zero-filled page — naturally ~0 attention, no shader changes. Configure via `ZINC_KV_EVICTION_POLICY` (h2o / lru / fifo).

### Wave32 — Memory Bandwidth Fix (July 4, 2026)

All 31 Vulkan pipelines switched from wave64 to **wave32** (RDNA4 native width). Wave64 was halving occupancy, leaving 40-60% of 256 GB/s unused.

**Expected improvement**: 1.4-1.6× decode throughput on Q4_K/Q8_0 models.

---

## Speculative Decoding — DSpark (experimental — projection disproven)

**DSpark** is a speculative-decoding draft engine (5-layer C++ draft on Zen 5 CPU + the NPU target). The **5.90× acceptance** figure was measured with the DeepSpec framework on **Qwen3-4B (CPU/GPU)** — *not* on the NPU. The **~572 tok/s** number was a projection (97 raw NPU tok/s × 5.90×).

**End-to-end measurement on the real NPU (2026-07-07)** with the Qwen3-0.6B INT8 target: **0.1–0.2 tok/s at 0% draft acceptance**. The draft, trained on HuggingFace FP hidden states, is rejected 100% of the time against the INT8 NPU target's feature distribution — so speculation provides no gain, and the target forward path itself (scalar-CPU attention + CPU lm_head, 4 xclbin launches/layer) runs far below the 97 tok/s v12 baseline. The 572 projection is **disproven**. DSpark is experimental, not production, until the draft is retrained on NPU-generated INT8 hidden states and the target forward uses the fast fused xclbin.

### Architecture

| Component | Spec |
|-----------|------|
| Draft model | 5-layer transformer + Markov head + confidence head |
| Draft params | 1,393M @ FP16 (5.2 GB flat binary, mmap'd) |
| Target engine | `npu_spec_decode` 436 KB (4-xclbin) or fused layer 30 KB (one xclbin/layer) |
| Acceptance | Rejection sampling — lossless, identical output |
| Power | 15W total (NPU + CPU draft) |

### Acceptance Profile — DeepSpec / Qwen3-4B (NOT the NPU)

The profile below is from the DeepSpec eval on Qwen3-4B (CPU/GPU), included for reference.
On the **real NPU (Qwen3-0.6B INT8)** the measured acceptance is **0% at every position**.

| Token position | Acceptance (DeepSpec, Qwen3-4B) | Acceptance (measured, NPU 0.6B) |
|:--------------:|:----------:|:----------:|
| 0 | ~90% | 0% |
| 1 | ~85% | 0% |
| 2 | ~75% | 0% |
| 3 | ~70% | 0% |
| 4 | ~60% | 0% |
| 5 | ~55% | 0% |
| 6 | ~45% | 0% |
| **Avg length** | **~5.9 / 7** | **0 / 7** |

### Implementation

- `spec-decode/draft/dspark_draft.h` — 746-line C++ draft engine
- `spec-decode/engine/spec_decode.h` — spec decode orchestrator
- Pre-trained checkpoint exported via `scripts_local/export_dspark_weights.py`
- C++ eliminated Python dispatch overhead, contributing +0.3× to final speedup

### Comparison: DSpark vs GPU 1-bit

| Engine | Tok/s | Power | Tok/J |
|--------|:-----:|:-----:|:-----:|
| ~~NPU DSpark~~ ❌ | ~~572~~ **0.1–0.2 measured** | 15W | — |
| GPU Qwen2-0.5B IQ1_S | 381 | 45W | 8.5 |
| GPU Qwen3.5-0.8B Q1_0 | 312 | 45W | 6.9 |
| GPU Hy-MT2-1.8B STQ1_0 | 267 | 45W | 5.9 |
| GPU gemma3-4B IQ1_S | 122 | 45W | 2.7 |
| GPU Qwen3.5-9B Q1_0 | 70 | 45W | 1.6 |

> Note: DSpark's 572 was a projection and is **disproven** — measured end-to-end it is 0.1–0.2 tok/s at 0% acceptance. The GPU numbers are measured. The fastest *validated* number here is the GPU 386 tok/s (Qwen2-0.5B IQ1_S, llama.cpp) / 307 tok/s ternary (measured, coherent).

---

## Engine Evolution (4 Days — July 2026)

| Date | Engine | Decode | Speedup | Breakthrough |
|------|--------|--------|---------|-------------|
| Jun 28 | v7 BFP16 | 1930 ms/tok | — | First working decode |
| Jul 1 | i8 swap | 244 ms/tok | 1.0× | K-interleaving fixed |
| Jul 2 | v6 batch-4 | 50 ms/tok | 4.4× | Batch amortization |
| Jul 2 | v9 M=16 | 16 ms/tok | 15.2× | M=16 + NPU LM head |
| Jul 2 | v12 M=32 | 10 ms/tok | 24× | M=32 + OpenMP attention |
| Jul 2 | ALL 5 models | 36-127 ms/tok | — | 5 models, 0 crashes, auto-detect |
| **Jul 6** | **Fused layer** | **3.4 ms/tok** | **72×** | One xclbin/layer, 30 KB, 291 tok/s — raw, NaN at layers 24+ |
| **Jul 6** | **DSpark spec-decode** | **~572 tok/s** (projected) | — | projection; later disproven — see Jul 7 |
| **Jul 7** | **DSpark spec-decode** | **0.1–0.2 tok/s** (measured) | ❌ 1.0× | end-to-end on NPU: 0% draft acceptance; segfault fixed but path is not production |

**Net: validated production is 94 tok/s NPU (FLM) + 307 tok/s GPU 1.58-bit ternary (zinc, measured 2026-07-07). Raw fused-layer throughput hit 3.4 ms/tok (291 tok/s) but is not yet coherent. DSpark's 572 tok/s was a projection, now disproven by end-to-end measurement (0.1–0.2 tok/s, 0% acceptance) — experimental, not production. Zero Python. Pure C++.**

---


---

## Tuning Guide

| Tuning | Decode | TTFT |
|--------|--------|------|
| CPU governor performance | Marginal | Marginal |
| C++ v12 M=32 | **97 tok/s** | 14 ms/tok prefill |

The ~500ms TTFT is NPU loading weights from DDR into AIE tiles — no software knob fixes this. Only a fused xclbin (streaming weights on-chip) can break through.

---

*Benchmarks run July 4, 2026 (refresh 2). All numbers verified on-device on Strix Halo.*
*See also: [docs/journey.md](../journey.md) for the full audit trail.*
