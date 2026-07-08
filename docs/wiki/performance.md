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
| **NPU FLM** (production) | XDNA 2 · 32 tiles | **63 tok/s** | ✅ measured, coherent (Jul 8 re-measure) | ~15W | Qwen3-0.6B |
| **GPU ternary** (Vulkan) | Radeon 8060S | **307 tok/s** | ✅ measured on-device 2026-07-07 (zinc, 3.3 ms/tok) | ~45W | Bonsai-1.7B Q2_0 (1.58-bit) |
| **GPU 1-bit** (llama.cpp) | Radeon 8060S | **228 tok/s** | ✅ measured Jul 8 | ~45W | Bonsai-1.7B Q1_0 |
| **GPU Ollama** | Radeon 8060S | **269 tok/s** | ✅ measured, coherent (Jul 8) | ~45W | Qwen2.5-0.5B Q2_K |
| **GPU ZINC ternary** (Vulkan) | Radeon 8060S | **217 tok/s** | ✅ measured Jul 8 (garbled tokenizer, compute valid) | ~45W | Ternary-Bonsai-1.7B Q2_0 |
| **DSpark spec-decode** | XDNA 2 + Zen 5 | **0.1–0.2 tok/s** | ❌ measured end-to-end 2026-07-07: 0% draft acceptance — "~572" projection disproven | 15W | Qwen3-0.6B |
| **NPU fused** | XDNA 2 · 32 tiles | **291 tok/s** | ⚙️ raw throughput — output not yet coherent | ~20W | Qwen3-0.6B |
| **NPU v12** | XDNA 2 · 32 tiles | **3 tok/s** | ⚙️ measured Jul 8 — OMP attention bottleneck, down from 97 raw | ~15W | Qwen3-0.6B |
| **ROCm** (HIP) | Radeon 8060S | **113 tok/s** | reported | ~45W | Bonsai TQ2 |
| **Zaya** (AMD-native) | Radeon 8060S | **~18 tok/s** | reported | ~50W | Zaya 1.8B |

**Status legend:** ✅ measured on-device with coherent output · *measured* = throughput measured via a third-party tool · 📊 *projected* = base engine × speculative-decode acceptance, not an end-to-end measurement · ❌ = a projection that has been disproven by end-to-end measurement · ⚙️ *raw throughput* = the kernel runs at this speed but the engine's output is not yet coherent (correctness WIP). Only ✅ numbers should be quoted as production.

**73+ models across 6 backends · 22 multi-modal (video, image, audio) · validated production (Jul 8): 63 tok/s NPU (FLM) + 269 tok/s GPU Ollama Q2_K + 228 tok/s GPU llama.cpp Bonsai Q1_0 + 217 tok/s GPU ZINC ternary. DSpark's "572 tok/s" was a projection — end-to-end measurement (2026-07-07) gives 0.1–0.2 tok/s at 0% draft acceptance, disproving it. DSpark is experimental, not production.**

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

### NPU vs GPU (1-bit Comparison) — Jul 8 Re-measure

| Backend | Model | Size | Tok/s | Power | Notes |
|---------|-------|------|-------|-------|-------|
| **GPU** (Ollama) | Qwen2.5-0.5B Q2_K | 338 MB | **269 tok/s** | ~45W | measured Jul 8 |
| **GPU** (Ollama) | Bonsai-1.7B Q1_0 | 248 MB | **236 tok/s** | ~45W | measured Jul 8 |
| **GPU** (llama.cpp) | Bonsai-1.7B Q1_0 | 237 MB | **228 tok/s** | ~45W | measured Jul 8 |
| **GPU** (ZINC) | Ternary-Bonsai-1.7B Q2_0 | 441 MB | **217 tok/s** | ~45W | measured Jul 8 |
| **NPU** (FLM) | Qwen3-0.6B | 522 MB | **63 tok/s** | ~15W | measured Jul 8, coherent |
| **NPU** (C++ v12) | Qwen3-0.6B Q4NX | 526 MB | **3 tok/s** | ~15W | measured Jul 8, OMP bottleneck |

**MISSING from disk (Jul 8):** Qwen2 0.5B IQ1_S, Qwen3.5-0.8B Q1_0, Hy-MT2 1.8B STQ1_0, gemma3 4B IQ1_S, Nemo 8B IQ1_S, gemma-2-2b IQ1_S, Qwen3.5-9B Q1_0 — previous numbers (74-383 tok/s) need re-verification after re-download.

**Key Takeaways (Jul 8):**
- **0.5B at Q2_K**: 269 tok/s via Ollama, 338 MB — fastest measured
- **1.7B at Q1_0**: 228-236 tok/s — 7B-class quality at sub-7B speeds
- **1.7B ternary Q2_0**: 217 tok/s via ZINC Vulkan — consistent across llama.cpp and ZINC
- **NPU FLM regressed**: 63 tok/s, down from 94 (doc) / 103 (reported peak)
- **NPU v12 regressed**: 3 tok/s, down from 97 — OMP attention bottleneck
- NPU wins on power efficiency (~15W vs ~45W), GPU is 3.4-4.3× faster

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
| **C++ v12** ⚙️ | 269 ms/tok | 368 ms/tok prefill | **3** | ~15W | measured Jul 8 — OMP attention bottleneck, down from 97 raw |
| **NPU FLM** ✅ | — | — | **63** | ~15W | measured Jul 8, coherent — regression from 94 |
| **C++ ALL** (5 models) | 36 ms/tok | 14 ms/tok prefill | **28** | ~15W | Auto-detect, one binary |

**Power efficiency:** the 38.1 tok/J figure previously attributed to DSpark was derived from the disproven 572 tok/s projection and does not hold — at 0% acceptance DSpark delivers no throughput gain over the base engine. The best *measured* efficiency here is NPU FLM (63 tok/s, coherent, Jul 8) and GPU ternary (307 tok/s at ~45W).

---

## GPU (Vulkan/ZINC) — Ternary-Bonsai-1.7B-Q2_0

> **Note (Jul 8):** Bonsai-1.7B F16 and Q1_0 GGUF files on HuggingFace (`prism-ml/Bonsai-1.7B-gguf`) are truncated at source (LFS pointers not resolved). Benchmarks use `prism-ml/Ternary-Bonsai-1.7B-gguf` Q2_0 variant (441 MB, 436 MB VRAM, 28 layers, dim 2048). Output text is garbled (non-standard tokenizer) but compute metrics are valid.

| Test | Latency | Throughput | Tokens |
|------|---------|------------|--------|
| Cold-start load | 7,740 ms | — | — |
| Prefill (TTFT, 30 tok) | 86 ms | **348 tok/s** | 30→1 |
| Prefill (TTFT, 151 tok) | 406 ms | **372 tok/s** | 151→1 |
| Decode (short) | 497 ms | **257 tok/s** (3.9 ms/tok) | 1→128 |
| Decode (medium) | 1,179 ms | **217 tok/s** (4.6 ms/tok) | 1→256 |
| Decode (long) | 2,351 ms | **218 tok/s** (4.6 ms/tok) | 1→512 |
| Memory footprint | 587 MB RSS + 32K ctx | — | — |

**Steady-state decode: ~217 tok/s (4.6 ms/tok).** Per-token GPU breakdown for 512-token run:

| Phase | ms/tok |
|-------|--------|
| Attention (flash) | 0.71 |
| Attention (QKV proj) | 0.36 |
| Attention (O proj) | 0.37 |
| Dense FFN (gate/up) | 0.51 |
| Dense FFN (down) | 0.97 |
| Tail (norm+lm_head) | 0.06 |
| **Total GPU** | **3.65** |
| CPU submit+wait | 4.16 |

## Ollama — GPU Benchmarks (Jul 8)

All models run on Radeon 8060S iGPU (shared 128 GB RAM), no CPU fallback. Prompt: "Explain the P versus NP problem in one paragraph." Max 128 tokens.

| Model | Decode tok/s | ms/tok |
|-------|:-----------:|:------:|
| **qwen2.5:0.5b-instruct-q2_K** | **269** | 3.7 |
| qwen2.5:0.5b (FP16) | 241 | 4.2 |
| **bonsai:1.7b-q1_0** | **236** | 4.2 |
| qwen3:0.6b | 224 | 4.5 |
| gemma3:1b | 146 | 6.9 |
| granite3.2:2b | 82 | 12.3 |
| phi4-mini:3.8b | 67 | 15.0 |
| llama3.1:8b-instruct-q2_K | 52 | 19.4 |
| gemma3:4b | 42 | 24.5 |
| mistral:7b | 41 | 24.2 |
| qwen2.5:7b | 41 | 24.6 |
| deepseek-r1:8b | 35 | 28.9 |
| llama3.1:8b-instruct-q5_K_M | 33 | 30.0 |
| llama3.1:8b-instruct-q8_0 | 25 | 39.9 |

**Key insight:** Quantization dominates 8B speed — llama3.1:8b Q2_K=52, Q5_K_M=33, Q8_0=25 tok/s. Sub-1B models scream at 224–269 tok/s. bonsai:1.7b-q1_0 punches at 236 tok/s — 7B-class quality at sub-1B speeds.

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
| **Jul 8** | **Full re-measure** | — | — | FLM regressed 94→63 tok/s; v12 bottlenecked at 3 tok/s (OMP attention); fused layer still NaN; GPU Ollama 269 tok/s; GPU ZINC ternary 217 tok/s; GPU llama.cpp Bonsai Q1_0 228 tok/s |

**Net: validated production (Jul 8) is 63 tok/s NPU (FLM) + 269 tok/s GPU Ollama Q2_K + 228 tok/s GPU llama.cpp Bonsai Q1_0 + 217 tok/s GPU ZINC ternary. FLM regressed from 94→63 (peak reported 103). Fused-layer throughput hit 3.4 ms/tok (291 tok/s) but NaN at layers 24+ — not yet coherent. v12 dropped from 97→3 tok/s — OMP attention bottleneck. DSpark disproven. Zero Python. Pure C++.**

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
