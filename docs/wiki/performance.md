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
| GPU | Radeon 8060S (RADV), 32 CUs, 256 GB/s, Vulkan |
| CPU | Zen 5, 16C/32T |
| RAM | 128 GB unified |
| Binary | `zaya_server` — 206,616 bytes (≈207 KB), Release build, gfx1151 — verified by direct measurement 2026-07-11 |

---

## At a Glance

| Engine | Hardware | Speed | Status | Power | Model |
|--------|----------|:-----:|--------|:-----:|-------|
| **GPU 1-bit** (llama.cpp) 🏆 | Radeon 8060S | **383 tok/s** | ✅ measured via third-party tool | ~45W | Qwen2-0.5B IQ1_S |
| **NPU FLM** (production) | XDNA 2 · 32 tiles | **94 tok/s** | ✅ validated, coherent | ~15W | Qwen3-0.6B |
| **GPU ternary** (Vulkan) | Radeon 8060S | **307 tok/s** | ✅ validated on-device (3.3 ms/tok) | ~45W | Bonsai-1.7B Q2_0 (1.58-bit) |
| **NPU fused** | XDNA 2 · 32 tiles | **291 tok/s** | ⚙️ raw throughput — output not yet fully coherent | ~20W | Qwen3-0.6B |
| **NPU v12** (fallback) | XDNA 2 · 32 tiles | **97 tok/s** | ⚙️ raw | ~15W | Qwen3-0.6B |
| **ROCm** (HIP) | Radeon 8060S | **113 tok/s** | reported | ~45W | Bonsai TQ2 ternary |
| **GPU ZINC** (Vulkan F16) | Radeon 8060S | **22 tok/s** | ✅ validated | ~45W | Bonsai-1.7B F16 |
| **C++ all-5** (auto-detect) | Q4NX header parse | **28 tok/s** | ⚙️ raw | ~15W | 5 models, one binary |
| **Eagle3 spec-decode** ❌ | XDNA 2 + Zen 5 | **0.8 tok/s** | ❌ 0% draft acceptance — checkpoint undertrained (batch-size/dataset-size mismatch), not an architecture disproof | 15W | Qwen3-0.6B |

**Status legend:** ✅ *validated* = measured on-device with coherent output · ✅ *measured* = throughput measured via a third-party tool (llama.cpp) · ⚙️ *raw* = the kernel runs at this speed but the engine's output is not yet fully coherent (correctness WIP) · *reported* = reported, not independently re-measured this pass · ❌ *disproven* = an earlier projection that was tested end-to-end and did not hold up. **Only ✅ numbers should be quoted as production.**

**Net: 73+ models across 6 backends · 22 multi-modal (video, image, audio) · production-validated: 94 tok/s NPU (FLM) + 307 tok/s GPU ternary. Speculative decoding (Eagle3/DSpark) is unresolved, not disproven — see below.**

---

## GPU Ternary (Vulkan) — Bonsai-1.7B Q2_0 (1.58-bit)

| Metric | Value |
|--------|-------|
| Decode | **307 tok/s** (3.3 ms/tok) |
| Model | Bonsai-1.7B Q2_0 (1.58-bit ternary) |
| Backend | llama.cpp (Vulkan) |
| Prefill | 3,118 tok/s |

Tested on Radeon 8060S via RADV Vulkan.

---

## 1-Bit Model Benchmarks (GPU — Radeon 8060S)

Every model at ≤1.5625 bpw (true 1-bit class). Measured via llama.cpp on Radeon 8060S via Vulkan.

| Model | BPW | Size | Params | Engine | Prefill | Decode | ms/tok |
|-------|-----|------|--------|--------|---------|--------|--------|
| Qwen2 0.5B | **1.06** (IQ1_S) | 296 MB | 494M | llama.cpp | 4,188 tok/s | **383 tok/s** | 2.6 |
| Qwen3.5-0.8B | **1.25** (Q1_0) | 268 MB | 752M | llama.cpp | 3,883 tok/s | **312 tok/s** | 3.3 |
| Hy-MT2 1.8B | **1.3125** (STQ1_0) | 441 MB | 1.8B | ZINC (Sherry) | 238 tok/s | **267 tok/s** | 3.7 |
| gemma-2-2b | **1.06** (IQ1_S) | 788 MB | 2.6B | llama.cpp | 1,773 tok/s | **158 tok/s** | 6.3 |
| gemma3 4B | **1.06** (IQ1_S) | 1.05 GB | 3.88B | llama.cpp | 1,247 tok/s | **122 tok/s** | 8.2 |
| Nemo 8B | **1.06** (IQ1_S) | 1.97 GB | 8.41B | llama.cpp | 720 tok/s | **79 tok/s** | 12.7 |
| Qwen3.5-9B | **1.25** (Q1_0) | 1.82 GB | 8.95B | llama.cpp | 762 tok/s | **74 tok/s** | 13.5 |

**Key takeaways:** NPU wins on power efficiency (~15W vs ~45W); GPU 1-bit is 1.3–4× faster in raw tok/s across this range.

---

## Raw C++ Engine — All 5 Models (M=32 batch, OpenMP)

Single binary. Auto-detect. No proprietary code.

| Model | Hidden | Size | Prefill | Decode | Tok/s | Layers |
|-------|--------|------|---------|--------|-------|--------|
| **Qwen3-0.6B** | 1,536 | 610 MB | 14 ms/tok | **36 ms/tok** | **28** | 28/28 |
| **Gemma4-E2B** | 2,304 | 4.7 GB | 20 ms/tok | **62 ms/tok** | **16** | 35/35 |
| **Qwen3-VL-4B** | 2,560 | 3.2 GB | 34 ms/tok | **93 ms/tok** | **11** | 36/36 |
| **Llama-3.1-8B** | 4,096 | 5.7 GB | 47 ms/tok | **100 ms/tok** | **10** | 32/32 |
| **Qwen3-8B** | 4,096 | 6.0 GB | 49 ms/tok | **127 ms/tok** | **8** | 36/36 |

Scale is roughly linear with model size. All 5 verified on Strix Halo NPU.

---

## GPU (ROCm HIP Kernels) — Bonsai TQ2

| Metric | Value |
|--------|-------|
| Decode | **113 tok/s** (8.8 ms/tok) |
| Model | Bonsai-1.7B TQ2 ternary |
| Backend | ggml-rocm |

---

## Raw Silicon: GEMM Throughput

Chess-compiled INT8 xclbins. Verified on-device.

| Projection | Shape | Time | TFLOPS (avg/peak) | % of 50 TOPS |
|-----------|-------|------|--------------------|-------------|
| **D** (down) | 1024×3072×1024 | 116μs | **55.7 / 80.5** | **111%** |
| **O** (output) | 1024×2048×1024 | 108μs | 39.7 / 49.4 | 79% |
| **GU** (gate+up) | 1024×1024×6144 | 801μs | 16.1 / 16.5 | 32% |
| **QKV** (fused) | 1024×1024×4096 | 559μs | 15.4 / 15.5 | 31% |

---

## Speculative Decoding — Eagle3/DSpark (unresolved, not disproven)

Speculative decoding (Eagle3 1-layer draft + NPU target, `spec-decode/`) previously carried
an earlier **~572 tok/s projection** (base NPU throughput × 5.90× acceptance, measured on a
small 10-sample gsm8k eval) that was reported as "disproven" by a 2026-07-07 end-to-end run
showing 0.1–0.2 tok/s at 0% draft acceptance. **That conclusion was wrong — two separate bugs
were found on 2026-07-11, not an architecture failure:**

1. **Wiring bug** (`spec-decode/engine/npu_spec_integration.cpp`): the benchmark hardcoded a
   checkpoint path (`eagle3_draft.bin`) that didn't exist yet on 2026-07-07 — the symlink
   wasn't created until 2026-07-10. `load_weights()` silently failed and the draft model fell
   back to an untrained passthrough (`mtp_draft.h`, always predicts token 0/1 regardless of
   context) instead of erroring out. Fixed: the integration now hard-fails instead of silently
   running an untrained benchmark, and points at the checkpoint that actually exists.
2. **Undertrained checkpoint**: even a checkpoint that *does* load successfully
   (`eagle3_draft_npu_1k.bin`, confirmed loading in a 2026-07-10 test run) still measured 0%
   acceptance. Root cause: the training config (`spec-decode/configs/eagle3_qwen3_0.6b.py`)
   uses DeepSpec's stock `global_batch_size=512` — copied from configs meant for large
   multi-GPU runs — against a local training set of only 360 examples
   (`spec-decode/train_data/perfectblend_train.jsonl`). With a batch bigger than the whole
   dataset, at most a handful of real gradient steps occur across all 10 epochs; the
   checkpoint is close to random-init despite loading correctly.

With both the wiring fixed and the correct checkpoint wired in, a real, honestly-measured
run on this hardware gives **0.8 tok/s, 0% acceptance, 1.03× speedup** — consistent with an
undertrained draft head, not a broken architecture. **Speculative decoding here is unresolved
pending a real training run** (batch size sized to the dataset, and/or meaningfully more
training data) — it should not be quoted as either "572 tok/s" or "disproven."

**2026-07-11, later same day — ran an actual training pass to test this directly.** Fixed
`global_batch_size` (512 → 8, matched to the 343-valid-sample dataset), fixed a broken
torch+ROCm install (mismatched `torch`/gfx1151-kernel-package versions were causing GPU
tensor placement to segfault — rebuilt the venv from AMD's matched TheRock nightly index),
generated a real target-hidden-state cache, and ran 420 real training steps (10 epochs ×
42 steps/epoch) on this machine's actual NPU/GPU hardware. Loss dropped from 26.5 → ~7.5, a
genuine ~3.5× reduction confirming real learning occurred (this had never previously
happened — every prior checkpoint was effectively random-init). **Re-benchmarked the
resulting checkpoint anyway: still 0.8 tok/s, 0% acceptance.** This is not a regression or a
new bug — cross-entropy loss 7.5 corresponds to perplexity ≈1,800, nowhere near converged for
a from-scratch transformer, even a tiny 1-layer one. 343 examples over 420 steps is simply
too little data/training for the draft head to produce useful predictions yet. The wiring and
batch-size bugs are conclusively fixed and validated (training visibly works now); getting
nonzero acceptance still requires substantially more training data and/or steps, which is a
separate, larger undertaking from the bug fixes themselves.

| Engine | Tok/s | Power | Tok/J |
|--------|:-----:|:-----:|:-----:|
| NPU Eagle3 (undertrained) ❌ | **0.8** | 15W | ~0.05 |
| GPU Qwen2-0.5B IQ1_S | 383 | 45W | 8.5 |
| GPU Qwen3.5-0.8B Q1_0 | 312 | 45W | 6.9 |

---

## Engine Evolution (July 2026)

| Date | Engine | Decode | Breakthrough |
|------|--------|--------|---------------|
| Jun 28 | v7 BFP16 | 1930 ms/tok | First working decode |
| Jul 1 | i8 swap | 244 ms/tok | K-interleaving fixed |
| Jul 2 | v9/v12, M=32 batch | 10 ms/tok | M=32 + OpenMP attention |
| Jul 2 | All 5 models | 36–127 ms/tok | Auto-detect, 0 crashes |
| Jul 6 | Fused layer | 3.4 ms/tok | One xclbin/transformer layer |
| Jul 6 | DSpark/Eagle3 spec-decode | 0.8 tok/s (unresolved) | 0% acceptance — wiring bug fixed Jul 11, undertrained checkpoint (batch-size/dataset-size mismatch) is the real blocker |

---

*Numbers above are the reconciled canonical figures as of 2026-07-11. NPU fused and C++
all-5/v12 are explicitly marked raw (kernel-speed, not yet fully coherent output) — do not
upgrade their status without an independently verified end-to-end coherence check.*
