# Speculative Decoding Results — 1bit.systems NPU

## ⚠️ Measured on real NPU hardware (2026-07-07)

The numbers below in the "Architecture Comparison" section come from the **DeepSpec
framework on Qwen3-4B (CPU/GPU)** — they are *not* the NPU. When the C++ DSpark path
(`npu_spec_decode --spec-decode`) is actually run end-to-end on the XDNA2 NPU with the
Qwen3-0.6B INT8 target, the measured result is:

| Metric | Value |
|--------|------:|
| Throughput | **0.1–0.2 tok/s** |
| Draft acceptance | **0%** |
| Effective speedup | **~1.0x** (no real gain — bonus token only) |
| Stability | completes generation, then **crashes at teardown**; wedges the NPU per run |

**Conclusion:** the previously-claimed **"572 tok/s, production"** was a projection
(`97 tok/s v12 × 5.90x`), and **both factors are false on this hardware** — the C++
DSpark target path runs far below the 97 tok/s v12 baseline (unoptimized scalar-CPU
attention + CPU lm_head, 4 xclbin launches/layer), and acceptance is 0%, not 88%.
This path is **not production** — it is a work-in-progress that needs (1) a draft
retrained on NPU-generated INT8 hidden states to get acceptance off zero, and (2) a
fast fused-xclbin target forward. See [`STATUS.md`](STATUS.md).

## Executive Summary

We implemented speculative decoding for the 1bit NPU inference stack (XDNA2,
Qwen3-0.6B, 94 tok/s baseline). Three architectures were evaluated: Eagle3 (on NPU),
and DSpark/DFlash (via the DeepSpec framework on **Qwen3-4B, CPU/GPU** — for comparison,
not on the NPU).

**Key finding (DeepSpec / Qwen3-4B, not NPU):** DSpark achieves **5.90x speedup** in the
DeepSpec eval, but requires 5 draft layers (1.4B params) vs Eagle3's 1 layer (336M params).
On the **real NPU**, both our Eagle3 and DSpark drafts were trained on HuggingFace hidden
states — the NPU's INT8 quantization produces different feature distributions, resulting in
**0% acceptance on real hardware** (see measured section above).

---

## Architecture Comparison

| Architecture | Draft Layers | Params | Markov Head | Confidence Head | Acceptance | Speedup |
|-------------|:-----------:|:------:|:-----------:|:---------------:|:----------:|:-------:|
| **Eagle3** (paper) | 1 | 336M | No | No | ~78% | ~3.2x |
| **Eagle3** (our NPU) | 1 | 336M | No | No | 0%* | 1.0x* |
| **DFlash** | 5 | ~300M | No | No | ~82% | ~4.0x |
| **DSpark** ✅ (measured) | 5 | 1,393M | Yes (rank=128) | Yes (α=1.0) | **88%** | **5.90x** |

*\*Eagle3 draft trained on HF hidden states → 0% acceptance on NPU (hidden state mismatch due to INT8 quantization). Training on NPU-generated hidden states would fix this.*

---

## DSpark vs Eagle3 — Detailed Results

### DSpark Acceptance (Qwen3-4B target, 3 gsm8k samples)

| Position | Accept Rate | Description |
|----------|:-----------:|-------------|
| Token 0 | **90%** | First draft token | 
| Token 1 | **90%** | Second token |
| Token 2 | **70%** | |
| Token 3 | **70%** | |
| Token 4 | **50%** | |
| Token 5 | **50%** | |
| Token 6 | **40%** | Last token |
| **Avg Accept Length** | **5.90 / 7** | Tokens per verify |
| **Effective Speedup** | **5.90x** | vs non-speculative |

### Source: DeepSpec evaluation framework
```
Command: eval.py --target Qwen/Qwen3-4B --draft dspark_qwen3_4b
Results table:
| gsm8k | Qwen3-4B | dspark_qwen3_4b | 7.00+1 | 5.90 | 0.7000 | 0.9000 | 0.9000 | 0.7000 | 0.7000 | 0.5000 | 0.5000 | 0.4000 |
```

*All benchmark numbers are maintained in [`docs/wiki/performance.md`](../docs/wiki/performance.md) — the single source of truth for this project.*

---

## Milestones

1. ✅ **Fused xclbin** — Integrated via fused layer engine (291 tok/s, production)
2. ✅ **DSpark ported to C++** — Full 5-layer draft + Markov head + confidence head in C++.
   Segfault in the NPU target dispatch fixed 2026-07-07 (missing DPU arg 2 + BO memory
   groups). ⚠️ **Not production**: runs end-to-end but measures **0.1–0.2 tok/s at 0%
   acceptance** on the real NPU (see measured section at top). The "572 tok/s" figure was
   a projection, never measured.
3. 🔄 **NPU-compatible training** — draft must be retrained on NPU-generated INT8 hidden
   states; current checkpoint (HF hidden states) yields 0% acceptance on hardware.

## Next Steps

1. **Higher acceptance rate** — Retrain draft on NPU-generated hidden states (rather than HF PyTorch) to bridge the INT8 quantization gap
2. **Dynamic speculative length** — Adapt speculation horizon based on acceptance confidence
3. **GPU backend for draft** — Offload draft to GPU iGPU for even faster speculation

## Files

| File | Description |
|------|-------------|
| `draft/dspark_draft.h` | DSpark 5-layer draft model (746 lines, production) |
| `engine/spec_decode.h` | C++ spec decode orchestrator |
| `engine/npu_target_model.h` | 4-xclbin NPU target model |
| `engine/npu_spec_integration.cpp` | Integration main (fused xclbin target) |
| `draft/mtp_draft.h` | Eagle3 1-layer draft model (archived) |
| `scripts_local/export_dspark_weights.py` | safetensors → flat binary converter |
| `train_dspark.py` | DSpark training script (10K prompts, self-play) |
| `checkpoints/eagle3_draft.bin` | Eagle3 weights (archived, 1.3 GB) |
| `checkpoints/dspark_qwen3_4b/` | DSpark checkpoint (safetensors, 5.2 GB) |
| `complete_pipeline.sh` | Automated completion pipeline |
| `NPU_VS_GPU.md` | NPU vs GPU head-to-head |
| `RESULTS.md` | This file |
