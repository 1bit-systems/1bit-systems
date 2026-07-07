# NPU vs GPU: Head-to-Head Benchmark

**Date:** July 6, 2026  
**Hardware:** AMD Strix Halo (XDNA2 NPU + Radeon 8060S GPU)

All benchmark numbers live in **[`docs/wiki/performance.md`](../docs/wiki/performance.md)** — the single source of truth for this project. This file only tracks architecture-specific notes and implementation context.

---

## Quick Summary

| Winner | Category | NPU | GPU |
|--------|----------|-----|-----|
| 🏆 **GPU** | Speed (measured, coherent) | DSpark 0.1–0.2 tok/s ❌ · fused 291 raw | **381 tok/s @ 45W** (measured) |
| 🏆 **GPU** | Model size | 0.6B max | Up to 9B |

---

## DSpark — Experimental (NOT production)

DSpark's "572 tok/s @ 15W / 38.1 tok/J" was a **projection** (fused 291 × 5.90×
acceptance). The 5.90× came from the DeepSpec eval on **Qwen3-4B (CPU/GPU)**, not
the NPU.

**End-to-end measurement on the NPU (2026-07-07, Qwen3-0.6B INT8 target):**

- **0.1–0.2 tok/s** — the target forward (scalar-CPU attention + CPU lm_head,
  4 xclbin launches/layer) is far slower than the 97 tok/s v12 baseline
- **0% draft acceptance** — the draft (trained on HF FP hidden states) is rejected
  100% of the time against the INT8 NPU feature distribution → no speculation gain
- The dispatch **segfault was fixed** 2026-07-07 (missing DPU arg 2 + BO memory
  groups), but generation still aborts at teardown and wedges the NPU per run

DSpark is a work in progress. It needs (1) a draft retrained on NPU-generated INT8
hidden states, and (2) a fast fused-xclbin target forward, before it is worth quoting.

### Key files

| File | Purpose |
|------|---------|
| `draft/dspark_draft.h` | DSpark 5-layer draft model (746 lines, experimental) |
| `engine/spec_decode.h` | Speculative decode orchestrator |
| `scripts_local/export_dspark_weights.py` | safetensors → flat binary converter |

---

## Source of Truth

- Engine benchmarks: [`docs/wiki/performance.md`](../docs/wiki/performance.md)
- DSpark implementation: `draft/dspark_draft.h`, `engine/spec_decode.h`
- GPU GPU 1-bit benchmarks: [`docs/wiki/performance.md`](../docs/wiki/performance.md)
