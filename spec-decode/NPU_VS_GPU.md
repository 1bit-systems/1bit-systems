# NPU vs GPU: Head-to-Head Benchmark

**Date:** July 6, 2026  
**Hardware:** AMD Strix Halo (XDNA2 NPU + Radeon 8060S GPU)

All benchmark numbers live in **[`docs/wiki/performance.md`](../docs/wiki/performance.md)** — the single source of truth for this project. This file only tracks architecture-specific notes and implementation context.

---

## Quick Summary

| Winner | Category | NPU | GPU |
|--------|----------|-----|-----|
| 🏆 **NPU** | Speed + Efficiency | **572 tok/s @ 15W (38.1 tok/J)** | 381 tok/s @ 45W (8.5 tok/J) |
| 🏆 **GPU** | Model size | 0.6B max | Up to 9B |

---

## DSpark — Production Notes

DSpark is the production speculative decoding engine:

- **5.90× speedup** — fused layer (291 tok/s) → **572 tok/s**
- **Lossless quality** — rejection sampling, identical target output
- **15W total** — NPU fused target + CPU 5-layer draft model
- **38.1 tok/J** — 4.5× more efficient than fastest GPU 1-bit path

### Key files

| File | Purpose |
|------|---------|
| `draft/dspark_draft.h` | DSpark 5-layer draft model (746 lines, production) |
| `engine/spec_decode.h` | Speculative decode orchestrator |
| `scripts_local/export_dspark_weights.py` | safetensors → flat binary converter |

---

## Source of Truth

- Engine benchmarks: [`docs/wiki/performance.md`](../docs/wiki/performance.md)
- DSpark implementation: `draft/dspark_draft.h`, `engine/spec_decode.h`
- GPU GPU 1-bit benchmarks: [`docs/wiki/performance.md`](../docs/wiki/performance.md)
