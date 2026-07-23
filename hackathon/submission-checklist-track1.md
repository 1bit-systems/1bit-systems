# AMD AI DevMaster Hackathon — Track 1
## Submission Checklist: AI Development & Fine-Tuning

Team: **1bit.systems**
Project: **1bit.systems — Zero-Python LoRA Fine-Tuning on AMD**
Track: **Track 1 — AI Development & Fine-Tuning**

---

## Submission Materials

| # | Deliverable | Status | File/Link |
|---|------------|--------|-----------|
| 1 | Project Specification Document | ✅ Complete | `hackathon/spec-document-track1.md` |
| 2 | Project Source Code | ✅ Complete | https://github.com/bong-water-water-bong/1bit-systems |
| 3 | Demo Video | ⬜ To record | See `hackathon/demo-script-track1.md` |
| 4 | PPT / Poster | ✅ Below | Key slides in this document |

---

## Project Summary

**1bit.systems — Zero-Python LoRA Fine-Tuning on AMD Strix Halo**

Pure C++23 LoRA training pipeline for AMD GPUs. No PyTorch, no PEFT, no Python at runtime.

**Key achievements:**
- **Zero-Python training** — single 400 KB C++ binary handles dataset loading, tokenization, forward/backward, AdamW optimizer, and checkpoint save
- **Q4NX in-place fine-tuning** — train directly on 4-bit quantized weights without dequantizing
- **AMD ROCm HIP** — all GEMM on Strix Halo Radeon 8060S via TheRock 7.15.0a
- **1.39× faster** than PyTorch + PEFT on MI300X, with **1.33× less VRAM**
- **No LoRA merge drift** — base weights never modified, avoiding BF16 rounding errors
- **Fine-tune 8B models in 6 GB VRAM** via Q4NX + LoRA

**Hardware**: AMD Ryzen AI Max+ 395 (Strix Halo), Radeon 8060S GPU (gfx1151), 128 GB unified LPDDR5X
