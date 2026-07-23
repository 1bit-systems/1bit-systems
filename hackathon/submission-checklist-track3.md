# AMD AI DevMaster Hackathon — Track 3
## Submission Checklist: AI Acceleration & Performance

Team: **1bit.systems**
Project: **1bit.systems — Open-Source AMD XDNA 2 NPU + ROCm GPU Kernels**
Track: **Track 3 — AI Acceleration & Performance**

---

## Submission Materials

| # | Deliverable | Status | File/Link |
|---|------------|--------|-----------|
| 1 | Project Specification Document | ✅ Complete | `hackathon/spec-document-track3.md` |
| 2 | Project Source Code | ✅ Complete | https://github.com/bong-water-water-bong/1bit-systems |
| 3 | Demo Video | ⬜ To record | See `hackathon/demo-script-track3.md` |
| 4 | PPT / Poster | ✅ Below | Key slides in this document |

---

## Project Summary

**1bit.systems — Open-Source AMD XDNA 2 NPU Stack + Custom ROCm GPU Kernels**

Full reverse-engineering of AMD's proprietary XDNA 2 NPU stack + custom ROCm HIP kernels for LLM inference.

**Key achievements:**
- **Reverse-engineered XDNA 2 NPU in 4 days** — 22 proprietary `.so` libraries disassembled, 209 xclbin bitstreams traced, 87.8 MB closed binary → 17.5 MB open-source
- **NPU v12: 97 tok/s** — beats FLM Kraken Point by 46%
- **First open-source Mamba1 GPU backend** — BlackMamba 1.5B at 79.8 tok/s
- **Fused ternary kernels at 415 tok/s** (Q1 GEMV) and 318 tok/s (Vulkan ZINC)
- **Token Router** — per-layer dispatch across NPU + GPU + CPU with auto-failover
- **TheRock 7.15.0a** — first project to adopt and validate AMD's nightly HIP SDK

**Hardware**: AMD Ryzen AI Max+ 395 (Strix Halo), Radeon 8060S GPU (gfx1151), 32 XDNA 2 NPU tiles, 128 GB unified LPDDR5X
