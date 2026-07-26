# FLUX / Stable Diffusion — Feasibility Study

## Summary

Adding image generation (FLUX, Stable Diffusion) to the 1bit-systems engine is feasible but requires significant new infrastructure. This document outlines the architecture, effort, and design decisions.

## Why Add Image Generation?

- **Completes the modality triad**: text (LLM) + speech (Whisper) + vision (ViT/ZAYA1-VL) + **image generation**
- **AMD ROCm native**: FLUX and SD can run entirely on AMD hardware — no NVIDIA dependency
- **NPU opportunity**: Strix Halo NPU could accelerate UNet/transformer denoising steps
- **Unique positioning**: Only engine with NPU-accelerated image generation on AMD hardware

## Architecture Options

### Option A: FLUX (Recommended)

FLUX is a rectified flow transformer by Black Forest Labs. It uses:

- **T5 text encoder** (already supported via existing LLM backends)
- **Transformer denoiser** (double-stream blocks + single-stream blocks)
- **VAE decoder** (for latent → pixel conversion)

**Pros**: MIT license, state-of-the-art quality, transformer-based (leverages existing attention kernels)
**Cons**: Requires VAE (convolutional decoder), T5 encoder runtime

### Option B: Stable Diffusion 3.5

Uses MM-DiT (multi-modal diffusion transformer).

**Pros**: Apache 2.0 license, well-documented
**Cons**: 3 separate text encoders, complex conditioning

### Option C: SDXL (Stable Diffusion XL)

Uses UNet-based denoiser.

**Pros**: Most widely deployed, smallest model size (2.6B UNet)
**Cons**: UNet architecture doesn't map cleanly to existing transformer kernels

## Recommended Approach: FLUX

```
Text prompt → T5 encoder (existing LLM backend) → 
  Transformer denoiser (reuse attention kernels) →
    VAE decoder (new conv2d engine) → output image
```

### Components Needed

| Component | Status | Effort |
|-----------|--------|--------|
| **T5 text encoder** | Can reuse existing LLM backend | 🟢 Low |
| **Transformer denoiser** | Reuses self-attention/cross-attention HIP kernels | 🟡 Medium |
| **VAE decoder** | New conv2d + upsampling engine | 🔴 High |
| **Scheduler** (flow matching) | ~50 lines of math | 🟢 Low |
| **CLIP text encoder** (optional) | Small transformer | 🟢 Low |

### Estimated Timeline

- **Phase 1**: T5 encoder + basic denoising loop (1-2 weeks)
- **Phase 2**: Full FLUX pipeline with CPU fallback (2-3 weeks)
- **Phase 3**: GPU-accelerated VAE + NPU support (3-4 weeks)

## Decision: Defer Until Core Modalities Are Solidified

Image generation is the next frontier but should wait until:
1. Whisper speech-to-text is production-ready (GPU-accelerated)
2. ZAYA1-VL has end-to-end validated text generation
3. The NPU backend is stable for multimodal workloads

**Recommendation**: Begin Phase 1 after P0/P1 items are complete and merged.
