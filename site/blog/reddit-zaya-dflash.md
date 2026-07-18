# Draft: Reddit Post — Showcasing 1bit.systems

## Target: r/LocalLLaMA

## Title

> Model-agnostic C++ inference engine for AMD Strix Halo, now with speculative decoding research (DSpark/DFlash) — 1bit.systems is open source

## Body

We've been building **1bit.systems** — an open-source (MIT) inference engine for AMD Strix Halo laptops.

What it does: a single C++ binary auto-detects any GGUF model's architecture and quantization, then routes it to whichever backend can actually run it — GPU (ROCm HIP / Vulkan), NPU (via FastFlowLM), or CPU. No config files, no model registry. Zero Python at runtime.

**[GitHub](https://github.com/bong-water-water-bong/1bit-systems) · [Install](https://1bit.systems/) · [Benchmarks](https://1bit.systems/bench)**

The numbers so far (validated, `site/benchmarks.json`):

- **11 GGUF quant formats** supported (Q4_0 through BF16), each dequantizer checked bit-exact against an independent reference
- **64 tok/s** GPU ROCm HIP kernel-level, **57 tok/s** NPU via FastFlowLM
- **30 tok/s** end-to-end on a real 27B Q4_K model, real prompt — not a synthetic microbenchmark
- **42.21 TFLOPS** raw INT8 GEMM prefill
- Model catalog includes Zamba2 (0.71B–7B), ZR1-1.5B, and Zaya1-8B, fine-tuned and validated on Strix Halo
- deb, snap, AppImage, Homebrew, AUR, Docker, Ollama, and curl\|bash packaging

We're honest about where we're behind, too: `llama.cpp` on the same hardware hits 229 tok/s end-to-end. We're not claiming to beat it — we're closing the gap, and the README says so instead of only publishing the numbers that look good.

### Speculative decoding research (new)

We're actively working on Multi-Token Prediction (MTP) speculative decoding for the NPU+GPU stack — draft a small model's tokens in parallel, verify them all in one forward pass on the target model, accept what matches. Two draft-model configs in active development in `spec-decode/configs/`:

- **DSpark** — 5 draft layers, full Markov head + confidence head
- **DFlash** — same 5-layer backbone, no Markov head, no confidence head, CE-only loss — cheaper to train, simpler acceptance logic

(DFlash is named after — and was built with reference to — [z-lab/dflash](https://github.com/z-lab/dflash), "Block Diffusion for Flash Speculative Decoding"; go check out their work if you haven't seen it.)

Both target Qwen3-0.6B as the draft, `block_size=7`. Still research-stage — not in the shipped inference path yet.

~48M Strix Halo APUs shipped in 2026 (per AMD). Every one has an NPU sitting idle. We open-sourced the engine so anyone can use theirs.

---

**Links**: [GitHub](https://github.com/bong-water-water-bong/1bit-systems) · [Install](https://1bit.systems/) (`curl -sL https://1bit.systems/install.sh | bash`)
**DFlash (external, inspiration for our draft config)**: [github.com/z-lab/dflash](https://github.com/z-lab/dflash) · [github.com/z-lab](https://github.com/z-lab)
