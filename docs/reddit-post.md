# Reddit r/LocalLLaMA Post Draft

## Title:
**Model-agnostic C++ inference engine for AMD Strix Halo — drop in any GGUF, it routes to NPU/GPU/CPU automatically**

## Body:

**tl;dr**: single C++ binary, no Python at runtime, auto-detects any GGUF
model's architecture/quantization and routes it to whichever of NPU
(via FastFlowLM), GPU (ROCm HIP / Vulkan), or CPU can actually run it. MIT.

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
cmake -B build -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build --target zaya_server -j8
./build/zaya_server --model /path/to/model.h1b
```

### Why this exists

Bought a Strix Halo laptop for the NPU. Found the official stack ties you to
one proprietary model format and doesn't let you mix NPU/GPU/CPU in the same
request. So I built a router that reads a model's own on-disk metadata and
picks a backend — no manifest files, no per-model config.

### What's actually in it (verified, not vibes)

| Piece | Detail |
|-------|--------|
| **Quant format support** | Q4_0/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K/Q2_K/Q3_K/Q8_K/BF16 — each dequantizer checked bit-exact against the independent `gguf` Python package |
| **CPU reference backend** | Llama/Mistral/Qwen2/Qwen3/Gemma/Phi, incl. MoE routing + Qwen3 Q/K-norm, verified against an independent numpy forward pass |
| **GPU backend** | ROCm HIP kernels + a Vulkan (ZINC) path |
| **NPU backend** | Delegates to FastFlowLM — see "the catch" below for why |
| **Video generation** | `tools/video-lora/` — Wan2.2, LTX-Video, AnimateDiff, CogVideoX, Stable Video Diffusion, with LoRA support |
| **Dependencies at runtime** | 0. No Python, no pip, no Docker |

### The catch — said plainly

The project's own in-process NPU engine (`engine/npu/`) has a **confirmed
GEMM kernel correctness bug** on real hardware — not "needs tuning," actually
produces wrong output. That's why the default NPU path is FastFlowLM (external
subprocess, already correct) instead of our own kernel. It's disclosed in the
README, not something you find out after building it.

Real numbers, current as of this post (`site/benchmarks.json`):

| Backend | tok/s | Status |
|---------|:-----:|--------|
| ROCm HIP (kernel-level) | 64 | validated |
| NPU via FastFlowLM | 57 | validated |
| GPU Vulkan (ZINC) | 22 | validated |
| zaya_server end-to-end, Qwen 27B Q4_K | 30 | real prompt |
| zaya_server end-to-end, Qwen 35B MoE Q4_K | 20 | real prompt |

`llama.cpp` on the same hardware hits 229 tok/s end-to-end. We're behind it
and saying so, rather than publishing a kernel-level microbenchmark next to
it without the disclaimer (we used to do that — a self-filed issue caught it
and the README's been fixed since).

### Links

GitHub: https://github.com/bong-water-water-bong/1bit-systems
Audit trail: `docs/journey.md` — every real bug and fix, including the ones
that were embarrassing

MIT. Your hardware, your model, your choice of backend.
