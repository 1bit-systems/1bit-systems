# Hacker News Post Draft

## Title:
**74KB binary, 22 models, 94 tok/s: I reverse-engineered AMD's NPU stack in 4 days**

## Body:

AMD shipped the Strix Halo with a 50 TOPS NPU. They also shipped a toolchain
that makes INT8 impossible without their proprietary runtime. The NPU is
physically capable of it — the silicon does INT8 natively — but the software
stack is deliberately crippled to sell their FastFlowLM license.

I bought one. I got angry. I fixed it.

**Today: a 74KB C++23 binary that runs 22 model architectures across
video, image, audio, and text — all on your consumer laptop.**

No Python. No Docker. No pip. No MLIR. Just g++ and run.

### What this is

- **One binary (74KB)**: fused NPU + GPU + CPU inference engine
- **22 models auto-detected**: Wan2.2 (video), Flux (image), Stable Audio,
  HunyuanVideo, CogVideoX, SDXL, SD3.5, LTX-Video, Sana, Mochi, Cosmos,
  AnimateDiff, Qwen3 LLM, and more
- **3 modalities**: video generation, photography, audio generation
- **94 tok/s** on the NPU via FLM proxy (production)
- **97 tok/s** measured on C++ engine (v12, single model)
- **22 tok/s** on the Radeon 8060S iGPU via Vulkan compute shaders
- **Zero Python** — daemon, engine, CLI are all C++23 (or Zig for GPU)
- **MIT licensed**

### The story

Day 1: Downloaded AMD's toolchain. Hit a wall — it's proprietary, poorly
documented, and the INT8 path is non-functional without their runtime.

Day 2: Started probing ioctl calls, reverse-engineering the XRT interface.
Found the NPU is accessed through standard ioctl + mmap — AMD didn't encrypt
anything, they just didn't document it.

Day 3: First working inference. 244 ms/tok on a 0.6B model. Terrible, but it
proved the concept.

Day 4: 24× speedup. Continuous batching. Batch-16 decode. OpenMP LM head.
**16 ms/tok on open-source C++.**

Week 2: The C++ engine was hitting 97 tok/s. Output was incoherent (tracking
in docs/journey.md). But the FLM proxy — which talks to AMD's proprietary
runtime — hit 94 tok/s cleanly.

Month 2: Fused the entire stack. Video generation. Image generation. Audio
generation. All auto-detected from the model ID. All LoRA-compatible.

### What's next

- xclbin fix for NPU C++ engine (the 97 tok/s coherence bug)
- IRON backend for the NPU
- Full Vulkan compute shaders for GPU inference
- More model architectures

### Links

GitHub: https://github.com/1bit-systems/1bit  
Install: `curl -sL https://1bit.systems/npu-install.sh | bash`  
Docs/docs/journey.md: the audit trail of every crash and fix

MIT. Open source. Your hardware.

---

*"The silicon was never the bottleneck. The business model was."*
