# Draft: Reddit Post — Showcasing 1bit.systems

## Target: r/LocalLLaMA

## Title

> Zero Python, 120KB binary, 94 tok/s on a laptop NPU — 1bit.systems is open source

## Body

We've been building **1bit.systems** — an open-source (MIT) inference engine for AMD Strix Halo laptops.

What it does: one 120KB C++ binary auto-detects 5 models and runs at **94 tok/s** on the built-in XDNA 2 NPU. Zero Python, zero dependencies, single g++ command to build.

**[GitHub](https://github.com/bong-water-water-bong/1bit-systems) · [Install](https://1bit.systems/) · [Benchmarks](https://1bit.systems/bench)**

The numbers so far:

- **5 models** from one binary: Qwen3-0.6B/8B, Qwen3-VL-4B, Llama-3.1-8B, Gemma4-E2B
- **94 tok/s** decode (FLM proxy) · **97 tok/s** open-source C++ engine
- **55.7 TFLOPS** raw INT8 GEMM — exceeding AMD's 50 TOPS spec
- **281 tok/s** 1-bit inference on the GPU side (Zig, Vulkan)
- **24× speedup** in one dev session (244 → 10 ms/tok)
- deb, snap, docker, ollama, curl|bash packaging

We're also connecting this to **dflash** for speculative decoding ([DFlash: Block Diffusion for Flash Speculative Decoding](https://github.com/z-lab/dflash), 5.3K⭐ from UMD). The idea: use the NPU's ultra-low-power inference to accelerate larger models on the GPU.

~48M Strix Halo APUs shipped in 2026 (per AMD). Every one has an NPU sitting idle. We open-sourced the engine so anyone can use theirs.

---

**Links**: [GitHub](https://github.com/bong-water-water-bong/1bit-systems) · [Install](https://1bit.systems/) (`curl -sL https://1bit.systems/install.sh | bash`) · [Discord](https://discord.gg/dSyV646eBs)  
**dflash**: [github.com/z-lab/dflash](https://github.com/z-lab/dflash) · [github.com/z-lab](https://github.com/z-lab)
