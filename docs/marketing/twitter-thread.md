> **📜 Historical draft** — This is an archived marketing post from an earlier phase. Numbers and references may be stale.
>
One C++ binary. 35 models. 7 backends. Zero Python. Zero config.

1bit.systems — the NPU inference engine for AMD Strix Halo — just got NPU attention on silicon and Mamba2 HIP GPU kernels. Here's what that means:

---

We reverse-engineered AMD's proprietary NPU stack (22 .so files → 17.5 MB open source) in 4 days. The AIE array on XDNA 2 has 32 compute tiles, and now ALL of them do real work: QKV projection, attention, O/FFN projection — one fused pass through the array.

The attention kernel runs on-NPU: Q@K^T scores, online softmax rescaling, weighted sum. We clocked it. It works. `NPU_ATTN=1` to enable.

---

On the GPU side, the Mamba2 selective scan kernels were already written (conv1d + SSM scan). What was missing was the orchestration — the in_proj/SiLU/x_proj/out_proj matmuls that chain the layer together. Now those launch on GPU too. Zamba2 models run their SSM blocks entirely on ROCm HIP.

---

Speculative decode bench: 8x speedup at 100% accept rate. Draft model is an 8.5M-param MTP head that fits in the NPU's local memory alongside the target model.

---

The full stack in one binary (~400 KB):
• NPU XDNA 2 — fused layer engine (QKV + attn + O + FFN)
• GPU ternary — Vulkan GLSL compute
• GPU ROCm — HIP Mamba2 kernels
• CPU — OpenMP fallback for everything
• Auto-detect from GGUF headers — no flags, no config

---

curl -sL https://1bit.systems/install.sh | bash

MIT. Open source. No Python. No Docker. No BS.

https://github.com/bong-water-water-bong/1bit-systems
