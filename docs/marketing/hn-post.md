> **📜 Historical draft** — This is an archived marketing post from an earlier phase. Numbers and references may be stale.
>
# Show HN: 291 tok/s NPU fused inference — one C++ binary, 35 models, 7 backends, zero Python

https://github.com/bong-water-water-bong/1bit-systems

**One ~400 KB binary auto-detects your model and dispatches to the fastest backend available.** NPU XDNA 2, GPU ternary (Vulkan), ROCm HIP, CPU — all in the same binary. No Python. No config. No Docker.

Recent progress:

**NPU attention kernel is now online.** The XDNA 2 AIE array runs full Q@K^T softmax attention on-NPU — score computation, online softmax rescaling, weighted sum. No CPU involvement for the attention step. Enabled via `NPU_ATTN=1`. Handles up to 16-token context windows; longer contexts fall back to OpenMP.

**Mamba2 HIP GPU kernel implemented.** Selective scan + conv1d fused for single-token decode on AMD ROCm. Zamba2 models now run their SSM blocks entirely on GPU instead of falling back to CPU.

| Backend | Tok/s | Notes |
|---------|:-----:|-------|
| Q1 GEMV kernel | 417 | validated |
| Fused TQ2 | 415 | validated |
| GPU ternary (Vulkan) | 318 | validated |
| NPU fused v12 | 97 | optimized |
| ROCm HIP | 64 | validated |
| Speculative decode | 8x | 100% accept rate in test |

**Speculative decoding** also works — draft-verify with MTP head, 8x speedup in the bench. Already built, tests pass.

**What's under the hood:**
- Pure C++17, zero Python at runtime, MIT license
- 35 supported model architectures (Qwen, Llama, DeepSeek, BitNet, Mamba2, etc.)
- 11 GGUF quant formats, each dequantizer bit-exact verified
- Auto-detects model arch from GGUF header — no config files, no flags
- Full engineering journal at `docs/journey.md` — every bug documented
- Runs on AMD Strix Halo (Ryzen AI Max) — ~48M APUs shipped this year

```
curl -sL https://1bit.systems/install.sh | bash
```

MIT. Your hardware, your model, your choice of backend.
