# Show HN: One Binary, All Backends — AMD NPU + GPU + Mamba1 SSM, Zero Python

https://github.com/bong-water-water-bong/1bit-systems

Single C++ binary (~1.8 MB). Drop in any GGUF model — it reads the
architecture header and auto-routes to the right backend. No Python,
no PyTorch, no Docker, no config files.

**New this week: Mamba1 GPU backend.** BlackMamba 1.5B at 79.8 tok/s
on the Strix Halo iGPU (ROCm HIP). Alternating SSM + MoE layers,
full autoregressive decode. The Mamba1 kernel code had 3 real correctness
bugs that would have silently produced garbage — all found and fixed
before shipping.

**What it runs today:**

| Architecture | Backend | Throughput |
|---|---|---|
| Mamba1 SSM+MoE (BlackMamba) | HIP GPU | 79.8 tok/s |
| Dense transformer (Qwen2/ZR1) | Vulkan ZINC | ~30 tok/s |
| Ternary (Bonsai TQ2) | HIP GPU | 415 tok/s (kernel) |
| Whatever GGUF you throw at it | Auto-routed | Depends on model |

**9 hardware backends auto-detected:** NPU XDNA2, Mamba1 HIP, ROCm HIP,
Vulkan ZINC, Vulkan raw, CPU AVX-512, CPU scalar, generic GGUF CPU.

**What's under the hood:**
- AMD's closed-source XDNA 2 NPU stack fully reverse-engineered and
  replaced (22 .so → 17.5 MB open source)
- Model-agnostic GGUF loader: 8 architectures, 13 quant formats
  (Q4_0-Q8_K), all dequantizers bit-exact verified
- Self-healing agent watchdog, OpenAI-compatible HTTP API
- 1800+ line engineering journal — every crash and bug documented

MIT. Builds in ~2 minutes on any AMD Strix Halo machine.
