# Reddit r/LocalLLaMA Post Draft

## Title:
**Mamba1 GPU backend live — BlackMamba 79.4 tok/s on Strix Halo, all in one C++ binary**

## Body:

**tl;dr**: Wired the Mamba1 GPU kernels (mamba1_engine.hip) into the full inference
pipeline. BlackMamba 1.5B: **79.4 tok/s**. BlackMamba 2.8B: **46.1 tok/s**.
Both running entirely on the Strix Halo iGPU via ROCm HIP, no Python, no PyTorch.
Alternating SSM + MoE layers, full autoregressive decode, single binary.

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems && source env.sh
cmake -B build -G Ninja
cmake --build build --target unified_server -j$(nproc)

# Load any GGUF — it auto-detects the architecture and routes to the right backend:
./build/unified_server -w /path/to/models/ -p 8088
```

### What's new

The engine already ran transformer, MoE, Mamba2-hybrid, and ternary models through
NPU + GPU + CPU backends. What was missing: **Mamba1 SSM** (Zamba, BlackMamba).
The kernels existed in `mamba1_engine.hip` but weren't wired into any build target
or MoE-aware loader. Now they are — and they work.

### The bugs we found along the way

Three correctness bugs in the original kernel code that would have silently
produced garbage output:

1. **Conv state buffer overflow** — the conv1d state shift loop wrote past
   the allocated buffer, corrupting adjacent GPU memory on every SSM layer
2. **A_log never exponentiated** — Mamba1 parameterizes `A = -exp(A_log)`,
   but the selective scan used `A_log` directly as `A`. The SSM dynamics
   were completely wrong
3. **HIP device stubs missing** — kernel launches used `<<<>>>` syntax in a
   file compiled as CXX, not HIP. Linker couldn't find `__device_stub__*`
   symbols

All fixed, all verified running on hardware.

### Current benchmark table (Strix Halo, real end-to-end)

| Model | Architecture | Throughput |
|-------|-------------|:----------:|
| BlackMamba 1.5B | 15 SSM + 15 MoE | **79.4 tok/s** |
| BlackMamba 2.8B | 18 SSM + 18 MoE | **46.1 tok/s** |
| ZR1 1.5B (Q4_K) | Dense transformer | **~30 tok/s** (Vulkan) |
| Prefill GEMV | 2560×6912 INT8 | **41.45 TFLOPS** |
| KV cache FD | L=2048 | **57.1 GB/s** (12.7× vs FP16) |

### What else the binary does

- **Auto-routes any GGUF**: drop a model file in the weights directory, the
  server reads its architecture header and picks the right backend — no config,
  no restart
- **9 hardware backends**: NPU XDNA 2, ZINC GPU (Vulkan), HIP GPU (ROCm),
  **Mamba1 HIP**, CPU AVX-512, CPU scalar, generic GGUF CPU
- **Model-agnostic loader**: Qwen2/Qwen3/Llama/Mistral/Gemma/Phi/Zamba2/Mamba
  — same binary, no per-model code paths
- **FastFlowLM fully replaced**: 22 proprietary `.so` libraries reverse-engineered,
  whole NPU stack rebuilt from source (87.8 MB closed → 17.5 MB open)

### Links

GitHub: https://github.com/bong-water-water-bong/1bit-systems
Audit trail: `docs/journey.md` — 1800+ lines, every bug and fix documented
PR #579: https://github.com/bong-water-water-bong/1bit-systems/pull/579

MIT. Your hardware, your model, your choice of backend.
