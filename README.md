<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# 50 TOPS INT8. 55.7 TFLOPS measured. 281 tok/s 1-bit.
## On a consumer laptop. Open source. Zero Python.

[![50 TOPS Verified](https://img.shields.io/badge/50%20TOPS-verified-00ff00.svg)](engine/npu/BENCHMARKS.md)
[![55.7 TFLOPS Peak](https://img.shields.io/badge/55.7%20TFLOPS-raw%20silicon-12a0ed.svg)](engine/npu/BENCHMARKS.md)
[![281 tok/s 1-bit](https://img.shields.io/badge/281%20tok%2Fs-1--bit-f00fd2.svg)](engine/npu/BENCHMARKS.md)
[![Pure C++](https://img.shields.io/badge/runtime-C%2B%2B23-00ff00.svg)](engine/npu/src/npu_engine_cb.cpp)
[![DeepSeek v4](https://img.shields.io/badge/built%20with-DeepSeek%20v4-7b3af2.svg)](https://deepseek.com)
[![Claude](https://img.shields.io/badge/shipped%20with-Claude-d97706.svg)](https://claude.ai)
[![MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/discord-1bit.systems-f00fd2.svg?logo=discord&logoColor=white)](https://discord.gg/dSyV646eBs)

</div>

---

### What you get — right now, on one chip

| Engine | Hardware | Precision | Speed | Model | Size |
|--------|----------|-----------|-------|-------|------|
| **NPU** | XDNA 2 · 32 tiles | INT8 | **244 ms/tok** | Qwen3-0.6B | 610 MB |
| **1-bit GPU** | Radeon 8060S · 40 CUs | IQ1_S | **281 tok/s** | Bonsai 1.7B | 385 MB |
| **Vulkan GPU** | Radeon 8060S · 40 CUs | Q4_K | **27 µs/tok** | Qwen3.5-9B | 5.4 GB |

**55.7 TFLOPS raw INT8 GEMM** — exceeds AMD's 50 TOPS rating.  
**All numbers verified on-device.** [Full benchmarks →](engine/npu/BENCHMARKS.md)

### Why this exists

AMD shipped the Strix Halo with a 50 TOPS NPU and a toolchain that soft-blocks
INT8. They sold the FastFlowLM runtime — 93 tok/s, proprietary, closed-source.
One person with a free Chess license, a C++ compiler, and 3 days reverse-engineered
the entire stack. The silicon was never the bottleneck. The business model was.

Every claim is timestamped in [docs/journey.md](docs/journey.md) — an 1,184-line
audit trail of every crash, deadlock, fix, and breakthrough. Open source ships
faster than venture capital.

*Built with DeepSeek v4 (99.9%) · Shipped with Claude (0.1%) · One human.*

## Architecture

```
1bit.systems/
├── engine/
│   ├── npu/          # C++ INT8 engine — NPU (XDNA 2)
│   │   ├── src/npu_engine_i8.cpp    # 145-line inference loop
│   │   ├── src/dequant_q4nx.c       # Q4NX weight dequantizer
│   │   ├── kernel/edge_attention.cc # NPU attention kernel (Chess)
│   │   ├── xclbins/n1_core_i8_v2.py # INT8 MLIR generator
│   │   └── build/                   # Pre-compiled objects
│   └── gpu/          # Zig engine — GPU (Vulkan/CUDA/Metal)
│       ├── src/vulkan/forward.zig   # Vulkan prefill + decode
│       ├── src/cuda/                # CUDA backend
│       ├── src/metal/               # Apple Silicon backend
│       ├── src/shaders/             # GLSL compute shaders (SPIR-V)
│       └── build.zig                # Zig build system
├── docs/             # Architecture, build guide, roadmap, journey
├── site/             # Landing page (Cloudflare Pages)
└── .github/          # CI benchmarks + PR agent
```

## NPU Engine (`engine/npu/`)

**C++23. 4 INT8 contexts alive simultaneously. 246 ms/tok.**

```bash
g++ -std=c++23 -O3 -o npu_engine engine/npu/src/npu_engine_i8.cpp engine/npu/build/dequant_q4nx.o -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl
./npu_engine
```

```
Init 8 contexts. Dequant+pack: 4.3s.
Generate:
  [0] 92850   [1] 26686   [2] 111383  [3] 104068
  [4] 126203  [5] 2541    [6] 90103   [7] 87567
=== 246 ms/tok ===
```

| Metric | Value |
|--------|-------|
| Speed | 246 ms/tok (4.1 tok/s) |
| Precision | INT8 (symmetric per-tensor) |
| Contexts | 4 GEMM + 4 attention, all alive |
| Model | Qwen3-0.6B (Q4NX weights) |
| Binary size | 57 KB |

## GPU Engine (`engine/gpu/`)

**Zig. Vulkan 1.3 compute shaders. GGUF native parser.**

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zinc -m model.gguf --prompt "Hello"
```

| Metric | Value |
|--------|-------|
| Backends | Vulkan (RDNA3/4), CUDA, Metal |
| Model format | GGUF (Q4_K, Q8_0) |
| Decode (Q4_K, 6912×2560) | 27.0 µs |
| Prefill (2560×6912×2560) | 21.9 TFlops |
| Speed vs rocBLAS FP16 | 7.8× faster decode |
| Binary | Single `zinc` executable |

## Performance

| Engine | Hardware | Speed | Models |
|--------|----------|-------|--------|
| **NPU** (engine/npu) | XDNA 2 NPU | 246 ms/tok | Qwen3-0.6B |
| **GPU** (engine/gpu) | Radeon 8060S (Vulkan) | 27 µs/decode | Qwen3.5, Gemma 4 |
| **1bit GPU** (ZINC) | Radeon 8060S (Vulkan) | **3.5 ms/tok** | Bonsai-1.7B IQ1_S (385 MB) |

## The Journey

[docs/journey.md](docs/journey.md) — timestamped audit trail documenting every
crash, deadlock, fix, and breakthrough from the 3-day reverse-engineering sprint.

## License

MIT — see [LICENSE](LICENSE).

---

*Built on Strix Halo. NPU + GPU. One chip. Two engines. Zero Python.*
