<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# 1bit.systems — NPU + GPU inference, Strix Halo native.

### Pure native. Zero Python.

Two engines. One silicon. The NPU AMD shipped locked, and the GPU they
already open. Both running local inference on the Ryzen AI Max+ 395.

[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.systems-12a0ed.svg)](https://1bit.systems)
[![Discord](https://img.shields.io/badge/discord-1bit.systems-f00fd2.svg?logo=discord&logoColor=white)](https://discord.gg/dSyV646eBs)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)

</div>

---

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

## The Journey

[docs/journey.md](docs/journey.md) — timestamped audit trail documenting every
crash, deadlock, fix, and breakthrough from the 3-day reverse-engineering sprint.

## License

MIT — see [LICENSE](LICENSE).

---

*Built on Strix Halo. NPU + GPU. One chip. Two engines. Zero Python.*
