<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# INT8 inference, unlocked on Strix Halo.

### Pure C++. 4.1 tok/s. Zero Python at runtime.

`1bit.systems` is an INT8 inference engine for the AMD Strix Halo NPU (XDNA 2).
It runs Qwen3-0.6B at **243 ms/tok** with diverse token output — on consumer
silicon that AMD's own Linux toolchain soft-blocks.

[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.systems-12a0ed.svg)](https://1bit.systems)
[![Discord](https://img.shields.io/badge/discord-1bit.systems-f00fd2.svg?logo=discord&logoColor=white)](https://discord.gg/dSyV646eBs)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)

</div>

---

## What this is

A **standalone C++ binary** that loads a Qwen3-0.6B model, dequantizes Q4NX
weights to INT8, and runs inference on the Strix Halo NPU via AMD's XRT
runtime — with 4 NPU contexts alive simultaneously and zero context swapping.

```bash
g++ -std=c++23 -O3 -o npu_engine engine/src/npu_engine_i8.cpp engine/build/dequant_q4nx.o \
    -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl
./npu_engine
```

```
=== NPU Engine i8 + Attention ===
Init 8 contexts (4 GEMM + 4 attention).
Dequant+pack: 4.5s

Generate:
  [0] 107325  [1] 40469   [2] 115358  [3] 127809
  [4] 121341  [5] 35443   [6] 16927   [7] 105629

=== 243 ms/tok ===
```

## Hardware

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen AI Max+ 395 (16 Zen 5 cores) |
| GPU | Radeon 8060S (40 RDNA 3.5 CUs) |
| NPU | XDNA 2 (32 AIE2P tiles, 50 TOPS INT8) |
| RAM | 128 GB LPDDR5x unified |
| OS | Ubuntu 26.04 LTS, kernel 7.0.0+ |

## Performance

| Engine | Speed | Tokens | Approach |
|--------|-------|--------|----------|
| **i8 4-live** | **243 ms/tok (4.1 tok/s)** | Diverse ✅ | 4 INT8 contexts + NPU attention |
| i8 swap | 446 ms/tok | Diverse ✅ | 1-at-a-time context swap |
| BFP16 v8 | 1335 ms/tok | Repeating ❌ | Chess GEMM, BO-cached swap |
| FLM (reference) | ~11 ms/tok (93 tok/s) | — | Proprietary AMD runtime |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   npu_engine (C++)                      │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ QKV GEMM │  │ O  GEMM  │  │ GU GEMM  │  │ D GEMM │ │
│  │ (INT8)   │  │ (INT8)   │  │ (INT8)   │  │ (INT8) │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───┬────┘ │
│       │              │              │            │      │
│       └──────────────┴──────────────┴────────────┘      │
│                          │                               │
│                    ┌─────▼──────┐                        │
│                    │ NPU Driver │                        │
│                    │ (amdxdna)  │                        │
│                    └────────────┘                        │
└─────────────────────────────────────────────────────────┘
```

All 4 GEMM contexts and 4 attention contexts are alive simultaneously.
No context swapping. Per-layer weight BOs are pre-loaded at startup —
zero weight memcpy during inference.

## Why this matters

AMD ships the Strix Halo NPU with:
- An MLIR parser that rejects `i8` types (only `v8bfp16ebs8` is accepted)
- A toolchain that requires a proprietary Chess compiler license
- No public INT8 inference examples on Linux

We fixed:
- **K-interleaving bug** — added `dataReuse` annotations to ObjectFifo DMA
- **INT8 xclbin generation** — Chess-compiled kernels with correct INT8 matmul
- **4-live contexts** — NPU2 supports multiple concurrent hw_contexts
- **Per-layer weight BOs** — pre-loaded, never copied during inference

## Quick Start

See [docs/building.md](docs/building.md) for full build instructions.

### Prerequisites

- AMD Strix Halo system (Ryzen AI Max+ 395)
- Ubuntu 26.04+ with kernel 7.0.0+
- AMD XRT 2.21+ (`xrt-smi examine` should show `RyzenAI-npu5`)
- Chess compiler license (free from [AMD Ryzen AI EA](https://account.amd.com/en/member/ryzenai-sw-ea.html))
- Qwen3-0.6B Q4NX model from FastFlowLM

### Build

```bash
# 1. Install XRT
sudo apt install libxrt2 libxrt-npu2 libxrt-dev xrt-smi

# 2. Set up torch2aie toolchain (for xclbin compilation)
git clone https://github.com/taowen/torch2aie.git
cd torch2aie && ./scripts/setup_python.sh && source scripts/env.sh

# 3. Place Chess license
mkdir -p torch2aie/licenses/
cp /path/to/Xilinx.lic torch2aie/licenses/

# 4. Build INT8 xclbins (one-time)
cd engine/xclbins
python3 n1_core_i8_v2.py -M 128 -K 1024 -N 4096 -m 32 -k 64 -n 128 > qkv.mlir
aiecc --aietools=$AIETOOLS_DIR --aie-generate-xclbin --no-compile-host \
      --xclbin-name=final_i8_QKV_v.xclbin qkv.mlir
# (repeat for O, GU, D projections)

# 5. Build engine
g++ -std=c++23 -O3 -o npu_engine engine/src/npu_engine_i8.cpp \
    engine/build/dequant_q4nx.o \
    -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl

# 6. Run
./npu_engine
```

## Roadmap

| Milestone | Status |
|-----------|--------|
| INT8 K-interleaving fix | ✅ Done |
| 4-live INT8 contexts | ✅ Done |
| NPU attention kernel | ✅ Built (not yet wired) |
| 174 ms/tok target | 📋 NPU attention dispatch |
| GGUF Q8_0 model loading | 📋 Direct INT8 weights |
| 1-bit / BitNet b1.58 | 🔮 Post-INT8 |

## The Journey

This engine is the result of a 3-day reverse-engineering sprint documented in
[docs/journey.md](docs/journey.md) — a timestamped handoff showing every
breakthrough, crash, deadlock, and fix:

- **UPDATE 1-7**: NPU benchmarking, FLM reverse engineering, pyxrt analysis
- **UPDATE 8**: Full-layer xclbin deadlock (ERT state 8 timeout)
- **UPDATE 9**: Chess license activated — 31.4 TFLOPS verified
- **UPDATE 10**: NaN accumulation fixed (safe softmax + error containment)
- **UPDATE 11**: NPU edge attention kernel built for 0.6B
- **UPDATE 12**: INT8 K-interleaving fixed (dataReuse on ObjectFifo)
- **UPDATE 13**: 4-live INT8 engine — 219 ms/tok

Every dead end is documented. Every wrong turn is timestamped. This is
real hardware, real crashes, real progress.

## License

MIT — see [LICENSE](LICENSE).

---

*Built on Strix Halo. Powered by the NPU AMD shipped but never unlocked.*
