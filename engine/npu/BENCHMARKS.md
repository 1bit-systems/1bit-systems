# 1bit.systems NPU Benchmarks

**Hardware**: AMD Ryzen AI Max+ 395 (Strix Halo), XDNA 2 NPU, 32 AIE2P tiles  
**GPU**: Radeon 8060S (gfx1151), 32 CUs, HIP + Vulkan  
**OS**: Ubuntu 26.04 LTS

---

## Binary/Ternary GPU Kernels (HIP — Radeon 8060S)

All kernels verified bit-exact against CPU reference on real Strix Halo hardware (gfx1151).

| Kernel | Format | Bits/W | Latency (4K×4K) | Correctness |
|--------|--------|:------:|:---------------:|:-----------:|
| Q1 GEMV fused | 128-block Q1_0 | 1.0 | — | ✅ exact |
| Fused TQ2 | QKV+GU fused | 2.0 | — | ✅ exact |
| TQ2 GEMV | Group-scaled ternary | 2.0 | — | ✅ exact |
| **BitNet TQ2_0** | GGML_TYPE_TQ2_0 (llama.cpp) | 2.06 | 2 µs | ✅ exact |
| **BitNet TQ1_0** | GGML_TYPE_TQ1_0 (base-3) | 1.69 | 202 GB/s | ✅ exact |
| **Q1_0 binary** | 128-block sign bits | 1.0 | 1.1 µs | ✅ exact |
| **TQ1 halo** | Base-3 H1B v4 | 1.58 | 17.5 µs | ✅ exact |

**Benchmarks**: M=4096, K=4096, synthetic weights, cache-hot. See `bench/` for methodology.

---

## NPU Ternary Kernels (XDNA 2 — AIE2P)

Three on-tile LUT-decode kernels, all compile via `xchesscc_wrapper aie2p`:

| Format | Bits/W | Decode Method | Object | DDR Savings |
|--------|:------:|:-------------:|:------:|:-----------:|
| **TQ2** | 2.0 | LUT[256] → byte→4×int8 | 9532 B | 4× vs INT8 |
| **TQ1** | 1.58 | LUT[243] → byte→5×int8 (base-3) | 9624 B | 4.9× vs INT8 |
| **Q1_0** | 1.0 | 64-bit sign mask → ±scale | 11984 B | 3.6× vs INT8 |

All three live in `engine/npu/kernel/`. Build: `./build_npu_ternary.sh tq2|tq1|q1 <tag> <H> <NH> <NKV> <HD> <IM>`.

MLIR designs (`n1_core_tq2_placed.py`, `n1_core_tq1_placed.py`, `n1_core_q1_placed.py`) in `~/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/`.

NPU bridge: `tq2_to_q4nx` converts any 1BP TQ2 model to Q4NX format for existing NPU engine (~3.5s for 112 tensors).

---

## NPU Classical Engines

| Engine | Tok/s | Status | Model | Notes |
|--------|:-----:|:------:|-------|-------|
| FLM turbo (production) | 94.7 | ✅ historical | Qwen3-0.6B | Proprietary, now fallback |
| C++ v12 (M=32) | 69 | ⚙️ raw | Qwen3-0.6B | Open source, OpenMP tuned |
| C++ auto-detect | 42 | ⚙️ raw | 0.6B-8B | Single binary, all models |
| NPU fused | 291 | ❌ broken | Qwen3-0.6B | Hangs on real generation |

### Raw C++ Engine — Auto-Detect (M=32 batch, OpenMP)

| Model | H | IM | Size | Prefill | Decode | Tok/s |
|-------|---|----|------|---------|--------|:-----:|
| **Qwen3-0.6B** | 1024 | 3072 | 610 MB | 14 ms/tok | 36 ms/tok | 28 |
| **Gemma4-E2B** | 1536 | 6144 | 4.7 GB | 20 ms/tok | 62 ms/tok | 16 |
| **Llama-3.1-8B** | 4096 | 14336 | 5.7 GB | 47 ms/tok | 100 ms/tok | 10 |
| **Qwen3-8B** | 4096 | 12288 | 6.0 GB | 49 ms/tok | 127 ms/tok | 8 |

All verified on Strix Halo NPU. Single auto-detecting binary.

---

## Raw Silicon: GEMM Throughput

Chess-compiled INT8 xclbins, verified on-device.

| Projection | Shape | Time | TFLOPS (avg/peak) | % of 50 TOPS |
|-----------|-------|:----:|:-----------------:|:------------:|
| **D** (down) | 1024×3072×1024 | 116 µs | 55.7 / 80.5 | 111% |
| **O** (output) | 1024×2048×1024 | 108 µs | 39.7 / 49.4 | 79% |
| **GU** (gate+up) | 1024×1024×6144 | 801 µs | 16.1 / 16.5 | 32% |
| **QKV** (fused) | 1024×1024×4096 | 559 µs | 15.4 / 15.5 | 31% |

---

## Engine Evolution

| Date | Milestone | Detail |
|------|-----------|--------|
| Jun 28 | First working decode | v7 BFP16, 1930 ms/tok |
| Jul 1 | K-interleaving fix | i8 swap, 244 ms/tok |
| Jul 2 | Auto-detect | All 5 models, 28-8 tok/s, 0 crashes |
| **Jul 24** | **Binary/ternary GPU kernels** | **Q1_0, BitNet, IQ GPU kernels verified exact on Strix Halo** |
| **Jul 24** | **NPU ternary LUT decode** | **TQ2/TQ1/Q1_0 on-tile decode, 3 Chess kernels compile** |
| **Jul 24** | **1BP TQ1 converter** | **`--tq1` flag, base-3 1.58-bit format end-to-end** |
| **Jul 24** | **NPU ternary bridge** | **`tq2_to_q4nx` converter ships, 3.5s per model** |

---

*Kernel-level numbers verified bit-exact on real Strix Halo (gfx1151). NPU ternary kernels compile through `xchesscc_wrapper aie2p`. See `site/benchmarks.json` for the full machine-readable dataset.*
