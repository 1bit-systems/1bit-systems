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

**Benchmarks**: M=4096, K=4096, synthetic weights, cache-hot. See `benchmarks/` for methodology.

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

## NPU Classical Engines — Superseded (2026-07-28)

| Engine | Tok/s | Status | Model | Notes |
|--------|:-----:|:------:|-------|-------|
| FLM turbo (production) | 94.7 | ❌ deprecated | Qwen3-0.6B | Replaced by npu_engine_universal (PR #1064) |
| C++ v12 (M=32) | 69 | ❌ deprecated | Qwen3-0.6B | Replaced by INT8 GEMM path |
| C++ auto-detect | 42 | ❌ deprecated | 0.6B-8B | Replaced by INT8 GEMM path |
| NPU fused | — | ✅ working | Qwen3-0.6B | All 4 ops (QKV/O/GU/D) verified 0/10000 errors, Peano-compiled |

### npu_engine_universal (2026-07-28 — current)

All 22 INT8 GEMM shapes across 5 models rebuilt via Peano + scalar kernel, verified
0/10000 errors on real hardware. The NPU engine now dispatches all 4 core ops (QKV, O,
GU, D) natively via XRT — replacing the prior FLM-subprocess and v12 paths.

**Toolchain**: Peano-only. Chess is permanently deprecated (multi-dim BD repeat
descriptors hang NPU2 DMA on all tested designs, including AMD's own official examples).

**Verified shapes**:

| model | QKV (K,N) | O (K,N) | G/U or GU (K,N) | D (K,N) | cols |
|---|---|---|---|---|---|
| qwen3_0_6b | 1024,4096 | 2048,1024 | GU: 1024,6144 | 3072,1024 | 8 |
| qwen3_8b | 4096,6144 | 4096,4096 | G/U: 4096,12288 | 12288,4096 | 8 |
| qwen3_vl_4b | 2560,6144 | 4096,2560 | G/U: 2560,9728 | 9728,2560 | 4 |
| llama | 4096,6144 | 4096,4096 | G/U: 4096,14336 | 14336,4096 | 8 |
| gemma4_e2b | 1536,2560 | 2048,1536 | GU: 1536,12288 | 6144,1536 | 4 |

See [`engine/npu/generators/README.md`](generators/README.md) for build instructions
and toolchain rationale.

---

## Raw Silicon: GEMM Throughput — Chess (historical, deprecated)

Chess-compiled INT8 xclbins (before Peano migration). All Chess designs hang on NPU2 hardware.
These numbers are from before the hang was root-caused and are kept for historical reference only.

| Projection | Shape | Time | TFLOPS (avg/peak) | % of 50 TOPS |
|-----------|-------|:----:|:-----------------:|:------------:|
| **D** (down) | 1024×3072×1024 | 116 µs | 55.7 / 80.5 | 111% |
| **O** (output) | 1024×2048×1024 | 108 µs | 39.7 / 49.4 | 79% |
| **GU** (gate+up) | 1024×1024×6144 | 801 µs | 16.1 / 16.5 | 32% |
| **QKV** (fused) | 1024×1024×4096 | 559 µs | 15.4 / 15.5 | 31% |

**Note**: These Chess-compiled xclbins hang on NPU2 hardware. Peano-compiled xclbins
(with flat BD descriptors, not multi-dim repeat) work correctly but throughput is
lower — each K-iteration requires a separate DMA transaction. See generators/README.md.

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
| **Jul 28** | **npu_engine_universal INT8 GEMM** | **All 22 shapes, 0 errors, Peano-compiled. Chess deprecated. NPU attention fixed.** |

---

*Kernel-level numbers verified bit-exact on real Strix Halo (gfx1151). NPU ternary kernels compile through `xchesscc_wrapper aie2p`. See `site/benchmarks.json` for the full machine-readable dataset.*
