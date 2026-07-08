# 1.58-bit Ternary Inference on NPU

**Status**: ✅ **HARDWARE VERIFIED** — 32-core native ternary runs on Strix Halo
**Date**: 2026-07-08 | **Device**: RyzenAI-npu5 (XDNA2) | **XRT**: 2.21.75

## Hardware Verification

| Configuration | Latency | Throughput | GMACs/s | XCLBin | All-ones |
|--------------|---------|------------|---------|--------|----------|
| 1 core | 68.3 µs | 14,636 calls/s | 0.120 | 16 KB | 32/32 ✅ |
| **32 cores** | **118.9 µs** | **8,410 calls/s** | **0.276** | **314 KB** | **128/128 ✅** |

BF16 accumulation, 2-bit packed ternary weights (4× memory density vs INT8).

## Architecture

Two paths to run ternary models on NPU:

1. **INT8 passthrough** (proven): Q2_0 GGUF → dequant to INT8 → existing INT8 GEMM pipeline
2. **Native ternary** (new, verified): On-the-fly 2-bit decode → BF16 MAC → 4× memory density

## Native Ternary Kernel

**`1bit-systems/engine/npu/kernel/mm_ternary_32x64x128.cpp`** — Chess C++ kernel for AIE2P.

```
Kernel:    mm_ternary_32x64x128(input, output, row_start, num_rows)
Input:    [M * K_packed bytes weights (uint8)] [M * 2 bytes scales (bf16)]
          [K_ternary * 2 bytes activations (bf16)]
Output:   num_rows bf16 scalars

Packing:  4 ternary weights per byte, 2-bit fields
Mapping:  00→-1.0, 01→0.0, 10→+1.0, 11→-1.0
Compute:  8 bytes→32 BF16 decode, 8 iterations→256 dot product, reduce_add + scale
```

`row_start`/`num_rows` enable multi-core tiling: all cores in a column share one
buffer, each picks its row slice. DIM_M is compile-time (per-column row count).

## MLIR Generators

| Generator | Cores | Pattern | DIM_M | Verified |
|-----------|-------|---------|-------|----------|
| `n1_core_native_ternary.py` | 1 | object_fifo, shim→mem→core | 32 | ✅ HW |
| `n1_core_native_ternary_8core.py` | 8 (1×8) | per-column DMA, separate per-col buffers | 4 | ⚠️ aiecc crash |
| `n1_core_native_ternary_32core.py` | 32 (4×8) | per-column DMA, same-column fan-out | 16 | ✅ HW |

All use `object_fifo` + `shim_dma_single_bd_task` dataflow — the proven pattern
from `n1_core_ternary.py`. The simple `aie.flow + writebd` pattern (build script
heredoc) is deprecated — it generates incomplete NPU sequencer instructions.

### 32-core Dataflow (Verified Working)

```
Per column (col 0..7):
  shim[col] → mem[col] → fan-out to core[0..3][col]  (same column ONLY)

Per-column flat buffer:
  [core0_weights|scales] [core1_w|s] [core2_w|s] [core3_w|s] [activations]

Each core: row_start + num_rows picks its M/32-row slice.
Output: per-core → gather per-column via object_fifo_link with offsets.
```

**Key finding**: NPU2 (XDNA2) AIE interconnect does NOT support cross-column
mem→core broadcast. Each mem tile can only reach cores in its own column.
The fixed design uses per-column DMA with same-column fan-out to 4 rows.

### Build

```bash
source engine/npu/build/env.sh

# 32-core (production):
bash engine/npu/build/build_native_ternary_32core.sh 128 64

# 8-core (experimental):
bash engine/npu/build/build_native_ternary_8core.sh 32 64
```

### Test

```bash
# All-ones validation (compiled from engine/npu/tests/test_ternary_32core.cpp)
g++ -O2 -std=c++17 -o test_ternary_32core \
    engine/npu/tests/test_ternary_32core.cpp -lxrt_coreutil

./test_ternary_32core \
    engine/npu/build/build/ternary_32core/ternary_32core.xclbin \
    engine/npu/build/build/ternary_32core/insts_ternary_32core.txt
# Expected: 128/128 outputs = -256.0000
```

## INT8 Passthrough Pipeline (Legacy)

```
Q2_0 / Q1_0 GGUF (ternary 1.58-bit / 1-bit binary)
  │
  ▼
tools/q2_0_to_q4nx.py    # Dequant → bake per-block scale → INT8
  │
  ▼
Q4NX file (INT8 weights)
  │
  ▼
npu_engine_universal     # INT8 xclbin GEMM
```

## Q2_0 Decoder

**`tools/q2_0_decode.py`** — bit-exact Q2_0 ternary GGUF decoder.

```
Format:   block_q2_0 { f16 d; uint8_t qs[32]; }  QK=128, 34 bytes/block
Decode:   value[j] = (code - 1) * d   code = (qs[j/4] >> ((j%4)*2)) & 3
Verified: cos_vs_F16 = 1.000000 across 4 tensors (Bonsai-1.7B)
```

⚠️ **ZINC collision**: GGML type 42 maps to `stq1_0` in ZINC's `gguf.zig` —
a DIFFERENT format. ZINC needs a distinct Q2_0 path for Bonsai ternary models.

## Models Benchmarked

| Model | Size | Format | Hidden | FFN | Layers | Heads | Vocab |
|-------|------|--------|--------|-----|--------|-------|-------|
| **Bonsai-1.7B-Q1_0** 🔵 | 250 MB | Q1_0 (1-bit) | 2048 | 6144 | 28 | 16/8 | 151,669 |
| ZAYA1-8B-zaya | 5.6 GB | Q4_K/F32 | 2048 | — | 40 | — | 262,147 |
| ZAYA1-8B-Q4_K_M | 5.6 GB | Q4_K | 2048 | — | 40 | — | 262,147 |

**Bonsai-1.7B-Q1_0** is the only 1-bit model available locally.
All weight tensors (197) use Q1_0 format. Norms/embed/lm_head use F32.

## Bit-Exact Verification

| Check | Result |
|-------|--------|
| Q2_0 decoder cos_vs_F16 | **1.000000** (4 tensors, Bonsai-1.7B) |
| Q2_0 → Q4NX converter | ✅ Round-trip lossless (INT8 passthrough) |
| Native ternary 1-core all-ones | ✅ **-256.0000** (32/32) |
| Native ternary 32-core all-ones | ✅ **-256.0000** (128/128) |

## NPU Hardware

| Component | Value |
|-----------|-------|
| Device | RyzenAI-npu5 (Strix Halo XDNA2) |
| XRT | 2.21.75 |
| Firmware | 1.1.2.65 |
| UEFI | Unlocked (AmdSetup verified) |
| Driver | amdxdna loaded |

## Ternary Packing Formats

| Format | bpw | Encoding | Use |
|--------|-----|----------|-----|
| Q1_0 (GGUF type 41) | 1.0 | 1 bit: +d/-d | Bonsai-1.7B |
| Q2_0 (GGUF type 42) | 2.0 | 2-bit: {-1,0,+1,+2}×scale | PrismML ternary |
| NPU packed uint8 | 2.0 | 4×2bit/byte: 00=-1,01=0,10=+1,11=-1 | mm_ternary kernel |
| TQ1 halo v4 | 1.6 | base-3: 5 values/byte | HIP GEMV |
| Sherry v3 | 1.25 | 3:4 sparsity (training-only) | HIP GEMV |

## GPU Kernels (RDNA 3.5)

13 HIP kernel sources across 5 packing formats in `1bit/kernels/`:
`ternary_gemv_tq1_halo`, `ternary_gemv_sherry`, `ternary_gemv_phase5_*`,
`zaya_moe_ternary_gemv`. 16 compiled `.o` files in `1bit/build/`.
Plus Vulkan `ternary_gemm.comp` + `.spv` in `npu-sandbox/vulkan-gevm/`.

## Next Steps

1. ✅ Multi-tile MLIR (8-core, 32-core)
2. ✅ NPU hardware validation (all-ones bit-exact)
3. 🔧 32-core random BF16 validation (output layout fix needed)
4. 🔧 Per-layer xclbins for Q/K/V/O/Up/Gate/Down dimensions
5. 🔧 Full model inference (tile K=2048 across NPU calls)
6. 🔧 End-to-end Bonsai-1.7B inference on NPU
