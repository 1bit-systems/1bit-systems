# 🏎️ Native Ternary NPU — Hardware Verified on Strix Halo

**2026-07-08** | **RyzenAI-npu5 (XDNA2)** | **XRT 2.21.75** | **FW 1.1.2.65**

## Hardware Verification

| Configuration | Latency | Throughput | GMACs/s | XCLBin | All-ones |
|--------------|---------|------------|---------|--------|----------|
| 1 core | 68.3 µs | 14,636 calls/s | 0.120 | 16 KB | 32/32 ✅ |
| **32 cores** | **118.9 µs** | **8,410 calls/s** | **0.276** | **314 KB** | **128/128 ✅** |

All outputs produce **-256.0000 exactly** (BF16) on the all-ones test.
Weights: 2-bit packed (4× memory density vs INT8). Accumulation: BF16.

## Architecture

```
Bonsai-1.7B-Q1_0.gguf (250 MB, 1-bit binary)
  │
  ▼
q2_0_decode.py (cos=1.000000 vs F16)
  │
  ▼
q2_0_to_q4nx.py (INT8 passthrough)
  │  └─→ npu_engine_universal (legacy INT8 path)
  │
  ▼
NativeTernaryCtx (flat buffer → NPU BO)
  │
  ▼
mm_ternary_32x64x128.o (Chess C++, 2-bit decode + BF16 MAC)
  │
  ▼
32-core xclbin (4 rows × 8 cols, per-column DMA, object_fifo)
  │
  ▼
128 bf16 outputs (128/128 bit-exact verified)
```

## 32-Core Dataflow

```
Per column (col 0..7):
  shim[col] → mem[col] → fan-out to core[0..3][col]  (same column ONLY)

Per-column flat buffer:
  [core0_weights(4×64B)|scales(4×2B)]
  [core1_weights|scales] [core2_w|s] [core3_w|s]
  [activations(256×2B shared)]

Each core: row_start, num_rows picks its M/32-row slice.
Output: per-core 4bf16 → gather per-column (16bf16) → DMA to host (128bf16).
```

Key finding: NPU2 AIE interconnect does NOT support cross-column mem→core broadcast.
Fix: per-column DMA with same-column fan-out to 4 rows.

## Models Analyzed

| Model | Size | Format | Hidden | FFN | Layers | Heads | Tensors |
|-------|------|--------|--------|-----|--------|-------|---------|
| **Bonsai-1.7B-Q1_0** 🔵 | 250 MB | Q1_0 | 2048 | 6144 | 28 | 16/8 | 310 (197 Q1_0) |
| ZAYA1-8B-zaya | 5.6 GB | Q4_K/F32 | 2048 | — | 40 | — | 1041 |
| ZAYA1-8B-Q4_K_M | 5.6 GB | Q4_K | 2048 | — | 40 | — | 921 |

## Files

| File | Purpose | Status |
|------|---------|--------|
| `1bit-systems/engine/npu/kernel/mm_ternary_32x64x128.cpp` | Chess kernel (+row_start/num_rows) | ✅ HW Verified |
| `engine/npu/kernel/n1_core_native_ternary.py` | Single-core MLIR generator | ✅ HW Verified |
| `engine/npu/kernel/n1_core_native_ternary_32core.py` | 32-core MLIR generator | ✅ HW Verified |
| `engine/npu/build/build_native_ternary_32core.sh` | 32-core build script | ✅ Builds |
| `engine/npu/tests/test_ternary_32core.cpp` | 32-core test harness | ✅ Compiles |
| `spec-decode/engine/npu_ternary_target.h` | spec-decode integration | ✅ Compiles |
| `tools/q2_0_decode.py` | Q2_0 bit-exact decoder | ✅ cos=1.0 |
| `tools/q2_0_to_q4nx.py` | Q2_0→Q4NX converter | ✅ Working |
| `docs/ternary-npu.md` | Main documentation | ✅ Updated |

## Quick Start

```bash
# Build 32-core xclbin
source engine/npu/build/env.sh
bash engine/npu/build/build_native_ternary_32core.sh 128 64

# Test on NPU
g++ -O2 -std=c++17 -o test engine/npu/tests/test_ternary_32core.cpp -lxrt_coreutil
./test  # Expected: 128/128 bit-exact
```
