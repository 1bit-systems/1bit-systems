# 🏎️ Native Ternary NPU — Verified on Strix Halo Hardware

**Date:** 2026-07-08 | **Device:** RyzenAI-npu5 (XDNA2) | **XRT:** 2.21.75

## Hardware Verification

### Single-Core (mm_ternary_32x64x128)

| Metric | Value |
|--------|-------|
| All-ones test | **-256.0000 exactly (32/32)** ✅ |
| Latency | 68.3 µs |
| Throughput | 14,636 calls/s |
| GMACs/s | 0.120 |
| XCLBin | 16 KB |

### 32-Core (4×8 grid, per-column routing)

| Metric | Value |
|--------|-------|
| All-ones test | **-256.0000 exactly (128/128)** ✅ |
| Latency | 118.9 µs |
| Throughput | 8,410 calls/s |
| GMACs/s | 0.276 |
| XCLBin | 314 KB |
| NPU Instructions | 572 dwords |

## Architecture

```
GGUF (Q2_0/Q1_0) → q2_0_decode.py → Q4NX → NativeTernaryCtx
                                              ↓
                                    mm_ternary_32x64x128 (Chess C++)
                                              ↓
                                    NPU xclbin (object_fifo)
                                              ↓
                                    32 cores: 4 rows × 8 cols
                                    Per-column DMA, same-column fan-out
```

## Dataflow (32-core)

```
Per column (col 0..7):
  shim[col] → mem[col] → fan-out to core[0..3][col] (same column only)

Per-column flat buffer:
  [core0_weights|scales] [core1_w|s] [core2_w|s] [core3_w|s] [activations]

Each core: row_start + num_rows to pick its M/32-row slice
Output: per-core → gather per-column via object_fifo_link
```

## Files

| File | Purpose | Status |
|------|---------|--------|
| `1bit-systems/engine/npu/kernel/mm_ternary_32x64x128.cpp` | Chess kernel | ✅ Verified |
| `engine/npu/kernel/n1_core_native_ternary.py` | Single-core MLIR generator | ✅ Verified |
| `engine/npu/kernel/n1_core_native_ternary_32core.py` | 32-core MLIR generator | ✅ Verified |
| `engine/npu/build/build_native_ternary_32core.sh` | 32-core build script | ✅ Builds |
| `spec-decode/engine/npu_ternary_target.h` | spec-decode integration | ✅ Compiles |
| `tools/q2_0_decode.py` | Q2_0 bit-exact decoder | ✅ Verified |
| `tools/q2_0_to_q4nx.py` | Q2_0 → Q4NX converter | ✅ Working |

## Build & Test

```bash
# Build 32-core
source engine/npu/build/env.sh
bash engine/npu/build/build_native_ternary_32core.sh 128 64

# Test (all-ones)
g++ -O2 -std=c++17 -o test test_ternary_32core.cpp -lxrt_coreutil
./test
```
