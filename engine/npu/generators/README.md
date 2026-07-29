# NPU INT8 MLIR Generators

MLIR generators for the 1bit-systems NPU INT8 GEMM engine. Each generator produces a `.mlir` file for the NPU2 AIE array (Strix Halo), which is compiled to an `.xclbin` via the [MLIR-AIE toolchain](https://github.com/Xilinx/mlir-aie) (`aiecc`).

## Generator Versions

| File | Status | Description |
|------|--------|-------------|
| `n1_core_i8_v24.py` | ✅ **Current** | BD descriptor pipelining — K-iteration batching in groups of 6 |
| `../xclbins/n1_core_i8_v2.py` | ✅ Stable | Flat K-iteration loop (baseline, 16.3s/tok) |
| `../xclbins/n1_core_i8_i32_4row_v10.py` | ⚡ Experimental | INT32 accumulator, 4-row task pipelining |

## v24: BD Descriptor Pipelining (Issue #1075)

### The Problem

In v2, each K-iteration issues a separate DMA start/wait cycle through the
object_fifo acquire/release mechanism. For K = 1024 with k_tile = 64, this
means 16 sequential DMA cycles per M-tile. The NPU shim DMA engine spends
most of its time in start/wait overhead instead of moving data.

Result: **16.3 seconds per token** for the D projection (worst-case K=3072).

### The Fix: K-Iteration Batching

v24 batches K-iterations in groups of **6** per DMA round:

```
v2 (flat):  for each K-iteration { acquire A; acquire B; compute; release }
v24 (batched): for batch of 6 { acquire 6 A + 6 B; wait once; compute 6; release 6 }
```

### Why 6?

The NPU shim DMA engine allows **16 BD descriptors per tile**. Each FIFO buffer
requires one BD for the L2→L1 DMA. The budget:

| FIFO | Buffers | BDs |
|------|---------|-----|
| A_l2l1 | 7 (batch 6 + 1 in-flight) | 7 |
| B_l2l1 | 7 (batch 6 + 1 in-flight) | 7 |
| C_l1l2 | 1 | 1 |
| **Total** | | **15** |

15 BDs ≤ 16 ✓ — stays within the hardware limit.

The L2 tile size (`mtk`) changes from 512 (8 K-iterations) to **384** (exactly
6 K-iterations). This means each L2→L1 batch feeds exactly one core batch.

### Memory Budget

Memory tile (512KB L2 scratchpad):

| Buffer | Count | Size | Total |
|--------|-------|------|-------|
| A_l1 buffers | 7 | 2,048 B (32×64) | 14 KB |
| B_l1 buffers | 7 | 8,192 B (64×128) | 56 KB |
| C_l1 buffers | 1 | 8,192 B (32×128) | 8 KB |
| **Total** | | | **78 KB** |

78 KB ≪ 512 KB ✓ — ample room in the memory tile.

Compute tile (64KB local memory): holds only 1 buffer of each at a time
(consumed from FIFO, not pre-loaded).

### Performance Impact

For Qwen3-0.6B (D projection: M=128, K=3072, N=1024):

| Metric | v2 (flat) | v24 (batched) | Improvement |
|--------|-----------|---------------|-------------|
| K-iterations per M-tile | 3072/64 = 48 | 48 | (same) |
| DMA start/await cycles | 48 | ceil(48/6) = 8 | **6× fewer** |
| Estimated decode time | 16.3 s/tok | ~2.7 s/tok* | **~6× faster** |

*Theoretical estimate: 6× reduction in DMA overhead. Actual depends on
compute-to-DMA overlap ratio on hardware.

### Usage

```bash
# Generate v24 MLIR for QKV projection (M=128, K=1024, N=4096)
python3 engine/npu/generators/n1_core_i8_v24.py \
    -M 128 -K 1024 -N 4096 > mm_qkv_v24.mlir

# Generate v24 MLIR for D projection (M=128, K=3072, N=1024)
python3 engine/npu/generators/n1_core_i8_v24.py \
    -M 128 -K 3072 -N 1024 > mm_d_v24.mlir

# Custom batch size (e.g., 4 for smaller memory budget)
python3 engine/npu/generators/n1_core_i8_v24.py \
    -M 128 -K 1024 -N 4096 --batch-size 4 > mm_qkv_v24_b4.mlir

# Compile with aiecc (requires MLIR-AIE toolchain)
# See: engine/npu/build_xclbins.sh
aiecc mm_qkv_v24.mlir ...
```

## v23 → v24 Changelog

| Aspect | v23 (v2) | v24 |
|--------|----------|-----|
| **L2 K-tile size** (`mtk`) | 512 | 384 (6 × 64) |
| **A_l2l1 FIFO depth** | 2 | 7 |
| **B_l2l1 FIFO depth** | 2 | 7 |
| **Core K-loop** | Sequential acquire→compute→release | Batch acquire 6 → compute 6 → release 6 |
| **Remainder handling** | N/A | Partial batch for leftover K-iterations |
| **BD descriptors/tile** | 5 | 15 |
| **Kernel** | `mm_32x64x128.o` | `mm_32x64x128.o` (same) |
| **Dtype** | int8 / int16 | int8 / int16 (same) |

## Toolchain Requirements

### MLIR-AIE Toolchain

The `.mlir` → `.xclbin` compilation requires:

- **aiecc** (MLIR-AIE v0.3.x) — compiles MLIR to AIE instructions + PDI
- **Peano compiler** (LLVM-based) — kernel compilation for GEMM xclbins
- **LLVM 21** (LLVM-AIE fork) — `opt`/`llc` passes for MLIR lowering

Verified toolchain setup (2026-07-29):

```bash
export AIE_TOOLS_DIR=~/mlir-aie/install_tmp
export PATH=$AIE_TOOLS_DIR/bin:$PATH
export PYTHONPATH=$AIE_TOOLS_DIR/python:$PYTHONPATH
```

### Fix Toolchain Script

Use `fix_toolchain.sh` to resolve opaque-pointer LLVM version mismatches:

```bash
# Check current toolchain
./engine/npu/generators/fix_toolchain.sh --check

# Set up environment (source from build script)
source engine/npu/generators/fix_toolchain.sh --setup-env

# Fix opaque pointers in generated LLVM IR
./engine/npu/generators/fix_toolchain.sh --fix build/dir/

# Generate aiecc wrapper for persistent fix
./engine/npu/generators/fix_toolchain.sh --generate-wrapper
```

The fix script:
1. Routes LLVM IR through LLVM-AIE's LLVM 21 for `opt`/`llc` passes
2. Uses Peano's clang for kernel compilation (correct pointer mode)
3. Patches typed-pointer IR to opaque-pointer format automatically

### Known Issues

- **Opaque pointer mismatch** (LLVM 15+ vs Peano): The `fix_toolchain.sh`
  wrapper handles this by routing typed-pointer IR through LLVM-AIE's LLVM 21
  `opt`/`llc`.
- **Chess vs Peano PDI divergence**: xclbins built with different compilers
  produce different PDI binaries. Always use Peano for GEMM xclbins.
  See [#1076](https://github.com/bong-water-water-bong/1bit-systems/issues/1076).
- **BD count limit**: If you increase batch_size beyond 6, verify total BDs
  stay under 16 per tile. See the table above for the formula.
