# 40-Column NPU2 XCLBIN Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and test the first 40-column (160-core) NPU2 xclbin for matrix multiplication on Strix Halo XDNA 2.

**Architecture:** Adapt the existing config2 gemm asymmetric tile buffering design from 8 columns to 40 columns by creating a parametric placement script. Compile with the rebuilt NPU2-40 compiler, run on hardware via XRT, and measure throughput.

**Tech Stack:** MLIR-AIE (modified), XRT 2025.2, NPU2 driver (amdxdna), v8bfp16ebs8 microkernel

## Global Constraints

- n_aie_cols = 40 (full NPU2), n_aie_rows = 4 (core rows, unchanged)
- Tile sizes: m=96, k=128, n=96 (from config2 microkernel) — must match `mm_{m}x{k}x{n}.o`
- Design matrix: M=6144, K=4096, N=6144 (where M/m % n_aie_rows == 0 and N/n % n_aie_cols == 0)
  - M/m = 6144/96 = 64, per_row = 64/4 = 16 (integer ✓)
  - N/n = 6144/96 = 64, per_col = 64/40 → NOT integer! 64/40 = 1.6
  - **Better: N=3840, N/n = 3840/96 = 40, per_col = 40/40 = 1** (integer ✓)
  - **M=6144, M/m = 6144/96 = 64, per_row = 64/4 = 16** (integer ✓)
  - Total: 16 × 1 = 16 groups, 64 × 40 × 0.75 = 1920 tiles across 160 cores
- DIV = 6 (asymmetric tile ratio, hardcoded in microkernel schedule)
- Use `v8bfp16ebs8` for weights, `v8bfp16ebs8` for input/output
- Compiler: `/home/bcloud/mlir-aie/npu2_40_toolchain/`
- Toolchain wrapper: `source /home/bcloud/mlir-aie/npu2_40_toolchain/aiecc_wrapper.sh`
- Use numpy and the iron Python API for MLIR generation

---
