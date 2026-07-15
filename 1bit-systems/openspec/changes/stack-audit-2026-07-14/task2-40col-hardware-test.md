# 40-column hardware test (2026-07-14, evening): built it, ran it, real answer

## What was built (all real, all verified, in this order)

1. Minimal single-tile smoke test at `tile(39, ...)` — compiled cleanly through the patched NPU2-40 `aie-opt`/`aiecc` toolchain. Confirms the open-source MLIR compiler layer genuinely accepts column 39, not just in the compiler-design doc's own bounds-check table but in a real generate→compile pass.
2. Adapted `n2_core_placed.py` (AMD's proven bf16 32-core template, the one STEP5-INT8-32TILE-PLAN.md used as its structural base) to `n_aie_cols=40`. Found and fixed **three real bugs** left over from the 8-column original: `for col in range(8)` (B-fifo setup, only 8 of 40 columns wired), `core_tiles[row][0:8]` (A-matrix broadcast only reaching 8 of 40 compute tiles), `c_base_idx = group_idx * 8` (should scale with `n_aie_cols`, not a literal 8).
3. Found the real reason the first compile produced an empty (16-byte) instruction file: `num_col_tile = N // n // n_aie_cols` underflows to **zero** with integer division when `N=4096, n=128, n_aie_cols=40` (32 // 40 = 0), so the task-generation loop never runs. Not a bug — the test GEMM shape was just too small to fill one 40-wide group. Fixed by regenerating with `N=5120` (`= n * n_aie_cols`).
4. Recompiled: **`final_40col_v2.xclbin`, 421,440 bytes, `insts_40col_v2.txt`, 10,976 bytes** — a real, non-trivial 40-column/160-core GEMM design. Two "loop count overflow" warnings from the backend persisted through this build; didn't block compilation, but worth revisiting as a possible correctness risk before trusting output numerically.
5. Rebuilt a real test harness (`test_mt_gemm3.cpp`, the one `rebuild_mt_40col.sh` already used for exactly this purpose) — its own dependency headers (`helper.h`, `gemm_atb_layout.h`) had been deleted from the original `torch2aie/examples/...` location, but intact copies existed in `~/mlir-aie/programming_examples/ml/block_datatypes/`. Adapted for `N=5120` and the new xclbin/insts paths, compiled clean.

## The actual hardware result

```
terminate called after throwing an instance of 'xrt_core::system_error'
  what():  DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22): Invalid argument
```

Identical failure to the original `engine/fusion` crash — but this time against an xclbin genuinely built and compiled for 40 columns, not an 8-column-built one. **The box did not hang or wedge** — clean, fast SIGABRT, fully responsive immediately after.

## What this actually proves

The earlier working hypothesis ("stock 8-column xclbins colliding with a 40-column-configured resource solver") is **wrong**, or at least incomplete. A properly-built, properly-compiled 40-column xclbin hits the exact same `EINVAL` at hardware-context creation. This narrows the real blocker down to the resource allocation/partition layer itself — `aie2_xrs_load`/`aie2_alloc_resource` in the driver, or the closed firmware behind it — not anything about how the xclbin was compiled. This matches `amd-xdna-column-unlock-knowledge.md`'s own "Real Bottlenecks" list, bottleneck #2: *"Partition request — if you increase the total column count but don't also increase the partition size, it'll still only use [stock]."* The `aie2_max_col=40` override appears to change what the driver **reports** (`metadata.cols`) without actually widening what the resource solver will **allocate** a context against.

## Honest bottom line

Reached, and hardware-verified, exactly the point `2026-06-28-40column-npu2-compiler-design.md`'s own "Next Steps" never got to ("compile and test xclbin on hardware") — and got a real, clean, reproducible answer, not a hang. It's not a yes. The remaining blocker lives one layer deeper than the compiler or the metadata query: in the actual partition/context allocation path. That's `aie2_solver.c` / `aie2_xrs_load` territory in `~/xdna-driver` — genuine kernel-driver-internals work, a different and harder problem than anything solved tonight (compiler patch, MLIR template, dimension math). Stopping here rather than guessing further into closed-source resource-solver behavior.

## Artifacts, all preserved for whoever continues this
- `~/npu-sandbox/npu-infer/bf16_kernel_dev/col40_gemm/n2_core_40col.py` — the fixed, working 40-column template
- `~/npu-sandbox/npu-infer/bf16_kernel_dev/col40_gemm/final_40col_v2.xclbin` + `insts_40col_v2.txt` — the compiled artifacts
- `~/npu-sandbox/npu-infer/bf16_kernel_dev/col40_gemm/testharness/test_40col.cpp` — the hardware test, ready to re-run once the partition-allocation issue is addressed
