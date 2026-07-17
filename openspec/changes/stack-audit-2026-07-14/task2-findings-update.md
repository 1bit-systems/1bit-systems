# Task 2 update (2026-07-14, later same day): the real root cause

## This supersedes the "concurrent hw_context" hypothesis in task2-findings.md

Found `~/amd-xdna-column-unlock-knowledge.md` — a detailed research log of a separate, ongoing effort (2026-07-13) to unlock the NPU from 8 to 40 columns at the driver/firmware level. Cross-checking it against live system state:

- The patched DKMS module (`updates/dkms/amdxdna.ko`, sig_key `56:67:41:...`) is **currently loaded**, not the stock in-tree one.
- Today's dmesg confirms it's active: `aie2_mgmt_fw_query: NPU UNLOCK: overriding metadata.cols from 8 to 40` → `NPU: 40 total_col (aie2_max_col=40, metadata.cols=40)`.
- The knowledge doc's own last session entry (2026-07-13 PM) ends with this exact configuration armed but **explicitly marked untested**: *"What happens if you pass max_col=40 ... on a Strix Halo that reports 40 cols? Does the firmware accept a 40-column partition, or does fw_patches_enable=1 wedge the PSP and hang the box? UNTESTED — this is the real risk."*

**The `engine/fusion/main.zig` crash reproduced earlier today is that untested scenario playing out.** `npu_engine_universal.cpp`'s xclbins (`~/npu-sandbox/npu-infer/build/int8/final_i8_*.xclbin`) were compiled against the stock 8-column partition shape. With the resource solver now configured for 40 total columns, the partition request those xclbins make no longer matches what the solver expects — hence `aie2_alloc_resource: Allocate AIE resource failed, ret -22`. This is not a concurrency bug in the C++ engine; it's a compile-time/runtime column-count mismatch. Good news buried in this: it fails *cleanly* (EINVAL, process aborts, box stays responsive) rather than wedging the PSP — that's real, if incidental, answer to the doc's open safety question, at least for this class of malformed request.

## Decision (2026-07-14): rebuild the inference pipeline for 40 columns

User confirmed: don't revert to the 8-column module, rebuild the pipeline to target 40 columns — this is the actual payoff of the original "unlock the 40 column" goal from earlier today.

## What's actually required (not done in this pass)

1. **Compile a minimal 40-column smoke test first**, not the full GEMM pipeline. The `2026-06-28-40column-npu2-compiler-design.md` spec's own "Verification Results" only checked compiler-level bounds (`aie-opt` accepting `tile(39, 2)`) — nobody has run an actual kernel at a column beyond 8 on this hardware yet. Do this before touching the real inference engine.
2. **Placement API mismatch found while attempting this**: the newer example designs in `~/mlir-aie/programming_examples/` use the IRON API (`Worker`, `ObjectFifo`, `@iron.jit`) which doesn't expose explicit column pinning in the same direct way the older raw-MLIR examples did (`Tile(col=N, row=M, ...)` is used for shim/PLIO routing in some examples, e.g. `passthrough_dmas.py`, but the simpler `passthrough_kernel.py` auto-places via `Worker`). Need to find or write a design that lets you force placement at a specific high column (e.g. col=39) to validate the toolchain end-to-end, using the `npu2_40_toolchain/aiecc_wrapper.sh` from the compiler-design doc.
3. Once a minimal kernel is verified working at a high column: port `n2_core_placed.py` (AMD's proven bf16 32-core template, already identified as the right structural template in STEP5-INT8-32TILE-PLAN.md) to a 40-column layout, following the same "swap dtype, keep tiling" approach STEP5 used to go from 8→32 cores, but now targeting columns instead of just rows.
4. Rebuild the 5 GEMM-variant xclbins (QKV, O, GU/G+U, D) at the new column count, repoint `npu_engine_universal.cpp`'s `xd` path at them, re-test with the same bounded-timeout methodology used to reproduce today's crash.

## Why I stopped here rather than push further tonight

Two independent reasons, not just caution for its own sake:
- The knowledge doc records real, expensive failure modes in this exact area this same week (a 33-minute hang requiring a hard power-cycle, from an unrelated Secure Boot step) — worth respecting that history rather than improvising against live hardware with unfamiliar IRON placement APIs at the tail of a long session.
- Getting the placement API right needs actual iteration/reading of IRON's tile-assignment docs, not a first-guess — worth doing carefully in a focused pass.

Concrete unblocked next step for whoever picks this up: read `~/mlir-aie/programming_guide/section-4/` (referenced by the passthrough examples) for how IRON API designs pin explicit tile/column placement, then build the minimal smoke test described in step 1 above.
