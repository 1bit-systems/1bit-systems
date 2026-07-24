
### Task 2 addendum: tested origin/main's xclbin-directory fix — reverted, not compatible

`origin/main` commit `268cd335` changed `npu_engine_universal.cpp`'s xclbin base directory from `.../build/int8` to `.../build/int8_32tile_v3`, claiming this (paired with a 5s timeout) made the custom engine work.

Applied the same directory change on top of tonight's bounded-timeout fix and tested: **it does not work on this system's current (stock, post-`apt reinstall`) configuration.** Instead of the graceful bounded-timeout failure, it reproduces `DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22)` — the exact same hw-context-creation crash from earlier tonight's 40-column investigation.

**Working theory:** `int8_32tile_v3`'s xclbins are almost certainly built for the 4-row/32-core partition shape (STEP5-INT8-32TILE-PLAN.md's design — 8 cols × 4 rows), a different hardware-context request than the single-row shape `I8Ctx::init()` currently asks for. Whatever environment `origin/main`'s author tested this in (possibly with the 40-column unlock or a different partition config active at the time) may not match this system's current stock state — same class of environment-dependence that caused tonight's earlier confusion. Not confirmed, not investigated further.

**Reverted** the xclbin directory back to `.../build/int8` and rebuilt. Confirmed back to the known-safe, graceful-failure state (exit 0, `NpuWorkerError`, no crash).

**Recommendation for whoever reconciles the two branches:** do not merge `origin/main`'s xclbin-directory change as-is without first confirming which partition shape `int8_32tile_v3`'s xclbins were actually compiled for, and whether `I8Ctx::init()`'s hw_context request needs to change to match (rows/cols), not just the file path.
