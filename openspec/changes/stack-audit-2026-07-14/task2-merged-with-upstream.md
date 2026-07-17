
### Task 2: merged tonight's timeout fix with origin/main's upstream fixes (2026-07-14, late)

`origin/main` moved forward again with several genuinely relevant fixes while this branch's own hang fix was in progress:
- `95980489 fix(#153): dequantizeI8Block 6-byte header offset — scales now read correctly` — real root-cause fix for the C++ engine's Q4NX dequant (FLM format has a 6-byte/3×bf16 header at the start of each tile's scale region that wasn't being skipped).
- `faf946b2 fix(#152): replace hardcoded /home/bcloud xclbin path with NPU_XCLBIN_DIR env var` — walked back the earlier `int8_32tile_v3` change (see `task2-xclbin-dir-tested.md` — that was tested here tonight and crashes on this system) to a configurable env var defaulting to a relative path.
- `076f1b73 perf(#56): re-enable pipelined NPU GEMM` — restructured `I8Ctx` with separate `quantize_async`/`sync_and_launch`/`dequantize` helpers; `go()` now returns `bool` (interface was already prepared for failure signaling, just not wired to detect it).

None of these upstream commits included the bounded-timeout fix from earlier tonight — `origin/main`'s worker mode was still the unbounded `r.wait()` + fake-success stub.

**Action taken:** took `origin/main`'s current `npu_engine_universal.cpp` (with the dequant + xclbin-path fixes) and re-applied the same bounded-timeout pattern from `task2-hang-fixed.md` on top of their restructured `I8Ctx::go()` (now genuinely returns `false` on `ERT_CMD_STATE_TIMEOUT`, calls `r.abort()`, matching the bool signature that was already there but unused). Rewired the worker loop the same way — real `cq`/`co`/`cg`/`cd` calls with honest success/failure reporting instead of the stub.

Deployed to `~/engine/npu/build/npu_engine_universal` (still not git-tracked — see prior note), built clean, tested with `NPU_XCLBIN_DIR=/home/bcloud/npu-sandbox/npu-infer/build/int8` (the known-safe directory, not `int8_32tile_v3`).

**Result: still exit 0, still gracefully degrades** (`Prefill error: NpuWorkerError`, no hang, no crash) — the underlying "why doesn't EXEC_CMD's fence ever signal" question remains open, same as before. What's better now: the fix incorporates the real dequant root-cause fix and the more portable env-var xclbin path from upstream, so it's not just safe, it's also closer to correct.

**Separate, not-yet-addressed finding:** `model_data.zig`'s own dequant warnings (`"Non-finite dequant scale/zp"`, 653 occurrences per run) are still present after this merge. This is a **different dequantizer** than the one `#153` fixed — that fix was in the C++ engine (`npu_engine_universal.cpp`'s `dequantizeI8Block`), used for packing weights sent to the NPU. `model_data.zig` has its own, separate Zig-side dequant path (used for CPU-side weights: lm_head, embeddings, CPU fallback). If the same 6-byte-header root cause applies there too, that's a real, findable fix someone should check — not confirmed tonight.

Reference copy of the final merged source saved at `engine/npu/src/npu_engine_universal_DEPLOYED_reference.cpp` (superseding the earlier reference copy from `task2-hang-fixed.md`) since the actual deployment location still isn't git-tracked.
