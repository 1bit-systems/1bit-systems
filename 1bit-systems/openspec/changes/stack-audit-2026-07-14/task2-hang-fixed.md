# Task 2: the hang is fixed (2026-07-14, late session)

## Result

**Before:** `engine/fusion/main.zig` hung indefinitely (exit 124 under a 30s `timeout`, would hang forever unbounded) on every run, blocked in `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` on `/dev/accel/accel0` — the NPU worker subprocess submits a real compute job (`DRM_IOCTL_AMDXDNA_EXEC_CMD` returns 0/success) but its completion fence never signals.

**After:** same reproduction now exits cleanly with code 0. The worker reports honest failure after an 8s bounded wait instead of blocking forever; the Zig side receives `NpuWorkerError` as designed and the run completes instead of hanging.

## What was actually wrong (two separate bugs, not one)

1. **The deployed binary doesn't match any source file that was easy to find.** `/home/bcloud/engine/npu/build/npu_engine_universal` — the exact path hardcoded as `DEFAULT_NPU_ENGINE` in `main.zig` — is **not tracked by git at all**. `~/engine/` has no `.git`. It's a completely separate, untracked directory tree from the `1bit-systems` repo (which has its own, differently-structured copy of `npu_engine_universal.cpp` with `quantize_async`/`sync_and_launch` split, not present in the deployed version). This is real, separate technical debt: the actual production dependency for the fusion engine lives outside version control.
2. **`I8Ctx::go()`'s `r.wait()` call blocked forever** (XRT's default timeout is 0 = block until complete) with no bound and no CPU-fallback signaling on failure. Confirmed via `strace -f -tt` on a live hung process, correlating `read(0, "\1\0\0\0...")` (the worker protocol header: op=1/QKV, layer=0, batch=1, in_dim=1024) directly to the `EXEC_CMD`+`SYNCOBJ_TIMELINE_WAIT` sequence that followed within microseconds — i.e. the very first QKV projection request is what hangs, not something deep in the pipeline.

## The fix

In `~/engine/npu/src/npu_engine_universal.cpp` (the deployed source — a copy is saved at `engine/npu/src/npu_engine_universal_DEPLOYED_reference.cpp` in this repo since the real location isn't git-tracked):

1. Added `I8Ctx::go_bounded(..., unsigned timeout_ms=8000)` — same as the existing `go()` but calls `r.wait(timeout_ms)` (XRT supports this — returns `ert_cmd_state`, `ERT_CMD_STATE_TIMEOUT` on timeout per `xrt_kernel.h`) instead of the unbounded `r.wait()`. On timeout, calls `r.abort()` (XRT's documented recovery path) and returns `false` instead of hanging. `go()` itself is left as-is for non-worker callers that have no fallback to use anyway.
2. Rewired the worker-mode dispatch loop (previously a stub that unconditionally wrote `resp[2]={0,out_dim}` — fake success — with zeroed data, silently feeding garbage into every downstream layer) to actually call the corresponding `I8Ctx` (`cq`/`co`/`cg`/`cd`) via `go_bounded()`, and report **honest** status: `resp[2]={ok?0u:1u, out_dim}`, only writing the payload on success. This matches exactly what `fused_execute.zig`'s `NpuSubprocess.call()` already expects (`if (resp_hdr[0] != 0) return error.NpuWorkerError;`) — the CPU-fallback infrastructure (`npu_broken` flag, checked/set in 6+ places) was already correct and complete, it just never had a real failure signal to react to.
3. Rebuilt directly (`g++ -std=c++23 -O3 ...`, since `build_npu.sh` doesn't actually rebuild this specific binary — another small gap) and redeployed to the exact path `main.zig` uses.

## Known rough edge, not yet resolved

`r.abort()` on a timed-out run throws an internal XRT exception in the child process: `terminate called after throwing an instance of 'std::runtime_error': qds_device::wait() unexpected command state`. The overall run still completes (exit 0) rather than hanging, but this should be wrapped in a try/catch so the abort path itself can't take down the worker process uncleanly. Also, `main.zig`'s top-level `executor.prefill(...) catch |err|` prints "Prefill error: NpuWorkerError" and does not appear to retry/degrade per-layer the way the decode loop's individual QKV/O-proj/FFN calls do — worth checking whether prefill has the same graceful degradation as decode, or needs it added.

## Why the NPU job itself never completes — still genuinely unknown

This fix makes the failure bounded and recoverable; it does not explain *why* `EXEC_CMD` succeeds but the fence never signals. That's still open, and (same as the firmware column-unlock work) is very likely a deeper hardware/driver/firmware question rather than something fixable in this call site. Did not chase it further tonight — the bounded-timeout fix converts an unbounded hang into a bounded, recoverable failure, which is the practically important outcome regardless of the underlying cause.
