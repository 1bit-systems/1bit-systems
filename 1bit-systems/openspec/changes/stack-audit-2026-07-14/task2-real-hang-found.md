# Task 2, corrected a second time: found the actual hang (2026-07-14, stock system)

## The crash reported earlier tonight was environment-specific, not the real bug

Re-ran `engine/fusion/main.zig` after the system was restored to genuine stock (8 columns, `apt reinstall`-verified driver/firmware). **The `EINVAL`/`AIE2_STATUS_MGMT_ERT_NOAVAIL` crash does not reproduce under stock conditions.** It gets past NPU hardware-context creation cleanly this time. That crash was real, but it was an artifact of testing against the 40-column-unlocked driver/firmware — not the bug the README describes.

## The actual bug: a real hang, empirically located

Under stock conditions, the process hangs instead — matching the README's original description much more precisely than the crash did. Reproduced with correct exit-code capture this time (first attempt was contaminated by piping through `tail`, which reports its own exit code, not the child's): **exit code 124, killed by `timeout` after 30s.**

Caught it live with `strace` while hung (process tree: `fused-engine` (parent, Zig) → `npu_engine_universal --worker` (child, C++)):

- **Parent** (`fused-engine`): blocked in `readv(12, ...)` — waiting to read the child's response over the stdout pipe. Expected; it sent a request and is waiting.
- **Child** (`npu_engine_universal`): blocked in `ioctl(3, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, ...)`. `fd 3` confirmed via `/proc/<pid>/fd/3` to be `/dev/accel/accel0` — **the NPU device itself.**

**The child successfully creates the NPU hardware context and submits a real compute job (op 1/2/3/5 — QKV/OPROJ/GATEUP/DOWN per the worker protocol in `npu_engine_universal.cpp`), but the job's completion sync/fence object never signals.** This is not a crash, not a driver-config issue, not related to column count at all — it's a genuine hang waiting on hardware to report job completion that never arrives. Process state confirmed `S` (sleeping/blocked), not spinning.

## What this rules out and narrows down

- Not the 40-column unlock (reproduces on genuinely stock 8-column hardware).
- Not the header/IPC protocol between Zig and C++ (`hdr = [4]u32{...}` on the Zig side matches `uint32_t hdr[4]` on the C side exactly, 16 bytes, no mismatch).
- Not stuck before submission — it gets all the way to a real hardware job submission and wait.
- **Is** something in the specific op's command construction/DMA setup that the NPU either never actually processes to completion, or processes but fails to signal — need to determine which op (1/2/3/5) is the first one requested and whether the bug is universal across all four or specific to one.

## Next steps for whoever continues this
1. Add a print statement (or use `--debug` output already present) right before each `call()` in `fused_execute.zig` to identify exactly which op/layer was in flight when the hang occurred — narrows from "the worker protocol hangs" to "op N specifically hangs."
2. Check `amdxdna`/`xrt-smi` for any built-in job/queue introspection (`xrt-smi examine` has device state; there may be a way to see in-flight/stuck jobs on `accel0` without needing to kill the process first).
3. Compare against the standalone `npu_engine_universal` test that worked earlier tonight (the one I ran directly, non-worker-mode, that got through prefill fine in the very first crash-investigation pass) — worker mode specifically may have a bug the standalone path doesn't hit, since they likely share most of the code but differ in the request-loop/sync-object handling.
4. This is real, tractable software debugging (not firmware/hardware RE) — much safer to keep investigating than the column-unlock firmware work.
