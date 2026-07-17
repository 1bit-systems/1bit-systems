# Task 2 findings (2026-07-14): NPU fusion — reproduced, and it's not what the README says

## Correction to the original bug report

The README/benchmarks.json describe this as "all-zero tokens, then hangs." Reproduced today (built `engine/fusion/main.zig` fresh with `zig build -Drelease=true`, ran `./zig-out/bin/fused-engine --debug --max-tokens 5`) and got a **different failure**: it never reaches token generation at all. It crashes with `SIGABRT` during prefill, at NPU hardware-context creation:

```
terminate called after throwing an instance of 'xrt_core::system_error'
  what():  DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22): Invalid argument
  Prefill error: EndOfStream
```

dmesg confirms, with more detail:
```
amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_xrs_load: create context failed, ret -22
amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_alloc_resource: Allocate AIE resource failed, ret -22
amdxdna 0000:c6:00.1: [drm] *ERROR* aie2_hwctx_init: Alloc hw resource failed, ret -22
amdxdna 0000:c6:00.1: [drm] *ERROR* amdxdna_drm_create_hwctx_ioctl: Init hwctx failed, ret -22
```

Ruled out "stale process holding the device" (checked `ps`/`lsof` — nothing else touching the NPU at the time).

**Working hypothesis, not yet confirmed:** `engine/npu/src/npu_engine_universal.cpp` opens **4–5 separate `xrt::hw_context` objects simultaneously** — one per GEMM shape variant (QKV, O, GU-or-G+U, D; see `I8Ctx cq, co, cg, cd` + optional `cu_ptr` in that file). Each hw_context creation calls into the same AIE2 resource allocator that Task 3 found hard-caps at 8 columns. If the resource solver can't satisfy 4–5 concurrent context allocations against that column budget, this crash is the same root constraint as the 40-column finding, just hit from a different angle (concurrent multi-context allocation instead of a single wide one). Not confirmed — would need to instrument which of the 4–5 `I8Ctx.init()` calls actually fails (add a printf per call, or run under `strace -e ioctl`), which I didn't do — this is the concrete next debugging step, not a fix.

## Second, independent, unrelated bug found on the way there

Before even reaching NPU init, model loading spams thousands of warnings dequantizing the I8 `lm_head` tensor (18992 tiles):

```
warning(model_data): Implausible dequant scale/zp at row=994 group=0 (scale=-20436880000000000000000000000000000000, zp=0.0057373047); zeroing group
```

Scale values escalate from plausible (~3.5) to 10^15–10^38 magnitude, including at least one `nan`. The code has a safety net that detects and zeros these ("Implausible... zeroing group"), so it doesn't crash — but it means large parts of the LM head weight matrix are silently zeroed at load time. This alone would degrade final-layer output quality independent of the NPU crash above, and is worth fixing regardless of the hwctx issue. Location: `model_data.zig`, I8 lm_head dequant path — the row/group indices repeating (`row=992/993/994` recurring many times with wildly different scale each time) suggests a stride or buffer-bounds bug reading past valid quant-metadata into adjacent memory, not a data-quality issue with the model file itself. Not yet isolated to a specific line.

## What's confirmed vs. still open

| Claim | Status |
|---|---|
| "All-zero tokens, then hangs" (original report) | Not reproduced as described — actual failure is an earlier SIGABRT crash at NPU context creation, before any tokens would be generated |
| NPU hwctx creation fails with EINVAL | Confirmed, reproducible, dmesg-verified |
| Root cause = concurrent multi-context resource exhaustion vs. 8-column cap | Plausible, well-supported, **not confirmed** — needs per-context instrumentation |
| LM-head dequant scale corruption | Confirmed, reproducible, real bug, independent of the above |
| PR #42's tokenizer/decode-loop fix | Not evaluated — never reached, crashes upstream of that code path now |

## Next steps for whoever picks this up
1. Instrument `npu_engine_universal.cpp`'s 4-5 `I8Ctx.init()` calls (or the `cu_ptr` one) to find exactly which context creation call fails — confirms or kills the concurrent-context hypothesis.
2. If confirmed: either serialize context creation/teardown (one GEMM variant's hw_context at a time instead of all 4-5 live simultaneously) or investigate whether the AIE2 resource solver has a documented concurrent-context limit to design around.
3. Separately, fix the lm_head dequant stride/bounds bug in `model_data.zig` — independent of (1) and (2), and currently masked by the safety-net zeroing so it doesn't crash, just silently corrupts output quality.
4. Re-run the README's benchmark methodology (5-token smoke test from PR #42) only after both of the above are actually fixed — the 291 tok/s figure and PR #42's claims can't be evaluated until the engine gets past prefill at all.
