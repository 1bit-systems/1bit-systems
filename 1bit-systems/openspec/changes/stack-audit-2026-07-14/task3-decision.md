# Task 3 decision: the 40-column contradiction (resolved 2026-07-14)

## The two documents aren't actually contradicting each other — they're answering different questions

- **`docs/superpowers/specs/2026-06-28-40column-npu2-compiler-design.md`** (dated Jun 28) proves the **MLIR-AIE compiler** can be patched to target 40 columns instead of 8 — two one-line hardcoded caps (`AIETargetModel.h`, `device/__init__.py`), no other code changes needed. Its own verification table only checks compiler-level acceptance (`aie-opt` bounds checking, tile counts reported) and its "Next Steps" section explicitly lists "compile and test xclbin **on hardware**" as *not yet done*. "Status: COMPLETE" refers to the compiler patch, not a working 40-column run on real silicon.
- **`bf16_kernel_dev/STEP5-INT8-32TILE-PLAN.md`** (commit `851b6bed`, Jul 13 — two weeks later) reports what happens when you actually try to run on hardware: **"Active firmware hard-caps columns at 8 (column_width>8 → EINVAL)."** This is a driver/firmware-level rejection at execution time, discovered after the compiler-side fix already existed.

## Decision

**The compiler patch is real and necessary, but not sufficient — it doesn't reach the actual blocker.** The 8-column limit is enforced in (or via) `amdxdna`/the NPU firmware at execution time, independent of what the compiler is willing to emit. STEP5 is the current, empirically-grounded status: **40 columns is blocked on this hardware/firmware as shipped.** The compiler-design doc should be marked superseded-in-part — its compiler patch is still useful groundwork (keep it, it's not wrong), but its "Status: COMPLETE" and 5x-throughput projections should not be read as "40 columns works," since the hardware test it names as a next step never happened, or happened later and hit the EINVAL wall documented in STEP5.

**Where the actual blocker lives is still open**: a search of `~/xdna-driver`'s open-source C driver didn't turn up an obvious `column_width > 8` check — it's likely enforced inside the closed firmware blob (`npu_7.sbin`), not the driver you control. If there's appetite to actually chase 40 columns further, the real next step is firmware reverse engineering (this is exactly what `npu_re_workspace`/`amdnpu-ps1p` — the PSP/AIE2 firmware disassembler already in this ecosystem — is built for), not more compiler work. That's a much bigger, riskier undertaking than the compiler patch was, and worth a deliberate go/no-go decision rather than picking it up by default.

## Action taken
- `STEP5-INT8-32TILE-PLAN.md` — current, no change needed.
- `2026-06-28-40column-npu2-compiler-design.md` — add a dated header note (see below) pointing to this decision, so it's not read in isolation as "40 columns works."
