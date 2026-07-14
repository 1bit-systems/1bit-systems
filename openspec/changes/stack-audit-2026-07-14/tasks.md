# Implementation Tasks: Stack Audit 2026-07-14

> Proposal: `openspec/changes/stack-audit-2026-07-14/proposal.md`
> Full audit: `~/STATE-OF-THE-STACK-2026-07-14.md`

## Status

- [~] **Task 1**: Router consolidation -- CORRECTED, see task1-findings.md. token-router already unifies cascade/spec_decode/content_router/performance in ONE Rust binary via StrategyConfig -- not 3 separate routers, that framing was wrong. Real gap: PerformanceStrategy is a static per-model table, not live throughput-ranking+fallback. C++/Python duplicates in 1bit-systems still genuinely separate, both touched 07-12. Not implemented -- scoped for a dedicated session.
- [~] **Task 2**: NPU fusion -- REPRODUCED + CORRECTED, see task2-findings.md. Not "all-zero tokens then hangs" -- actual failure is SIGABRT at NPU hwctx creation (EINVAL), before token gen. Hypothesis: 4-5 concurrent hw_context allocs exceed the column/tile budget from Task 3. Also found an independent lm_head dequant corruption bug. Not fixed -- needs per-context instrumentation next.
- [x] **Task 3**: 40-column decision -- RESOLVED, see task3-decision.md. Compiler patch (Jun 28) is real but only reaches the compiler; STEP5 (Jul 13) found the actual blocker is firmware/driver EINVAL on column_width>8. Both docs updated with cross-references.
- [ ] **Task 4**: Repo/directory consolidation — one canonical location per component. Known duplicate pairs: `~/1bit` vs `~/projects/1bit`; `~/npu-infer` vs `~/npu-sandbox/npu-infer` vs `~/projects/1bit-systems/npu-infer`; nested `1bit-systems/1bit-systems/engine` inside the `1bit-systems` checkout itself; `~/token-router` vs `~/projects/token-router`. Move the losing side aside (don't hard-delete without confirming which has the latest commits) before removing. Add a CI check or pre-commit hook that fails on new duplicate top-level dirs.
- [ ] **Task 5**: Re-baseline ROCm HIP (113 tok/s) and DSpark (0.8 tok/s) -- see task5-findings.md, the ROCm HIP number has zero recorded methodology, DSpark needs a training-data decision not a re-run

## Task 5 notes (started 2026-07-14)

Looking for the benchmark commands/scripts behind these two numbers before re-running:
- ROCm HIP kernels: likely `benchmarks/` or `tools/` in `1bit-systems`, HIP kernel path (`src/`, `kernels/`) — needs the exact invocation used for the 113 tok/s figure, not yet located.
- DSpark: `spec-decode/` — the 2026-07-11 fix notes (checkpoint-path wiring bug + oversized `global_batch_size`) are in-repo; the 0.8 tok/s / 0% acceptance number was re-measured after both fixes, so this one may already be current. Worth confirming the *training data volume* problem (343 examples/420 steps, too little for a from-scratch draft head) is still the open blocker before spending more time on it, rather than re-running the same measurement.

## Verification Gates

| Gate | Cmd | Expected | Status |
|------|-----|----------|--------|
| Router unified | single `/v1/router` reports one active strategy set, not 3 separate services | 🔧 not started |
| NPU fusion works | `engine/fusion/main.zig` end-to-end run, non-zero tokens, no hang | 🔧 not started |
| 40-column resolved | one doc marked current, other marked superseded with date | ✅ done 2026-07-14 |
| No duplicate dirs | `find ~ -maxdepth 3 -iname '.git' -type d` shows one entry per remote URL | 🔧 not started |
| ROCm HIP re-measured | fresh run, methodology documented same as NPU v12 entry | 🔧 in progress |
| DSpark re-measured | confirm training-data blocker still the root cause | 🔧 in progress |

Any agent picking up a task: update its checkbox and the matching verification-gate row when done, and note what changed in `~/STATE-OF-THE-STACK-2026-07-14.md` so the audit doesn't go stale.
