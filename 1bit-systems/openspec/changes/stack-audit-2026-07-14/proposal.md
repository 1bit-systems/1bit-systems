# Stack Audit 2026-07-14: Close the Vision Gap

## Problem

An architecture audit across all 7 active repos (`1bit-systems`, `zinc`, `token-router`, `npu-infer`, `strixhalo-npu-setup`, `npu_re_workspace`, `xdna-driver`) found five concrete gaps between the stated vision — one binary, zero config, a single router that picks the best backend transparently, native 1-bit/ternary everywhere — and what's actually running today. Full findings: `~/STATE-OF-THE-STACK-2026-07-14.md` (or the styled version, `~/1bit-systems-report.html`).

Short version: the core "one binary, auto-detect, zero config" claim is real and works (`zaya_server.cpp`, 207 KB). Everything downstream of "which backend runs this" does not yet match the vision:

1. Three separate, uncoordinated routers exist (`cascade` in `token-router`, `spec_decode` and `content` in `1bit-systems`) instead of one.
2. NPU+GPU fusion — the mode closest to "NPU or NPU+GPU fuse" in the vision — is currently broken (all-zero tokens, hangs on the real CLI). The 291 tok/s figure attached to it predates the regression.
3. The 40-NPU-column target has two documents in the same repo (`npu-infer`) disagreeing on whether it's firmware-blocked or actively in progress.
4. Repo/directory duplication is structural, not incidental — nested dirs inside `1bit-systems` itself, plus multiple stale clones across `~/`. A prior dedup attempt didn't fully stick.
5. Two "raw"/"reported" benchmark numbers (ROCm HIP kernels: 113 tok/s, DSpark: 0.8 tok/s) haven't had the same correctness scrutiny that just caught real bugs in NPU v12 and the C++ auto-detect path this same week.

## Solution

Five independently-scoped tasks below. Any agent picking this up should read the relevant section of the audit report first, then work the task — they don't depend on each other and can be picked up in any order or by different agents/sessions.

## Out of Scope

- Rewriting the whole router from scratch (Task 1 folds existing strategies into `cascade`, it doesn't replace it)
- New NPU column-count work beyond resolving which of the two existing documents is current (Task 3 is a decision, not new engineering)
- Any BIOS/firmware-level changes (resolved separately, see `strix-halo-gmktec-evo-x2` commit `5ec3c17`)
