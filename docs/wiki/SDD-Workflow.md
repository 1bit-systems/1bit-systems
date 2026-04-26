---
phase: verified
owner: cartograph
---

# Spec-driven development workflow (1bit systems)

1bit systems runs on a spec-first loop. Every crate / kernel / service integration lands with a `docs/wiki/Crate-*.md` (or `docs/wiki/Why-*.md`) spec BEFORE code. Code implements the spec. Tests verify the spec. Agents are briefed against the spec, not ad-hoc prose.

Inspired by BMAD-Method (`docs.bmad-method.org`) phase gates + Kiro / Spec-Kit specification patterns. **We skip the frameworks** (Rule A — no extra runtime deps). The markdown-plus-clangd-plus-tests workflow already gives spec → code → verification; we borrow only the phase tags + agent routing.

## Four phases

| Phase | Content | Gate to next phase | Default reviewer specialist |
|---|---|---|---|
| `analysis` | Problem statement, invariants, non-goals | Invariants + interface shape approved in review | **magistrate** |
| `solutioning` | API signatures, file layout, test matrix | Test matrix covers every invariant | **cartograph** |
| `implementation` | Spec-cross-ref per commit, TODO list | All TODOs closed, green tests, clippy clean | **anvil** (kernel) / **scribe** (prose) |
| `verified` | Bench + parity numbers, shadow-burnin passed | Never — this is the end state | **sentinel** |

Every `docs/wiki/Crate-*.md` carries a YAML frontmatter block at the top:

```yaml
---
phase: analysis | solutioning | implementation | verified
owner: <specialist-name>          # who owns review at this phase
blocks: [Crate-foo, Crate-bar]    # optional: downstream specs that can't advance until this one verifies
---
```

## Phase transitions

Promotion is a commit. The commit message pattern:

```
spec(<crate>): promote from <old> → <new> — <reason>
```

Examples:
- `spec(1bit-helm): promote from analysis → solutioning — API signatures reviewed by cartograph 2026-04-20` (crate renamed from halo-gaia on the same day)
- `spec(halo-ep): promote from implementation → verified — 201 tests green, bit-exact parity vs Hip on 10k prompts`

If a spec needs to go BACK a phase (discovered-constraint rewrites the design), commit message says `spec(<crate>): demote from <old> → <new> — <reason>`. Demotions are normal; they're a sign the spec is learning.

## Required sections per phase

**analysis (minimum viable spec):**
- `## Problem`
- `## Invariants` — what MUST hold. One bullet per invariant.
- `## Non-goals` — what this explicitly does NOT do.

**solutioning adds:**
- `## Interface` — public API signatures, FFI contracts, CLI flags.
- `## Test matrix` — one row per invariant, one column per test name.

**implementation adds:**
- `## TODO` — checklist of tasks. PR commits tick them.
- `## Spec cross-ref` — table mapping spec sections to code files.

**verified adds:**
- `## Bench` — reproduced numbers (tok/s, PPL, etc.), box + config.
- `## Parity` — shadow-burnin byte-exact rate or parity oracle result.

## Agent briefs quote the spec

When I spawn a background agent (research or build), the prompt points at the spec file rather than re-describing the problem. Agents diff their output against the spec sections. Example prompt framing:

> Implement `cpp/helm/src/view/mod.rs` per `docs/wiki/Crate-1bit-helm.md` § Interface. Keep invariants §2 + §3 green. Non-goals in §5 are out of scope.

## Retroactive application

Modules already in tree without a matching spec file get one **the next time they're touched**. Currently unspec'd (as of 2026-04-20):

- 1bit-echo — `phase: implementation` already (Opus path, 10 tests). Needs a retrospective spec.
- 1bit-lemonade — `phase: implementation` already. Retrospective spec needed.
- 1bit-helm — renamed from halo-gaia on 2026-04-20 (AMD GAIA clash). Spec promoted analysis → solutioning alongside first egui/eframe scaffold.
- 1bit-landing — `phase: implementation` already. Retrospective spec needed.

Well-spec'd (templates for the above):
- `docs/wiki/Why-Ternary.md`
- `docs/wiki/Peak-Performance-Projection.md`
- `docs/wiki/NPU-Kernel-Design.md`
- `docs/wiki/Ternary-on-AIE-Pack-Plan.md`
- `docs/wiki/VPN-Only-API.md`
- `docs/wiki/Beta-10-Day-TTL.md`
- `docs/wiki/CPU-Lane-Plan.md`
- `docs/wiki/Halo-Whisper-Streaming-Plan.md`
- `docs/wiki/Medusa-Integration-Plan.md`
- `docs/wiki/Rotorquant-Default-Decision.md`
- `docs/wiki/Hermes-Integration.md`

## What we DON'T do

- Adopt a spec framework (BMAD / Kiro / Spec-Kit / OpenSpec). Adding tooling violates Rule A + adds build-time complexity.
- Gate trivial changes on a spec (one-line clippy fix, typo).
- Freeze specs at inception. Specs are living — demote + iterate when the code discovers a new constraint.

## Cross-refs

- `feedback_spec_driven_dev.md` in memory — policy pointer.
- CLAUDE.md § Hard rules — Rule A / B / C / D / E frame every spec.
- `crates/1bit-agents/src/lib.rs` — the 17 specialists that own phase reviews.
