---
name: scope-decomposer
description: Breaks large tasks into atomic, independently-assignable subagent tasks. Produces parallelizable work plans with scope boundaries, acceptance criteria, and dependency ordering. Use before any multi-agent orchestration.
---

# Scope Decomposer

Turns "build the fused runtime" into 5 scout, 8 worker, and 3 reviewer tasks —
each scoped tightly enough that a single agent can complete it without drifting.

## Decomposition Framework

### Phase 1 — Task Analysis

Before decomposing, classify the task:

| Dimension | Question | Classes |
|-----------|----------|---------|
| **Domain** | What kind of work? | code, docs, debug, research, refactor, review |
| **Coupling** | How interdependent? | independent, sequential, intertwined |
| **Risk** | What breaks if wrong? | low (docs), medium (new feature), high (refactor), critical (build system) |
| **Scope** | Size estimate? | S (<5 files), M (5-15 files), L (15-50 files), XL (50+ files) |

### Phase 2 — Atomic Decomposition Rules

Every decomposed task must:

1. **Touch ≤5 files** — if more, split by file group
2. **Take ≤10 minutes** — if longer, split by sub-goal
3. **Have a single clear deliverable** — "create X", "fix Y", "verify Z"
4. **Have explicit acceptance criteria** — "compiles", "all tests pass", "output matches expected"
5. **Include all needed context** — file paths, API references, patterns to follow
6. **Be independently verifiable** — can check success without running other tasks

### Phase 3 — Dependency Ordering

```
┌─ No dependencies → parallel (fan-out)
├─ Depends on 1 other → sequential chain
├─ Depends on N others → wait for all, then run
└─ Nothing depends on it → can run last or be skipped
```

### Phase 4 — Agent Assignment

| Task Type | Agent | Why |
|-----------|-------|-----|
| Find/explore/grep | `scout` | Fast, compressed context |
| Plan/architecture | `planner` | Structured thinking |
| Implement/create | `worker` | Full tool access |
| Review/audit | `reviewer` | Security/quality focus |
| Ideate/research | `brainstormer` | Creative exploration |

## Decomposition Patterns

### Pattern A: Explore → Plan → Execute → Review

Best for: New features, refactors, complex implementations.

```
Task: "Add GPU attention module to fused executor"

Phase 1 (parallel scouts):
  scout-1: Read engine/fusion/fused_execute.zig — identify NPU↔GPU handoff points
  scout-2: Read engine/fusion/gpu_attn.zig — document API surface
  scout-3: Read engine/npu/src/npu_engine.zig — find runSimple() call signature
  scout-4: Read engine/fusion/model_data.zig — find KV cache layout

Phase 2 (planner from scout outputs):
  planner: "Synthesize scout findings → create integration plan with file list,
            call sequence, and data flow. Output: PLAN.md"

Phase 3 (sequential workers):
  worker-1: "Create KV cache sync layer per PLAN.md §3.2"
  worker-2: "Wire GPU attention call into fused_execute.zig per PLAN.md §4.1"
  worker-3: "Update main.zig CLI flags per PLAN.md §5"

Phase 4 (parallel reviewers):
  reviewer-1: "Review KV sync for race conditions"
  reviewer-2: "Review GPU attention integration for correctness"
```

### Pattern B: Shotgun Fix (parallel workers on independent files)

Best for: "Fix all lint errors", "Update all copyright headers", "Add logging to all handlers".

```
Task: "Fix all compiler warnings in engine/"

Phase 1 (scout maps the territory):
  scout: "zig build 2>&1 | grep error → list all files with errors, group by file"

Phase 2 (parallel workers, one per file):
  worker-1: "Fix warnings in engine/npu/src/npu_engine_cb.cpp"
  worker-2: "Fix warnings in engine/fusion/dispatcher.zig"
  worker-3: "Fix warnings in engine/fusion/kernel_vulkan.zig"
```

### Pattern C: Research → Decide → Implement

Best for: Underspecified tasks, technology choices, approach decisions.

```
Task: "Choose best approach for NPU-GPU spec decode"

Phase 1 (parallel brainstormers):
  brainstormer-1: "Approach: synchronous co-processor. GPU waits for NPU."
  brainstormer-2: "Approach: async pipeline. GPU works ahead of NPU."
  brainstormer-3: "Approach: unified scheduler. Single dispatch for both."

Phase 2 (planner):
  planner: "Evaluate 3 approaches against: latency, throughput, complexity.
            Recommend one with rationale."

Phase 3 (workers execute the chosen approach)
```

### Pattern D: Swarm Harvest (fan-out identical tasks over a dataset)

Best for: Issue triage, benchmark collection, codebase audit.

```
Task: "Benchmark all 5 NPU models"

Phase 1 (parallel workers, one per model):
  worker-1: "Benchmark qwen3-0.6b-FLM → save to bench/qwen_0.6b.json"
  worker-2: "Benchmark qwen3vl-it-4b-FLM → save to bench/qwen_4b_vl.json"
  worker-3: "Benchmark gemma4-it-e2b-FLM → save to bench/gemma_4b.json"
  worker-4: "Benchmark llama3.1-8b-FLM → save to bench/llama_8b.json"
  worker-5: "Benchmark bonsai-1.7b → save to bench/bonsai.json"

Phase 2 (planner):
  planner: "Aggregate all bench/*.json → produce BENCHMARKS.md"
```

## Scope Fencing

Every worker task must include explicit scope fences to prevent drift:

```
<scope_fence>
- ONLY modify these files: [explicit list]
- Do NOT touch: [files that might be tempting to refactor]
- If you see adjacent issues, note them in output but DO NOT fix them
- Acceptance: [specific, verifiable condition]
- Timebox: [max expected minutes]
</scope_fence>
```

## Decomposition Checklist

Before launching any subagents:

- [ ] Every task touches ≤5 files
- [ ] Every task has explicit scope fence
- [ ] Every task has verifiable acceptance criteria
- [ ] Dependencies are mapped and ordered
- [ ] Parallel tasks are truly independent (no shared files)
- [ ] Each task has the right agent type assigned
- [ ] Human review gates are placed at high-risk handoffs
- [ ] Babysitter watchdog is configured for each session
- [ ] Quality gate will run after each worker completes

## Integration

- **supervisor-orchestration**: Decomposer produces the task plan → orchestrator executes it
- **babysitter**: Every decomposed task gets a watchdog
- **quality-gate**: Every worker task's output passes through the gate
- **result-synthesizer**: Parallel results from decomposer fed to synthesizer
