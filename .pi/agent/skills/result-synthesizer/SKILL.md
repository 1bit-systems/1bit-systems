---
name: result-synthesizer
description: Merges outputs from parallel subagents into a coherent synthesis — detects conflicts, resolves overlaps, reconciles contradictions. Use after fan-out/fan-in patterns or any time multiple agents produce results on the same topic.
---

# Result Synthesizer

Fan-out gives you N results. The synthesizer merges them into one coherent
output — finding conflicts, resolving overlaps, and producing actionable conclusions.

## When to Use

- After any `parallel()` / fan-out subagent call
- After swarm operations (issue-swarm.py)
- When multiple reviewers produce overlapping findings
- When combining scout outputs before passing to planner
- Anytime you have >1 agent result on the same topic

## Synthesis Layers

### Layer 1 — Deduplication

Same finding from multiple agents → merge into one, note consensus strength.

```
Agent A: "Fix: bounds check in tensor_read() at kernel.cpp:142"
Agent B: "Fix: add bounds validation to tensor_read() at kernel.cpp:142"
Agent C: "Critical: kernel.cpp:142 — tensor_read() missing bounds check"

→ MERGED: "tensor_read() at kernel.cpp:142 missing bounds check (found by A, B, C — strong consensus)"
```

### Layer 2 — Conflict Detection

Different agents say conflicting things → flag, don't silently merge.

**Conflict types:**

| Type | Example | Resolution |
|------|---------|------------|
| **Factual** | Agent A: "uses Vulkan" / Agent B: "uses ROCm" | Check source of truth, pick correct one |
| **Approach** | Agent A: "rewrite in Zig" / Agent B: "patch existing C++" | Flag for human decision |
| **Priority** | Agent A: "fix crash first" / Agent B: "add tests first" | Note trade-off, recommend order |
| **Scope** | Agent A: "change 3 files" / Agent B: "change 8 files" | Agent A missed something or Agent B over-scoped |

### Layer 3 — Coverage Analysis

What was covered vs. missed?

```
Requested: "Review for security, performance, correctness"
Agent 1 (security): ✅ covered
Agent 2 (performance): ✅ covered
Agent 3 (correctness): ⚠️ partial — only checked engine/, missed tools/
Agent 4 (docs): ❌ no agent assigned

→ Gap: correctness review incomplete, docs not reviewed
```

### Layer 4 — Confidence Scoring

Assign confidence to each finding based on consensus and agent quality.

| Level | Criteria |
|-------|----------|
| **High** | 3+ agents agree, or verified against source code |
| **Medium** | 2 agents agree, no dissent |
| **Low** | Single agent, unverified |
| **Uncertain** | Agents disagree, needs investigation |

### Layer 5 — Actionable Output

Produce a structured synthesis document:

```markdown
## Synthesis: [Topic]

### Consensus Findings (High Confidence)
- [Finding 1] (agents: A, B, C)
- [Finding 2] (agents: A, B)

### Conflicts (Requires Decision)
- [Conflict 1]: Agent A says X, Agent B says Y
  Recommendation: [prefer X because...]
- [Conflict 2]: Agent C says Z, Agent D says W
  Recommendation: [need more investigation]

### Coverage Gaps
- correctness review incomplete (engine/tools/ not checked)
- no agent reviewed test coverage

### Next Actions (ordered by priority)
1. [Action 1 — from high-confidence finding]
2. [Action 2 — resolve conflict]
3. [Action 3 — fill coverage gap]
```

## Synthesis Protocol

```
1. Collect all agent outputs (raw)
2. Extract findings from each (normalize format)
3. Layer 1: Deduplicate (hash findings, merge identical)
4. Layer 2: Detect conflicts (cross-reference all pairs)
5. Layer 3: Coverage analysis (what was asked vs. what was delivered)
6. Layer 4: Confidence score each finding
7. Layer 5: Produce structured synthesis
8. Route conflicts to supervisor for decision
```

## Conflict Resolution Patterns

### Factual Conflict ("A says X, B says Y, truth is Z")

```
1. Read the source file referenced by both agents
2. Determine ground truth from code
3. Flag the incorrect agent — this agent may have other errors
4. Use correct fact in synthesis
```

### Approach Conflict ("A says rewrite, B says patch")

```
1. Extract pros/cons from each agent
2. Build a decision matrix:
   | Criterion | Rewrite | Patch |
   |-----------|---------|-------|
   | Risk      | High    | Low   |
   | Speed     | Slow    | Fast  |
   | Quality   | Better  | Good  |
   | Scope     | Large   | Small |
3. Recommend with rationale
4. Flag for human if risk × impact is high
```

### Priority Conflict ("A says crash first, B says tests first")

```
1. Assess dependencies: does fixing the crash enable adding tests?
2. Recommend order: what unblocks the most downstream work?
3. Default: fix blocking issues first, then quality improvements
```

## Integration

- **scope-decomposer**: Decomposer produces tasks → agents produce results → synthesizer merges
- **supervisor-orchestration**: Synthesizer is the "fan-in" step after every fan-out
- **quality-gate**: Synthesizer output itself passes through gate (are all coverage gaps addressed?)
- **babysitter**: Failed agent results are included in synthesis (partial results are valuable)

## Quick Reference

```javascript
// After fan-out:
const results = await parallel([
  () => agent("Review security of {diff}", { label: "security" }),
  () => agent("Review performance of {diff}", { label: "perf" }),
  () => agent("Review correctness of {diff}", { label: "correctness" }),
])

// Synthesize:
const synthesis = await agent(`
  You are a result synthesizer. Merge these review findings:
  ${JSON.stringify(results)}
  
  Produce:
  1. Consensus findings (what all reviewers agree on)
  2. Conflicts (where reviewers disagree)
  3. Coverage gaps (what wasn't reviewed)
  4. Next actions (ordered by priority)
  
  Use the synthesis protocol: deduplicate → detect conflicts → coverage → confidence → output.
`, { label: "synthesis" })
```
