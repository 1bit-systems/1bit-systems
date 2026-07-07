---
name: supervisor-orchestration
description: Multi-agent orchestration patterns — fan-out/fan-in, pipeline, supervisor gate, multi-perspective review, error recovery, and quality control. Use when coordinating multiple sub-agents, managing their outputs, handling failures, or needing structured patterns beyond simple single-agent calls. For tool mechanics and agent selection, see the subagent-deployment skill.
---

# Supervisor Orchestration — Patterns Guide

This skill teaches **orchestration patterns** for coordinating multiple sub-agents.
For **tool mechanics** (how to use the `subagent` tool vs `workflow` tool,
which agent to pick), see the `subagent-deployment` skill.

---

## Pattern 1: Fan-Out/Fan-In (Parallel)
```
You ──┬── agent A ──┐
      ├── agent B ──┤── You synthesize
      ├── agent C ──┤
      └── agent D ──┘
```

**When:** Multiple independent perspectives needed. All tasks run concurrently.

```javascript
// Using workflow tool (for analysis/synthesis tasks)
const results = await parallel([
  () => agent("Research approach X", { label: "approach X" }),
  () => agent("Research approach Y", { label: "approach Y" }),
])
const synthesis = await agent(
  "Synthesize these results: " + JSON.stringify(results),
  { label: "synthesis" }
)

// Using subagent tool (for file/bash tasks)
subagent({
  tasks: [
    { agent: "scout", task: "Find all API routes" },
    { agent: "scout", task: "Find all database models" },
  ]
})
```

---

## Pattern 2: Pipeline (Sequential Stages)
```
You ──→ Stage 1 ──→ Stage 2 ──→ Stage 3 ──→ You
```

**When:** Each stage depends on the previous. Output flows through a chain.

```javascript
// Using workflow tool (pipeline combinator)
const results = await pipeline(
  items,
  (item) => agent("Scout: " + item, { label: "scout " + item }),
  (prev) => agent("Plan: " + prev, { label: "plan" }),
  (prev) => agent("Execute: " + prev, { label: "execute" }),
)

// Using subagent tool (chain)
subagent({
  chain: [
    { agent: "scout", task: "Explore: {task}" },
    { agent: "planner", task: "Plan: {previous}" },
    { agent: "worker", task: "Execute: {previous}" },
  ]
})
```

---

## Pattern 3: Supervisor Gate (Gated Review)
```
You ──→ scout ──→ planner ──→ REVIEW ──→ worker ──→ reviewer ──→ REVIEW ──→ done
                    You approve          You review results
```

**When:** You need to approve intermediate results before proceeding. You act as a quality gate.

```
Step 1: subagent({ agent: "scout", task: "Explore the problem..." })
  → You review scout output. OK?

Step 2: subagent({ agent: "planner", task: "Plan: {scout_output}" })
  → You review plan. Approve?

Step 3: subagent({ agent: "worker", task: "Execute plan: {plan}" })
  → Wait for completion.

Step 4: subagent({ agent: "reviewer", task: "Review changes" })
  → You verify final output.
```

---

## Pattern 4: Multi-Perspective Review
```
You ──┬── reviewer (security) ──┐
      ├── reviewer (performance) ──┤── You synthesize
      ├── reviewer (correctness) ──┘
```

**When:** One piece of work needs multiple specialized reviews.

```
subagent({
  tasks: [
    { agent: "reviewer", task: "Review for SECURITY issues in: {diff}" },
    { agent: "reviewer", task: "Review for PERFORMANCE issues in: {diff}" },
    { agent: "reviewer", task: "Review for CORRECTNESS issues in: {diff}" },
  ]
})
// Then synthesize all three reviews
```

---

## Pattern 5: Brainstorm → Plan → Execute → Review
```
You ──→ brainstormer ──→ planner ──→ worker ──→ reviewer ──→ You
```

**When:** Facing an open-ended problem that needs creative solutions before execution.

```
Step 1: subagent({ agent: "brainstormer", task: "Generate approaches for {problem}" })
Step 2: You pick best approach.
Step 3: subagent({ agent: "planner", task: "Plan: {best_approach}" })
Step 4: subagent({ agent: "worker", task: "Execute plan: {plan}" })
Step 5: subagent({ agent: "reviewer", task: "Review output" })
Step 6: You verify.
```

---

## Error Recovery Strategy

When a sub-agent fails:

1. **Diagnose the failure** — Tool failure? Timeout? LLM error?
2. **Retry with different agent/tools** — Try `worker` instead of `scout`
3. **Fall back to yourself** — Do the work directly if sub-agent can't
4. **Report** — Include the failure in your synthesis. Partial results are still valuable.

```javascript
// Error-resilient orchestration pattern
const results = await parallel(tasks.map(t => () =>
  agent(t.prompt, { label: t.label })
))
const succeeded = results.filter(r => r !== null)
const failed = results.filter(r => r === null)

const synthesis = await agent(
  `Synthesize (${succeeded.length}/${tasks.length} succeeded).
   Succeeded: ${JSON.stringify(succeeded.map(r => r.verdict))}
   Failed: ${failed.map(f => f.label).join(', ')}`,
  { label: "final synthesis" }
)
```

---

## Quality Control Checklist

Before declaring done:

- [ ] Did all sub-agents complete successfully?
- [ ] Is the output consistent across sub-agents?
- [ ] Does the final synthesis cover all aspects requested?
- [ ] Are contradictory findings reconciled?
- [ ] Did you include partial results from failed agents?
- [ ] Would a second review pass catch anything missed?

---

## Pattern Summary

| Pattern | Shape | Use Case |
|---------|-------|----------|
| Fan-Out/Fan-In | Tree | Independent parallel research |
| Pipeline | Line | Sequential dependency chain |
| Supervisor Gate | Line with gates | Need human approval between stages |
| Multi-Perspective Review | Star | Need multiple specialized viewpoints |
| Brainstorm→Plan→Execute→Review | Loop | Open-ended → concrete execution |
