---
name: subagent-deployment
description: How to deploy and manage sub-agents — tool mechanics, agent capabilities, and when to use each mechanism. Use for choosing between subagent tool and workflow tool, finding the right agent for a task, or setting up parallel/chain execution. For orchestration patterns (fan-out, pipeline, supervisor gate), see supervisor-orchestration skill.
---

# Subagent Deployment — Mechanics Guide

This skill teaches the **mechanisms** for deploying sub-agents: the tools available,
what each agent is good at, and how to choose between them.
For **orchestration patterns** (fan-out/fan-in, pipeline gates, error recovery),
see the `supervisor-orchestration` skill.

---

## Two Deployment Mechanisms

### 1. The `subagent` tool (process-level)
Spawns isolated `pi` subprocesses. Best when:
- Tasks need **full tool access** (bash, read, write, edit)
- You want **truly isolated context windows** per sub-agent
- You need **parallel execution** with concurrency control
- The task is large enough to warrant a separate process (~seconds startup)

```javascript
// Single agent
subagent({ agent: "scout", task: "Find all auth routes" })

// Parallel (multiple independent tasks)
subagent({
  tasks: [
    { agent: "scout", task: "Find all API routes" },
    { agent: "scout", task: "Find all database models" },
  ]
})

// Chain (sequential, output feeds next)
subagent({
  chain: [
    { agent: "scout", task: "Explore the codebase for {task}" },
    { agent: "planner", task: "Using: {previous}, create a plan" },
    { agent: "worker", task: "Execute: {previous}" },
  ]
})
```

### 2. The `workflow` tool (in-process)
Runs deterministic JS workflows with embedded `agent()` calls. Best when:
- Tasks are **lightweight** (research, code review, analysis)
- You need **fast orchestration** without process overhead
- You want structured `parallel()` and `pipeline()` combinators
- Tasks are primarily analysis/thinking, not file manipulation

```javascript
workflow({
  script: `
    export const meta = { name: 'my_flow', description: 'X' }
    const [r1, r2] = await parallel([
      () => agent("Research approach A", { label: "approach A" }),
      () => agent("Research approach B", { label: "approach B" }),
    ])
    const synthesis = await agent(
      "Synthesize: " + JSON.stringify({ r1, r2 }),
      { label: "synthesis" }
    )
    return { ok: true, verdict: synthesis }
  `
})
```

---

## Available Agents

| Agent | Purpose | Best For | Tools |
|-------|---------|----------|-------|
| `scout` | Fast codebase recon, compressed context for handoff | Finding files, understanding structure, grep | read, grep, find, ls, bash |
| `planner` | Creates implementation plans from context | Architecture decisions, step-by-step plans | read, grep, find, ls |
| `worker` | General-purpose, isolated context, full capabilities | Implementation, file edits, complex tasks | (all default) |
| `reviewer` | Code review for quality/security | PR review, diff analysis, security audit | read, grep, find, ls, bash |
| `brainstormer` | Ideation, trade-off analysis, strategic planning | Open-ended problems, creative solutions | read, grep, find, ls |

**Quick selection guide:**
- Need to find something fast? → `scout`
- Need a plan before implementing? → `scout` → `planner`
- Need to implement? → `worker`
- Need code reviewed? → `reviewer`
- Need creative options? → `brainstormer`

---

## Workflow Prompts (Built-in Chains)

| Command | Flow |
|---------|------|
| `/implement <query>` | scout → planner → worker |
| `/scout-and-plan <query>` | scout → planner |
| `/implement-and-review <query>` | worker → reviewer → worker |

---

## Tool Selection Decision Tree

```
┌─ Does the task need bash/file manipulation?
│  YES → Use `subagent` tool (process-level isolation)
│  NO  → ↓
│
└─ Is it quick analysis / research / synthesis?
   YES → Use `workflow` tool (faster, lighter)
   NO  → ↓
│
└─ Is it a complex multi-step pipeline with gating?
   YES → Use `workflow` tool (parallel/pipeline combinators)
   NO  → ↓
│
└─ Default: `subagent` tool (most flexible)
```

---

## Guidelines

1. **Prefer `subagent` tool** when sub-agents modify files, run bash, or need isolated contexts.
2. **Prefer `workflow` tool** when sub-agents do analysis/thinking/synthesis only.
3. **Use `scout` first** before `planner` or `worker` — scouts produce compressed context cheaply.
4. **Chain agents** when output of one is input to another.
5. **Parallelize** when tasks are independent.
6. **Review after implementation** using `reviewer`.
7. For **patterns and orchestration** (fan-out, pipeline gates, supervisor patterns,
   error recovery, quality control), see the `supervisor-orchestration` skill.
