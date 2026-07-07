---
name: swarm-dashboard
description: Real-time health dashboard for all running agents — what's active, what's stuck, what's done, resource consumption, and quick actions. Use to monitor multi-agent operations at a glance.
---

# Swarm Dashboard

When you have 3+ agents running, you need a dashboard. This skill provides
the patterns and tooling to see everything at a glance.

## Dashboard Views

### View 1: At-a-Glance (compact)

```
═══ Swarm Dashboard ═══ 2026-07-07 14:32:00 ═══

RUNNING (3)
  🟢 calm-reef  worker    "Fix fused_execute KV sync"    DeepSeek   ████░░ 42K/128K  4m32s
  🟢 keen-cove   reviewer  "Review KV cache changes"       Claude     ██░░░░ 28K/200K  1m15s
  🟡 bold-fox    worker    "Wire CLI entry point"          DeepSeek   █████░ 58K/128K  8m44s  ⚠ approach rot.

COMPLETED (5)
  ✅ silent-moon  scout    "Explore gpu_attn API"          NPU-local  3K tokens   0m42s  → 142 lines
  ✅ wild-grove   scout    "Explore npu_engine API"        NPU-local  2K tokens   0m38s  → 98 lines
  ✅ damp-reef    planner  "Synthesize scout findings"     DeepSeek   8K tokens   2m12s  → PLAN.md
  ✅ quiet-lake   scout    "Explore model_data.zig"        NPU-local  4K tokens   0m51s  → 210 lines
  ✅ swift-bay    worker   "Create model_data.zig"         DeepSeek  22K tokens   5m38s  → +320 lines

FAILED (1)
  ❌ dark-mesa    worker   "Create gpu_attn module"        DeepSeek   35K tokens  12m10s  OOM → retrying

QUEUED (2)
  ⏳ pending-1    worker   "Build verification"            DeepSeek   —
  ⏳ pending-2    reviewer "Final review"                  Claude     —

Budget: $3.42 / $25.00 (13.7%) | NPU-local: 23 calls (free)
```

### View 2: Per-Session Detail

```
═══ calm-reef (worker) ═══
Task:     "Fix fused_execute KV sync"
Provider: DeepSeek v4 | Tokens: 42K/128K | Cost: ~$0.15
Runtime:  4m32s | Status: running
Last tool: edit → engine/fusion/fused_execute.zig:215 (+12 lines)
Stalls:   0 | Retries: 0
Gate:     pending (not yet completed)
```

### View 3: Dependency Graph

```
                    ┌─────────────────────┐
                    │ swarm-140000         │
                    │ "Fused Runtime"       │
                    └──────┬──────────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      [silent-moon]  [wild-grove]   [quiet-lake]
      scout: gpu     scout: npu     scout: model
      ✅ done         ✅ done         ✅ done
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                    [damp-reef]
                    planner: synth
                    ✅ done
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
      [calm-reef]    [bold-fox]     [dark-mesa]
      worker: KV      worker: CLI     worker: attn
      🟢 running      🟡 warm rot.    ❌ OOM retry
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼
      [pending-1]    [pending-2]
      worker: build   reviewer: final
      ⏳ queued        ⏳ queued
```

### View 4: Resource Monitor

```
═══ Resources ═══

CPU:  ████████░░ 78%  (8 cores active of 16)
RAM:  ██████░░░░ 62%  (19.8G / 32G)
NPU:  ████░░░░░░ 38%  (19 TOPS / 50 TOPS)
GPU:  ██░░░░░░░░ 20%  (idle, no GPU tasks running)
Disk: ██░░░░░░░░ 22%  (120G / 540G)

Active Processes:
  npu-gpu-cpud        CPU: 2%   RAM: 45M   NPU daemon
  spec-decode/build/  CPU: 0%   RAM: 1.2G  idle
  flm-server          CPU: 1%   RAM: 380M  NPU FLM proxy
  lemond              CPU: 3%   RAM: 210M  chat UI
  pi (3 instances)    CPU: 12%  RAM: 3.1G  subagents
```

## Dashboard Protocol

### On Every Subagent State Change

```
1. Update dashboard state:
   - Session started → add to RUNNING
   - Session completed → move to COMPLETED
   - Session failed → move to FAILED with reason
   - Session stalled → flag in RUNNING with ⚠️

2. Check for blockers:
   - Any FAILED blocking QUEUED tasks? → retry or skip
   - Any RUNNING approaching rotation? → prepare checkpoint
   - Any RUNNING past expected runtime? → check for stall

3. Update budget counter

4. If all QUEUED have unmet dependencies → nothing can run,
   alert supervisor
```

### Quick Actions from Dashboard

| Situation | Action |
|-----------|--------|
| Session stalled >3 min | `kill` + `retry-simpler` |
| Session OOM | `retry-simpler` with smaller scope |
| All workers done | Spawn reviewers |
| Budget at 70% | Switch pending to NPU-local |
| CPU at 90%+ | Reduce concurrency |
| RAM at 85%+ | Kill lowest-priority session, retry later |
| Dependency chain stuck | Escalate to supervisor |

## Dashboard Command

```bash
# Full dashboard (all views)
bash ~/.pi/agent/skills/swarm-dashboard/dashboard.sh

# Compact at-a-glance only
bash ~/.pi/agent/skills/swarm-dashboard/dashboard.sh --compact

# Watch mode (refresh every 15s)
bash ~/.pi/agent/skills/swarm-dashboard/dashboard.sh --watch

# Export to JSON for programmatic use
bash ~/.pi/agent/skills/swarm-dashboard/dashboard.sh --json

# Show only RUNNING sessions
bash ~/.pi/agent/skills/swarm-dashboard/dashboard.sh --running

# Show dependency graph
bash ~/.pi/agent/skills/swarm-dashboard/dashboard.sh --graph
```

## Dashboard State File

State is persisted at `~/.pi/agent/cache/swarm-state.json`:

```json
{
  "swarm_id": "swarm-140000",
  "updated": "2026-07-07T14:32:00Z",
  "sessions": {
    "calm-reef": {
      "status": "running",
      "agent_type": "worker",
      "task": "Fix fused_execute KV sync",
      "provider": "deepseek",
      "tokens": 42000,
      "tokens_max": 128000,
      "runtime_s": 272,
      "stalls": 0,
      "retries": 0,
      "started": "2026-07-07T14:27:28Z"
    }
  },
  "budget": { "spent": 3.42, "limit": 25.00 },
  "queue": ["pending-1", "pending-2"],
  "resources": {
    "cpu_pct": 78,
    "ram_used_gb": 19.8,
    "ram_total_gb": 32,
    "npu_pct": 38,
    "gpu_pct": 20
  }
}
```

## Integration

- **babysitter**: Dashboard shows babysitter status (stall count, retry count)
- **context-rotation**: Dashboard shows token depth and rotation warnings
- **token-budget**: Dashboard shows real-time budget consumption
- **quality-gate**: Dashboard shows gate status (pending/passed/failed)
- **supervisor-orchestration**: Dashboard is the supervisor's primary monitoring tool
