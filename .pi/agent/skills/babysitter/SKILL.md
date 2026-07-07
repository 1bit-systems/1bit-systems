---
name: babysitter
description: Agent watchdog — monitors running subagent sessions, detects hangs/loops/failures, auto-recovers with degraded retry. Use whenever spawning hands-free or dispatch subagents that need supervision.
---

# Agent Babysitter / Watchdog

Watches running subagent sessions and intervenes when they go wrong. No more
flying blind — every hands-free/dispatch session gets a watchdog.

## When to Use

- Any `interactive_shell` with `mode: "hands-free"` or `mode: "dispatch"`
- Any `subagent` call over 60 seconds
- Long-running swarm operations
- When you can't afford silent failures

## Watchdog Modes

### 1. Inline Babysitter (for interactive_shell sessions)

Wrap every hands-free/dispatch spawn with monitoring:

```
1. SPAWN: interactive_shell({ command: 'pi "Fix X"', mode: "hands-free" })
2. POLL every 90s: interactive_shell({ sessionId: "calm-reef" })
3. CHECK for stall signals in output
4. RECOVER if stalled — kill and retry with degraded task
```

**Stall signals** (any of these → session is stuck):
- Same output for 3 consecutive polls (>4 min of no progress)
- Output contains error loops (same error >3 times)
- Session runtime > 10 min with no tool calls visible
- Output contains: "I apologize", "Let me try again" repeated >2 times
- Session exit code is non-zero with empty output

### 2. Watchdog Script (for background monitoring)

```bash
bash ~/.pi/agent/skills/babysitter/watchdog.sh \
  --session-id calm-reef \
  --max-stall 180 \
  --max-runtime 600 \
  --on-fail "retry-simpler"
```

### 3. Swarm Babysitter (for issue-swarm.py)

When running `issue-swarm.py plan` with concurrency, the babysitter watches
all spawned sessions:

```bash
bash ~/.pi/agent/skills/babysitter/watchdog.sh \
  --swarm-id swarm-143000 \
  --concurrency 5 \
  --max-stall 240
```

## Recovery Strategies

| Strategy | What It Does | When To Use |
|----------|-------------|-------------|
| `retry-same` | Kill and respawn with identical task | Transient failure (timeout, OOM) |
| `retry-simpler` | Split task into smaller sub-tasks, retry one at a time | Task too complex for single agent |
| `retry-different-agent` | Retry with `worker` → `brainstormer` or vice versa | Wrong agent type for task |
| `escalate` | Stop babysitting, report failure to supervisor | Can't recover, needs human |
| `skip` | Log failure, move to next task | Non-critical task in a batch |

## Recovery Decision Tree

```
┌─ Session stalled?
│  ├─ Error loop? → retry-simpler (task likely too complex)
│  ├─ Timeout? → retry-same (transient)
│  ├─ OOM kill? → retry-simpler (split memory load)
│  ├─ Exit code non-zero? → retry-different-agent
│  └─ Unknown hang? → escalate after 2 retries
│
└─ Session completed but output empty?
   └─ retry-simpler with explicit "write results to file" instruction
```

## Babysitter Protocol

When you launch a subagent, always:

```
1. Record: sessionId, task summary, start time, max expected runtime
2. Set watchdog timer: max-stall = 180s, max-runtime = 600s
3. Poll at intervals (90s for hands-free, notified for dispatch)
4. On stall → apply recovery strategy
5. On completion → verify output (not empty, contains expected artifacts)
6. Log result to babysitter journal
```

## Babysitter Journal

Keep a running log at `~/.pi/agent/cache/babysitter-journal.jsonl`:

```json
{"session_id":"calm-reef","task":"Fix TS errors","agent":"pi","mode":"hands-free","started":"2026-07-07T14:00:00Z","status":"completed","runtime_s":245,"stalls":0,"retries":0,"output_lines":142}
{"session_id":"keen-cove","task":"Review security","agent":"claude","mode":"dispatch","started":"2026-07-07T14:05:00Z","status":"stalled","runtime_s":340,"stalls":2,"retries":1,"recovery":"retry-simpler","output_lines":0}
```

## Quick Integration

When using the `subagent-deployment` or `supervisor-orchestration` skills,
always activate the babysitter:

```
// Before any subagent call:
// 1. Set babysitter mode: inline
// 2. Record task in journal
// 3. Spawn with watchdog parameters

// After each poll:
// 1. Check for stall signals
// 2. Apply recovery if needed
// 3. Update journal
```

## Files

| Path | Purpose |
|------|---------|
| `SKILL.md` | This file — babysitter protocol and recovery strategies |
| `watchdog.sh` | Shell watchdog — monitors sessions, auto-recovers |
