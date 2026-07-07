---
name: context-rotation
description: Manages agent context windows — checkpoint, compact, rotate sessions before hitting limits. Prevents context overflow, preserves critical state, and enables long-running multi-agent workflows.
---

# Context Rotation Manager

Agents burn context. Without rotation, sessions degrade into loops, lose earlier
context, and produce garbage. This skill manages context proactively.

## When Context Rotation Matters

Context limits by provider:

| Provider | Max Context | Degradation Starts | Rotation Threshold |
|----------|------------|-------------------|-------------------|
| DeepSeek v4 | 128K | ~80K | 60K tokens |
| Claude Opus | 200K | ~120K | 90K tokens |
| Codex 5.5 | 200K | ~100K | 80K tokens |
| NPU-local (qwen 0.6B) | 32K | ~20K | 16K tokens |

**Rotation threshold = 50% of degradation point** — rotate before quality drops.

## Rotation Strategies

### Strategy 1: Checkpoint → Fresh Session → Resume

Best for: Single long-running task that outgrows context.

```
1. Agent at 55K tokens → approaching rotation threshold
2. CHECKPOINT: Agent writes state file:
   - What's been done
   - What remains
   - Key decisions made
   - Files modified (with current state)
   - Current blockers
3. SPAWN fresh session with checkpoint as initial prompt
4. New session loads checkpoint, continues from where old left off
5. Old session terminates
```

**Checkpoint format** (`~/.pi/agent/cache/checkpoint-{task}.md`):

```markdown
# Checkpoint: Fused Runtime Integration
Rotation: 1 | Time: 2026-07-07T14:30:00Z | Tokens: 58,234

## Completed
- [x] model_data.zig — loads Q4NX, extracts embeddings, norms, RoPE
- [x] gpu_attn.zig — Vulkan flash attention module compiles
- [x] npu_engine.zig — runSimple() interface defined

## In Progress
- [ ] fused_execute.zig — KV cache sync between NPU and GPU (50% done)
  - NPU-side KV write: done
  - GPU-side KV read: done
  - Synchronization protocol: IN PROGRESS
  - Current file: /home/bcloud/engine/fusion/fused_execute.zig:215

## Remaining
- [ ] main.zig — CLI entry point wiring
- [ ] Build verification
- [ ] 15 test suite pass

## Key Decisions
- KV cache uses shared BOs, not copies (latency-critical)
- Pipeline overlap: NPU layer N while GPU does attention N-1
- Batch size fixed at M=16 for now

## Files Changed
- engine/fusion/model_data.zig (+320 lines)
- engine/fusion/gpu_attn.zig (+180 lines)
- engine/fusion/npu_engine.zig (+45 lines)
- engine/fusion/fused_execute.zig (+150 lines, in progress)

## Blockers
None
```

### Strategy 2: Parallel Rotation (for fan-out workloads)

Best for: Multiple independent subagents, each with their own context.

```
Session A (scout): 8K → no rotation needed
Session B (worker): 45K → approaching, checkpoint
Session C (worker): 62K → ROTATE NOW
Session D (reviewer): 12K → no rotation needed
```

Track per-session token counts and rotate individually. No need to rotate all.

### Strategy 3: Compaction (in-place)

Best for: Quick context reduction without new session overhead.

```
1. Agent at 40K tokens → still room but accumulating
2. COMPACT: Summarize conversation so far into 2K token summary
3. Replace earlier context with summary
4. Continue in same session
```

**Compaction summary format:**

```
<context_summary>
Task: [one-line description]
Completed:
- [done item 1]
- [done item 2]
In progress: [current item, exact file:line]
Key decisions: [decision 1], [decision 2]
Files modified: [list with line counts]
Next: [immediate next step]
</context_summary>
```

### Strategy 4: Split and Delegate

Best for: Task too large for single context — split into multiple independent sessions.

```
Original: "Refactor entire engine/npu/ directory" → 120K tokens estimated

Split into:
  Session 1: "Refactor engine/npu/src/npu_engine_cb.cpp" (est. 30K)
  Session 2: "Refactor engine/npu/src/npu_engine_v12.cpp" (est. 25K)
  Session 3: "Refactor engine/npu/src/npu_engine_universal.cpp" (est. 35K)
  Session 4: "Refactor engine/npu/src/model_reader.zig" (est. 20K)
```

No dependencies between sessions (different files), so all can run in parallel.
No rotation needed — each fits within context.

## Rotation Protocol

### Before Every Agent Call

```
1. Estimate tokens needed: sum of file sizes to read + prompt + expected output
2. Check against rotation threshold for provider
3. If estimated > threshold:
   a. Can we split into smaller tasks? → use Strategy 4
   b. Is there an existing checkpoint? → resume from it
   c. Need to checkpoint first? → use Strategy 1
4. If session is active and approaching threshold:
   a. Compact if possible → Strategy 3
   b. Checkpoint and rotate if needed → Strategy 1
```

### During Agent Execution

```
1. After every 3-4 tool calls, estimate current token count
2. At 75% of rotation threshold → prepare checkpoint
3. At 90% of rotation threshold → checkpoint NOW, spawn rotation
4. Never let a session run past degradation point
```

### Preventative Measures

- **Scout first** — scouts produce compressed context, reducing downstream token usage
- **Read only what's needed** — don't `cat` 2000-line files when you need 50 lines
- **Batch similar work** — one 40K session is better than four 15K sessions (no context duplication)
- **NPU-local for large reads** — reading files doesn't need reasoning, use free local inference

## Context Health Dashboard

Per-session tracking:

```
═══ Context Health ═══

Session: calm-reef (worker, DeepSeek v4)
Tokens: ~42,000 / 128,000 (32%) | Status: ✅ Healthy
Rotation at: 60,000 | ETA to rotation: ~4 tool calls

Session: keen-cove (worker, Claude Opus)
Tokens: ~78,000 / 200,000 (39%) | Status: ⚠️ Approaching
Rotation at: 90,000 | ETA to rotation: ~2 tool calls
Action: Prepare checkpoint now

Session: bold-fox (reviewer, NPU-local)
Tokens: ~14,000 / 32,000 (43%) | Status: ⚠️ Approaching
Rotation at: 16,000 | ETA to rotation: ~1 tool call
Action: Compact or rotate immediately
```

## Integration

- **babysitter**: Babysitter tracks session runtime; context rotation tracks token depth
- **scope-decomposer**: Decomposer splits tasks to fit within single contexts
- **token-budget**: Rotation checkpoints help track per-session token costs
- **quality-gate**: Never let a session degrade past the point where quality gates become unreliable
