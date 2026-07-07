---
name: token-budget
description: Tracks token consumption and cost across all agents and providers — DeepSeek, Claude, GPT, NPU-local. Prevents surprise bills, enforces budgets, and provides per-task cost attribution.
---

# Token Budget Tracker

Know what every agent costs before you get the bill. Tracks tokens consumed
per session, per agent, per provider — with budget enforcement and cost attribution.

## Providers Tracked

| Provider | Cost Model | Tracking |
|----------|-----------|----------|
| **DeepSeek** (v4-flash) | ~$0.50/M input, ~$2.00/M output | Via `/home/bcloud/.pi/agent/auth.json` key |
| **Claude** (Opus) | Via Anthropic billing | Via `~/.claude/.credentials.json` |
| **Codex** (GPT-5.5) | Via OpenAI billing | Via `~/.codex/config.toml` |
| **NPU-local** | Free (local inference) | Counted but zero cost |

## Budget Configuration

Set budgets in `~/.pi/agent/settings.json`:

```json
{
  "token_budget": {
    "daily_limit_usd": 25.00,
    "per_task_limit_usd": 5.00,
    "warning_threshold_pct": 75,
    "providers": {
      "deepseek": { "daily_limit_usd": 15.00 },
      "anthropic": { "daily_limit_usd": 8.00 },
      "openai": { "daily_limit_usd": 2.00 }
    },
    "auto_fallback": true,
    "fallback_chain": ["npu-local", "deepseek", "anthropic"]
  }
}
```

**Auto-fallback:** When a provider hits budget, automatically downgrade to the next
in the fallback chain. NPU-local is always free and last resort.

## Tracking Protocol

### Before Every Agent Call

```
1. Estimate token cost for task:
   - scouts (read-only): ~2K input, ~500 output → ~$0.002
   - workers (implementation): ~5K input, ~2K output → ~$0.007
   - reviewers (analysis): ~3K input, ~1K output → ~$0.004
   - brainstormers: ~2K input, ~1K output → ~$0.003

2. Check against per-task limit:
   └─ Over limit? → split task or use cheaper provider

3. Check against daily remaining:
   └─ Over limit? → use fallback provider or defer
```

### After Every Agent Call

```
1. Log: session_id, provider, model, tokens_in, tokens_out, cost_est, task_label
2. Update running daily total
3. If >75% daily limit → warn user
4. If >100% daily limit → block further paid calls, force NPU-local
```

## Cost Estimation Table

Quick reference for estimating task costs:

| Agent Type | Est. Input Tokens | Est. Output Tokens | DeepSeek Cost | Claude Cost |
|-----------|-------------------|-------------------|---------------|-------------|
| scout | 2,000 | 500 | ~$0.002 | ~$0.03 |
| planner | 3,000 | 1,500 | ~$0.005 | ~$0.06 |
| worker | 5,000 | 2,000 | ~$0.007 | ~$0.09 |
| reviewer | 3,000 | 1,000 | ~$0.004 | ~$0.05 |
| brainstormer | 2,000 | 1,500 | ~$0.004 | ~$0.05 |
| synthesis | 6,000 | 2,000 | ~$0.007 | ~$0.10 |
| NPU-local | any | any | **$0.00** | **$0.00** |

## Budget-Aware Agent Selection

When choosing an agent provider for a task:

```
1. Is the task simple and NPU-local can handle it?
   → Use NPU-local (free, fast, always available)

2. Is the task complex but cheap providers work?
   → DeepSeek (cheapest cloud, good for most tasks)

3. Is the task code-heavy requiring Claude-level reasoning?
   → Check budget, use Claude if available

4. Is the task image/design generation?
   → Codex GPT-5.5 (only option)

Decision matrix:
```
| Task Complexity | NPU-local (qwen 0.6B) | DeepSeek v4 | Claude Opus | Codex 5.5 |
|----------------|----------------------|-------------|-------------|-----------|
| Simple file ops | ✅ Best | Overkill | Overkill | Overkill |
| Code generation | ⚠️ Adequate | ✅ Best value | ✅ Best quality | ✅ Good |
| Complex refactor | ❌ Insufficient | ⚠️ Maybe | ✅ Best | ✅ Good |
| Architecture | ❌ Insufficient | ✅ Good | ✅ Best | ✅ Good |
| Security review | ❌ Insufficient | ✅ Good | ✅ Best | ✅ Good |
| Image gen | ❌ N/A | ❌ N/A | ❌ N/A | ✅ Only |

## Budget Journal

Maintain at `~/.pi/agent/cache/token-budget.jsonl`:

```json
{"date":"2026-07-07","provider":"deepseek","model":"deepseek-v4-flash","session":"calm-reef","tokens_in":4231,"tokens_out":1847,"cost_est":0.0058,"task":"fix TS errors","agent_type":"worker"}
{"date":"2026-07-07","provider":"npu-local","model":"qwen3-0.6b-FLM","session":"npu-1","tokens_in":1200,"tokens_out":300,"cost_est":0.0,"task":"read file structure","agent_type":"scout"}
```

## Daily Summary

End of session, produce:

```
═══ Token Budget: 2026-07-07 ═══
Daily limit: $25.00 | Spent: $3.42 (13.7%)

DeepSeek:    $2.15 / $15.00 (14.3%)
Claude:      $1.27 / $8.00  (15.9%)
Codex:       $0.00 / $2.00  (0.0%)
NPU-local:   $0.00 (free, 23 calls)

Sessions: 14 | Avg cost: $0.24/session
Most expensive: "fused runtime integration" ($0.89)
```

## Cost-Saving Strategies

1. **Scout first, always.** A $0.002 scout saves a $0.09 worker from reading the wrong files.
2. **NPU-local for simple tasks.** File reads, grep, small edits — local can handle it.
3. **Batch parallel reads.** One agent can read 5 files; don't spawn 5 agents.
4. **Synthesize locally.** After fan-out, use NPU-local for synthesis (it's just merging text).
5. **Compact context before expensive calls.** Use `/compact` equivalent to reduce input tokens.

## Integration

- **scope-decomposer**: Decomposer estimates costs before spawning
- **babysitter**: Babysitter logs retry costs (failed runs still cost tokens)
- **supervisor-orchestration**: Orchestrator picks providers based on budget
