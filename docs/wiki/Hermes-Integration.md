# Hermes Agent integration

[NousResearch/hermes-agent](https://github.com/nousresearch/hermes-agent) is a self-improving agent with a skill-creation loop, FTS5 session search, Honcho dialectic user modeling, messaging-gateway (Telegram/Discord/Slack/WhatsApp/Signal), cron scheduler, MCP integration, and subagent spawn. MIT, Python.

Rule A (no Python in runtime) forbids Hermes **on strixhalo as a service**. But nothing stops a user from running Hermes *on their laptop* talking to strixhalo's OpenAI-compat endpoint. That is the sanctioned path.

## External client — 5 minute setup

On the user's machine (Linux / macOS / WSL2 / Android Termux):

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
source ~/.bashrc
hermes setup         # wizard
hermes model         # pick "custom endpoint"
```

When prompted for the endpoint, use the 1bit-server URL from the mesh:

```
base_url:    http://100.64.0.1:8180/v1          # Headscale mesh, strixhalo
model:       halo-1bit-2b
api_key:     sk-halo-local                       # ignored by 1bit-server
```

That's it. `hermes` TUI now drives 1bit-server. Every request lands on our native HIP kernels, no detour through NousPortal/OpenRouter.

## Expose 1bit-mcp to Hermes

1bit-mcp is the Rust stdio JSON-RPC bridge that exposes every 1bit-agents specialist as an MCP tool. Point Hermes at it:

```bash
hermes config set mcp.servers.halo.command /path/to/1bit-mcp
hermes config set mcp.servers.halo.args '[]'
```

Then inside a Hermes conversation:

```
/mcp list                   # see 1bit-agents specialists as tools
```

Hermes' 40+ built-in tools + our 17 1bit-agents specialists → 57-tool agent on commodity Python, talking to native-HIP inference. Rule A untouched: Python lives on the laptop, kernels stay on strixhalo.

## How Hermes' "self-improving skills" actually work

Reading the docs closely: **the self-improvement is LLM-driven, not RL.** There is no training loop, no reward signal, no weights update. The agent calls a `skill_manage` tool with four actions — `create`, `patch` (preferred for targeted fixes), `edit` (full rewrite), `delete` — against markdown files at `~/.hermes/skills/<category>/<name>/SKILL.md`. Autonomous-creation triggers are heuristic: ≥5 successful tool calls, error-recovery path found, user correction, non-trivial workflow. The LLM reads its own skill file, edits it via `patch`, saves. That's the whole loop.

Good news for us: **Rule A-safe to replicate.** No Python dependency, no training infra. Just a Rust trait + file-edit tool + prompt.

## Skill format (adopt verbatim for agentskills.io compat)

```
~/.halo/skills/<category>/<name>/
├── SKILL.md               # required — markdown body + YAML frontmatter
├── references/            # optional supporting docs
├── templates/             # optional output formats
├── scripts/               # optional helper scripts
└── assets/                # optional binary assets
```

Frontmatter (YAML):

```yaml
---
name: my-skill
description: Brief description
version: 1.0.0
platforms: [linux]
metadata:
  halo:
    tags: [bitnet, hip]
    category: kernels
    fallback_for_toolsets: [web]
    requires_toolsets: [terminal]
---
```

Same keys as Hermes' `metadata.hermes.*` → `metadata.halo.*`. Preserves interop: a skill authored on either platform is readable by both. `agentskills.io` becomes a shared ecosystem.

## Memory format (match Hermes, replace `hermes` → `halo`)

Hermes stores memory in plain markdown files with a hard char cap:

- `~/.hermes/memories/MEMORY.md` — 2,200 char cap, `§` delimiter between entries
- `~/.hermes/memories/USER.md` — 1,375 char cap
- `~/.hermes/state.db` — SQLite + FTS5 for session history, immutable mid-session snapshot injected at start

We already do MEMORY.md at `~/.claude/projects/-home-bcloud/memory/`. Match the format in 1bit-agents' new memory layer:

- `~/.halo/memories/MEMORY.md` + `USER.md` (same caps, same `§` delimiter)
- `~/.halo/state.db` via `rusqlite` with FTS5 module

## Features worth porting into 1bit-agents

Revised rank after reading the architecture (`AIAgent` 10,700 LOC, `HermesCLI` 10,000 LOC, `GatewayRunner` 9,000 LOC — three god-classes):

| Feature | Mechanism | Effort | Notes |
|---|---|---|---|
| SKILL.md format + `~/.halo/skills/` layout | File + YAML frontmatter, same keys as Hermes | **1 day** | Interop win. Free. Do this first. |
| `skill_manage` tool (create/patch/edit/delete) | Text-edit tool exposed to 1bit-agents specialists | **3 days** | The entire "self-improvement loop" reduces to this one tool. |
| FTS5 session search | `rusqlite` `fts5` feature on `~/.halo/state.db` | **3 days** | Replaces 1bit-agents linear-scan recall. |
| MEMORY.md / USER.md file layer | Same 2200/1375 char cap + `§` delimiter | **1 day** | Matches format we use; formalize it. |
| Autonomous skill-creation trigger | Heuristic: successful run length ≥5 tools, recovery path, user correction | **1 week** | After skill_manage lands. |
| Honcho dialectic user model | Reimplement in Rust; Honcho itself is just one of 8 memory plugins in Hermes — don't treat it as special | **2 weeks** | Lower priority after rereading docs; Hermes pluggable design means it's not core. |
| Messaging gateway (Telegram/Discord/Slack/WhatsApp/Signal) | Hermes has 18 platform adapters in `gateway/` | **Do not port** | Wrap Hermes instead. Python on a user VPS, strixhalo stays clean. |
| Three API modes (chat_completions / codex_responses / anthropic_messages) | Hermes' `runtime_provider.py` | **Already done** | 1bit-server is OpenAI-compat; we're the endpoint, not the caller. |
| 47-tool registry with `check_fn` gating | Import-time `registry.register()` | **Already better** | Our 17 typed specialists beat a 47-tool `**kwargs` bag. Keep our shape. |

Skip: OpenClaw migration, Termux, serverless (Modal/Daytona/Singularity), 3 API-mode adapter layer.

## Integration order (concrete)

1. **Today**: wire Hermes → 1bit-server + expose 1bit-mcp (5-min operator setup, above).
2. **Week 1**: adopt SKILL.md + `~/.halo/skills/` layout. Write halo-skills format doc. 1 day.
3. **Week 1**: MEMORY.md / USER.md + `~/.halo/state.db` FTS5. 4 days.
4. **Week 2**: `skill_manage` Rust tool exposed to 1bit-agents. 3 days. This is the entire "self-improvement loop" — it's that simple.
5. **Week 3**: autonomous skill-creation trigger, heuristic. 1 week.
6. **Month 2+**: Honcho reimpl if still warranted, after we see real usage of the skill loop.

All of it behind Sherry + BitNet v2 on inference critical path, but the agent-UX lane can progress in parallel (different crates, different files — 1bit-agents vs rocm-cpp).

## Honest comparison

What Hermes has that 1bit-agents doesn't:
- Skill self-improvement loop (genuinely novel)
- Multi-platform messaging bridge
- 104k users, battle-tested

What 1bit-agents has that Hermes doesn't:
- Native HIP inference (Hermes is model-agnostic — the model is whatever endpoint you point it at)
- Rule A compliance (no Python in the hot path)
- Shadow-burnin parity harness
- Live on Strix Halo as a tiny-box mini-PC

Complementary, not competitive. Hermes is the cockpit; 1bit-systems is the engine.

## References

- [Hermes README](https://github.com/NousResearch/hermes-agent)
- [Hermes docs](https://hermes-agent.nousresearch.com/docs/)
- [Honcho upstream](https://github.com/plastic-labs/honcho)
- [agentskills.io](https://agentskills.io) — open standard for skill sharing
- Our 1bit-mcp crate: `cpp/mcp/`
