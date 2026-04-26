# Claude Code plugin for 1bit systems — design

A single `/plugin install 1bit systems` should give a Claude Code user:

- `1bit-mcp` wired as an MCP server (17 specialists via 1bit-agents registry).
- `/halo` slash commands — `status / doctor / chat / bench / ppl / say / version / install`.
- A `halo-ops` skill teaching Claude Code about Rule A/B/C/D + our 11-crate layout.
- A `PreCompact` hook auto-saving the current transcript into `~/.claude/projects/-home-bcloud/memory/`.
- A statusline showing live tok/s from `http://127.0.0.1:8180/metrics`.

## File tree

```
plugins/1bit systems/
├── .claude-plugin/
│   └── plugin.json                    # manifest (MCP, skills, hooks, statusline)
├── skills/
│   └── halo-ops/
│       └── SKILL.md                   # Rule A/B/C/D + 11-crate layout + deploy flow
├── commands/
│   ├── halo-status.md
│   ├── halo-doctor.md
│   ├── halo-chat.md
│   ├── halo-bench.md
│   ├── halo-ppl.md
│   ├── halo-say.md
│   ├── halo-version.md
│   └── halo-install.md
├── hooks/
│   └── hooks.json                     # PreCompact → memory-save.sh
├── bin/
│   └── halo-statusline.sh             # one-line tok/s renderer
└── scripts/
    ├── memory-save.sh                 # transcript → auto-memory MD file
    └── validate-halo.sh               # post-install smoke
```

## Manifest (`.claude-plugin/plugin.json`)

```json
{
  "name": "1bit systems",
  "version": "0.1.0",
  "description": "1bit systems ternary BitNet stack — Strix Halo gfx1151, 93 tests, 80+ tok/s",
  "homepage": "https://github.com/bong-water-water-bong/1bit-systems",
  "mcpServers": {
    "1bit-mcp": {
      "command": "${HOME}/.cargo/bin/1bit-mcp",
      "args": [],
      "env": {
        "RUST_LOG": "onebit_mcp=info"
      },
      "transport": "stdio"
    }
  },
  "skills": [
    "skills/halo-ops"
  ],
  "commands": [
    "commands/halo-status.md",
    "commands/halo-doctor.md",
    "commands/halo-chat.md",
    "commands/halo-bench.md",
    "commands/halo-ppl.md",
    "commands/halo-say.md",
    "commands/halo-version.md",
    "commands/halo-install.md"
  ],
  "hooks": "hooks/hooks.json",
  "statusLine": {
    "type": "command",
    "command": "${PLUGIN_DIR}/bin/halo-statusline.sh"
  }
}
```

## Skill (`skills/halo-ops/SKILL.md`)

```markdown
---
name: halo-ops
description: Conventions + deploy flow for the 1bit systems ternary BitNet stack.
triggers:
  - "1bit systems"
  - "1bit-server"
  - "1bit-cli"
  - "strix halo"
  - "rocm-cpp"
  - "bitnet"
---

# halo-ops

Working in 1bit-systems or 1bit systems-core? These rules are non-negotiable:

- **Rule A** — no Python at runtime. Scripts OK, systemd services not.
- **Rule B** — C++20 only for HIP kernels. FFI, never port-to-Rust.
- **Rule C** — hipBLAS banned. Native kernels only.
- **Rule D** — Rust 1.86, edition 2024.

## 11-crate workspace

- `1bit-cli`, `1bit-core`, `lemond`, `1bit-server`, `1bit-agents`,
  `1bit-mcp`, `1bit-landing`, `1bit-lemonade`, `1bit-helm`,
  `1bit-hip`, `1bit-mlx`.

## Deploy flow (1bit-server only)

Binary is held open by the unit — must stop first:

    systemctl --user stop strix-server
    cp target/release/1bit-server ~/.local/bin/1bit-server-real
    systemctl --user start strix-server

Or use `halo install core` which does it for you.

## Cutover gates

See `CUTOVER.md`. Current parity: 96.66% byte-exact, PPL 9.18 vs 9.16.
```

## Slash commands (example — `commands/halo-doctor.md`)

```markdown
---
description: Run halo doctor (GPU + kernel + services + endpoints + network)
argument-hint: (none)
---

Invoke the halo CLI doctor subcommand and surface the output:

!halo doctor
```

(Same pattern for `status`, `chat`, `bench`, `ppl`, `say`, `version`, `install`.)

## Hooks (`hooks/hooks.json`)

```json
{
  "PreCompact": [
    {
      "matcher": "",
      "hooks": [
        {
          "type": "command",
          "command": "${PLUGIN_DIR}/scripts/memory-save.sh",
          "timeout": 15
        }
      ]
    }
  ]
}
```

`memory-save.sh` reads `$CLAUDE_TRANSCRIPT` (path to the JSONL transcript
Claude Code sets in the hook env), extracts user+assistant messages,
writes a compacted markdown file with frontmatter into
`~/.claude/projects/-home-bcloud/memory/auto_precompact_<timestamp>.md`.

## Statusline (`bin/halo-statusline.sh`)

```bash
#!/usr/bin/env bash
# Output one line: [1bit systems] ● 72 tok/s · 340 reqs · p95 1.6s
m=$(curl -fsS --max-time 1 http://127.0.0.1:8180/metrics 2>/dev/null)
if [[ -z "$m" ]]; then
    printf "[1bit systems] ○ offline\n"
    exit 0
fi
tokps=$(echo "$m" | jq -r '.tokps_recent // 0 | floor')
reqs=$(echo "$m"  | jq -r '.requests // 0')
p95=$(echo "$m"   | jq -r '(.p95_ms // 0) / 1000 | floor')
printf "[1bit systems] ● %s tok/s · %s reqs · p95 %ss\n" "$tokps" "$reqs" "$p95"
```

Timeout 1s keeps the statusline responsive if 1bit-server is down.

## Install flow (today)

Manual until Claude Code gets a marketplace:

```bash
git clone git@github-bong:bong-water-water-bong/1bit-systems.git
mkdir -p ~/.claude/plugins
cp -r 1bit-systems/plugins/1bit systems ~/.claude/plugins/
claude plugin reload 1bit systems
```

Then in any Claude Code session:
- `/halo status` → live service snapshot
- The statusline shows live tok/s
- 1bit-mcp tools appear in `tools/list` (17 specialists)
- On PreCompact the transcript archives to memory

## Open questions (need follow-up research)

1. **Marketplace**: does Claude Code ship a plugin marketplace today, or is manual install the only path?
2. **Hook transcript path**: does the `CLAUDE_TRANSCRIPT` env var point at the live transcript or the post-compact version?
3. **MCP tool auto-discovery**: when 1bit-agents specialists become real (not stubs), will Claude Code re-fetch `tools/list` or cache at startup?
4. **Statusline contract**: is the expected output exactly one line? What's the width budget before truncation?
5. **Windows compat**: the statusline shell script needs a PowerShell mirror for Windows Claude Code users — required or skip?
6. **Metrics endpoint auth**: `/metrics` is public on localhost today. If we expose 1bit-server via Caddy, should the statusline probe go through the bearer gate?

## Surprise from the research

Claude Code hooks run **unsandboxed shell commands** with the same trust as the user's shell — which means a PreCompact hook can back up transcripts, a PostToolUse hook can auto-run tests, and a FileChanged hook can rebuild. The plugin system is effectively "package arbitrary ops automation behind a slash command". Much more power than the skill/command surface alone suggests.

## Not in this design (deferred)

- A packaged marketplace listing — wait until Claude Code's marketplace protocol is public.
- The Windows PowerShell mirror — no Windows tester on the 1bit systems team today.
- Auto-register 1bit-mcp as a *system-level* MCP server (not per-plugin) — cleaner but depends on Claude Code config surface.

## Memory links

- `project_strix_ai_rs.md` — gen-2 Rust context
- `project_upstream_watch.md` — related Claude Code skill we already use (caveman)
