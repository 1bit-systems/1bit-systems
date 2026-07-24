# 1bit = pi Clone: Agent Harness Rebrand

## Problem

Pi.dev is a powerful coding agent harness with extensions, skills, themes,
packages, MCP, and multi-model orchestration. 1bit has a world-class NPU
inference engine (16ms/tok INT8) and API server, but no agent system of its
own. Users currently launch a 1bit-managed NPU stack then interact through
pi or manual terminal commands.

The vision: **1bit IS the agent — not a service managed by one.**

## Solution

Fork/clone pi.dev's agent architecture under the 1bit brand. Bundle every
piece of pi's functionality (agent core, TUI, AI router, package manager,
extensions, skills, themes, MCP) into a self-contained 1bit distribution
that:

1. Ships with the NPU as the **default inference provider**
2. Has its own `1bit` CLI for packages (`1bit install`), updates (`1bit update`),
   config (`1bit config`), and agent sessions
3. Boots automatically on login via systemd user service
4. Provides a complete out-of-the-box agent experience — `/1bit` in terminal
   → ask a question → NPU answers locally

## Approach

Rather than rewriting pi from scratch (which would be years of work), we
**vend pi's npm packages under the 1bit namespace** and build a shim CLI +
service layer that rebrands everything — while keeping the pi source reference
so we can track upstream updates.

Components:
- **`@1bit/agent-core`** — symlink/vendored pi-agent-core
- **`@1bit/ai`** — symlink/vendored pi-ai, with NPU provider wired as default
- **`@1bit/tui`** — symlink/vendored pi-tui, branded "1bit"
- **`@1bit/cli`** — new CLI shim: `1bit [command]`
- **`@1bit/settings`** — settings files with NPU as default model
- **systemd user service** — auto-start 1bit agent on login
- **branding** — all terminal UI, help text, prompts say "1bit"

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Fork strategy | Vendored npm packages + shim CLI | Minimal maintenance burden; upstream pi drives actual agent work |
| Default model | NPU (localhost:9090) via OpenAI-compatible endpoint | Zero config out of box |
| Package manager | Reuse pi's package system with custom settings path | Don't reinvent; just rebrand CLI |
| Agent state dir | `~/.1bit/` | Parallel to `~/.pi/` |
| Boot service | systemd user service (`1bit-agent.service`) | Standard Linux, auto-starts on login |

## File Changes

### New files in `/home/bcloud/1bit-systems/`:

| File | Purpose |
|------|---------|
| `package.json` | Root package — `@1bit/cli` entry via bin scripts |
| `src/cli.ts` | CLI parser — `1bit up`, `1bit chat`, `1bit install`, etc. |
| `src/agent/session.ts` | Agent session manager — wraps pi-agent-core |
| `src/agent/provider.ts` | AI provider config — NPU as default + provider selection |
| `src/agent/pi-shim.ts` | Wraps pi-agent-core for session management |
| `src/commands/chat.ts` | Interactive chat mode |
| `src/commands/packages.ts` | Package management (`1bit install`, `1bit remove`, `1bit list`) |
| `src/commands/update.ts` | Update 1bit and packages |
| `src/commands/config.ts` | Configuration management |
| `src/commands/up.ts` | Start NPU stack (reuse existing `1bit up`) |
| `src/commands/down.ts` | Stop NPU stack |
| `src/branding/tui.ts` | TUI branding module — colors, ASCII art, theme |
| `src/branding/config.ts` | Default settings with NPU provider |
| `extensions/1bit-npu/index.ts` | Extension: NPU status, model info, engine control |
| `skills/1bit-npu/SKILL.md` | Skill: manage 1bit NPU stack from within the agent |
| `prompts/1bit.md` | System prompt for 1bit agent personality |
| `themes/1bit.json` | 1bit-themed color scheme |
| `services/1bit-agent.service` | systemd user unit for auto-start |
| `services/install-service.sh` | Script to install/remove systemd unit |
| `bin/1bit` | CLI entry point (symlinked to dist) |
| `tsconfig.json` | TypeScript config |
| `.pi/settings.json` | Project-level pi settings for 1bit development |

### Modified files:

| File | Change |
|------|--------|
| `CLAUDE.md` | Add agent workflow for 1bit CLI development |
| `docs/architecture.md` | Add agent architecture section |
| `README.md` | Add agent features to README |
| `packaging/deb/` | Add 1bit CLI + service to packaging |
| `packaging/deb/usr/bin/1bit` | Wrapper script for deb package |

## Out of Scope

- Custom LLM inference engine (uses existing NPU stack + pi-ai routing)
- Custom extension SDK (reuses pi's extension system)
- Custom skill format (reuses pi's skill system)
- Migration of existing pi sessions to 1bit
