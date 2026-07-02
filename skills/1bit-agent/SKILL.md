---
name: 1bit-agent
description: How the 1bit coding agent works — CLI, chat, package management, extensions, skills, themes, and NPU integration.
---

# 1bit Agent Architecture

## Overview

1bit is a fork of [pi.dev](https://pi.dev) — a minimal terminal coding agent harness. It shares the same architecture but is rebranded and defaults to the local NPU inference stack rather than cloud providers.

## Entry Points

- **CLI**: `1bit <command>` — routes to `dist/cli.js` (or `tsx src/cli.ts` in dev)
- **Chat mode**: `1bit chat` — interactive readline session with NPU backend
- **Default command**: `1bit` without arguments starts chat

## Commands

| Command | Description |
|---------|-------------|
| `chat` | Interactive agent session (default) |
| `up` | Start NPU stack (API bridge + Lemond) |
| `down` | Stop NPU stack |
| `status` | Show NPU stack health |
| `build` | Build NPU engine from source |
| `install <pkg>` | Install a package (npm:, git:, or path) |
| `remove <name>` | Remove a package |
| `list` | List installed packages |
| `update [pkg]` | Update 1bit and packages |
| `config [key=val]` | View or set configuration |

## Configuration

Settings stored at `~/.1bit/agent/settings.json`:

- `theme`: UI theme name (default: `"1bit"`)
- `defaultProvider`: AI provider (default: `"npu"`)
- `defaultModel`: Model name (default: `"npu-local"`)
- `npuEndpoint`: NPU API URL (default: `"http://127.0.0.1:9090/v1"`)
- `packages`: Array of installed packages

## Extensions & Skills

1bit reuses pi's extension and skill system. Extensions add tools (TypeScript/JS modules), skills add markdown-based instruction files:

- Extensions: `extensions/*/index.js`
- Skills: `skills/*/SKILL.md`
- Prompts: `prompts/*.md`
- Themes: `themes/*.json`

## Sessions

1bit stores session data in `~/.1bit/agent/`. Session files track conversation history.

## NPU Integration

The NPU is the default inference backend. The stack runs two servers:
- **API bridge** (port 9090) — OpenAI-compatible chat API + terminal
- **Lemond** (port 13305) — Web chat UI

Both must be running for the agent to respond. Use `1bit up` to start them.
