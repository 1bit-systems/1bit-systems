# 1bit = pi Clone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clone pi.dev's agent harness under the 1bit brand — self-contained CLI, TUI, package manager, extension/skill system, with NPU as default provider.

**Architecture:** Vendored pi npm packages under `@1bit/` namespace with a shim CLI (`1bit`) that rebrands every aspect (CLI name, TUI, help text, settings paths, model defaults). A systemd user service auto-boots the agent on login.

**Tech Stack:** TypeScript, Node.js, pi packages (`@earendil-works/pi-agent-core`, `@earendil-works/pi-ai`, `@earendil-works/pi-tui`), systemd, Fastify (existing 1bit API bridge)

## Global Constraints

- All user-facing text must say "1bit" not "pi" — CLI help, TUI prompts, error messages, wallpapers
- Agent state directory: `~/.1bit/` (not `~/.pi/`)
- Settings file: `~/.1bit/agent/settings.json`
- All source in `/home/bcloud/1bit-systems/`
- NPU must be the default model provider (wired to the existing `localhost:9090` endpoint)
- Must preserve the existing `1bit up|down|status|build` commands
- Must work with installed pi packages system (reuse pi's npm install logic)
- Must not modify `/home/bcloud/.pi/` in any way — 1bit is independent

## File Structure

```
1bit-systems/
├── package.json              # Root — pi package for extension+skill+prompts
├── tsconfig.json             # TypeScript config
├── src/
│   ├── cli.ts                # CLI entry — parses commands, dispatches
│   ├── branding/
│   │   └── config.ts         # Default settings with NPU provider
│   └── commands/
│       ├── chat.ts           # `1bit chat` — interactive agent session
│       ├── packages.ts       # `1bit install|remove|list`
│       ├── update.ts         # `1bit update`
│       ├── config.ts         # `1bit config`
│       ├── up.ts             # `1bit up` — delegate to existing script
│       ├── down.ts           # `1bit down` — delegate to existing script
│       └── status.ts         # `1bit status`
├── extensions/
│   └── 1bit-npu.js          # Extension: NPU tools
├── skills/
│   ├── 1bit-npu/
│   │   └── SKILL.md         # Skill: manage NPU stack from chat
│   └── 1bit-agent/
│       └── SKILL.md         # Skill: how 1bit agent works
├── prompts/
│   └── 1bit.md              # System prompt: 1bit agent personality
├── themes/
│   └── 1bit.json            # TUI color theme
├── services/
│   ├── 1bit-agent.service   # systemd user unit
│   └── install-service.sh   # Install/uninstall script
├── bin/
│   └── 1bit                 # CLI entry shim
├── .pi/
│   └── settings.json        # pi settings for 1bit development
└── dist/                    # Built output (gitignored)
```

## Task Overview

| # | Task | Key Files | Depends |
|---|------|-----------|---------|
| 1 | Scaffold: package.json, tsconfig, CLI bin, shim | `package.json`, `tsconfig.json`, `bin/1bit`, `src/cli.ts` | — |
| 2 | Branding: default settings, TUI theme, system prompt | `src/branding/config.ts`, `themes/1bit.json`, `prompts/1bit.md` | 1 |
| 3 | Commands: up, down, status | `src/commands/up.ts`, `down.ts`, `status.ts` | 1 |
| 4 | Chat: interactive agent session | `src/commands/chat.ts` | 2, 3 |
| 5 | Packages: install, remove, list | `src/commands/packages.ts` | 1 |
| 6 | Update, Config | `src/commands/update.ts`, `config.ts` | 1 |
| 7 | Extensions + Skills | `extensions/1bit-npu.js`, `skills/*/SKILL.md` | 3 |
| 8 | Systemd service + CLI install | `services/1bit-agent.service`, `install-service.sh` | 4 |
| 9 | .pi settings + pi manifest | `.pi/settings.json`, update `package.json` | 1 |
| 10 | Packaging: deb + install script | Update `packaging/deb/`, `packaging/install.sh` | 8 |

## Task Details

### Task 1: Scaffold

Create `package.json`, `tsconfig.json`, `src/cli.ts`, `bin/1bit`.

`package.json`:
```json
{
  "name": "1bit-systems",
  "version": "2026.07.02",
  "description": "1bit — your NPU-native coding agent. Fork of pi.dev, wired to local NPU.",
  "type": "module",
  "bin": { "1bit": "dist/cli.js" },
  "main": "./dist/index.js",
  "files": ["dist", "extensions", "skills", "prompts", "themes", "docs"],
  "scripts": { "build": "tsc", "dev": "tsx src/cli.ts", "watch": "tsc --watch", "start": "node dist/cli.js" },
  "keywords": ["1bit", "coding-agent", "ai", "npu", "pi-package"],
  "dependencies": {
    "@earendil-works/pi-agent-core": "^0.80.3",
    "@earendil-works/pi-ai": "^0.80.3",
    "@earendil-works/pi-tui": "^0.80.3",
    "yaml": "^2.9.0", "minimatch": "^10.2.5", "chalk": "^5.6.2", "semver": "^7.8.0"
  },
  "devDependencies": {
    "typescript": "^5.9.3", "@types/node": "^24.12.4", "tsx": "^4.19.0"
  },
  "engines": { "node": ">=22.19.0" },
  "license": "MIT"
}
```

`tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2024", "module": "ESNext", "moduleResolution": "bundler",
    "strict": true, "esModuleInterop": true, "skipLibCheck": true,
    "outDir": "dist", "rootDir": "src", "declaration": true, "declarationMap": true, "sourceMap": true
  },
  "include": ["src/**/*"], "exclude": ["node_modules", "dist"]
}
```

`src/cli.ts` — CLI entry that prints 1bit-branded help, routes commands:
```
Commands: chat, up, down, status, build, install, remove, list, update, config, help, version
```
Default command: `chat`. Uses dynamic imports for each command module.

`bin/1bit` — Node.js shim that uses dist/cli.js if built, else tsx src/cli.ts in dev.

Steps: create files, `npm install`, verify `npx tsx src/cli.ts help` prints 1bit help.

### Task 2: Branding

Create directories: `src/branding`, `themes`, `prompts`.

`src/branding/config.ts`:
- Exports `ONE_BIT_DIR = ~/.1bit`, `AGENT_DIR`, `SETTINGS_PATH`
- Settings interface: theme, defaultProvider, defaultModel, defaultThinkingLevel, packages[], apiKeys, npuEndpoint
- `defaultSettings()` returns NPU provider, npuEndpoint = "http://127.0.0.1:9090/v1"
- `loadSettings()` reads/merges with defaults, `saveSettings()` writes

`themes/1bit.json`:
- Colors: background #0a0a0f, primary #00ff87, secondary #00d4ff, accent #f00fd2, etc.

`prompts/1bit.md`:
- System prompt: "You are 1bit, an AI coding agent powered by the local NPU on this machine"
- Describes identity, capabilities, NPU stack, default behavior (prefer local NPU)

Steps: create dirs, create files, init ~/.1bit/agent/settings.json.

### Task 3: NPU Stack Commands

Create `src/commands/up.ts`, `down.ts`, `status.ts`.

Each delegates to `/home/bcloud/.pi/agent/bin/1bit <command>` via execSync.

Exports: `startUp()`, `startBuild()`, `shutDown()`, `showStatus()`.

Build and verify `npx tsx src/cli.ts status` works.

### Task 4: Chat

Create `src/commands/chat.ts`.

- `printBanner()` — ASCII art "1bit" with version, 50 TOPS INT8, 63 tok/s
- `checkNpuStack()` — HTTP GET to localhost:9090, returns boolean
- `queryNpu(prompt)` — POST to localhost:9090/v1/chat/completions, returns response text
- `handleCommand(cmd, rl)` — /help, /status, /up, /down, /clear, /exit
- `startChat()` — prints banner, checks NPU, starts readline with "1bit> " prompt
- Non-slash input goes to NPU if running, else tells user to start stack

Test with `timeout 3 npx tsx src/cli.ts chat <<< "/help"`.

### Task 5: Packages

Create `src/commands/packages.ts`.

- `installPackage(spec)` — validates spec, adds to settings.packages[], saves
- `removePackage(name)` — filters out matching packages from settings
- `listPackages()` — prints installed packages or "No packages installed."

### Task 6: Update and Config

`src/commands/update.ts`: `runUpdate(spec?)` — git pull + npm install + npm run build if no spec.

`src/commands/config.ts`: `manageConfig(args[])` — show all or get/set individual keys.

### Task 7: Extensions and Skills

`extensions/1bit-npu.js` — pi-compatible extension with tools:
- npu_status, npu_up, npu_down, npu_query

`skills/1bit-npu/SKILL.md` — describes NPU stack management

`skills/1bit-agent/SKILL.md` — describes 1bit agent architecture

### Task 8: Systemd Service

`services/1bit-agent.service`:
```ini
[Unit]
Description=1bit NPU-native coding agent
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/1bit chat
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

`services/install-service.sh` — install/remove/status commands, creates user systemd unit.

### Task 9: .pi Settings

`.pi/settings.json` — points pi at 1bit's extensions/skills/prompts/themes.

Update `package.json` with pi manifest:
```json
"keywords": ["pi-package", "1bit"],
"pi": { "extensions": ["./extensions"], "skills": ["./skills"], "prompts": ["./prompts"], "themes": ["./themes"] }
```

### Task 10: Packaging

`packaging/deb/usr/bin/1bit` — wrapper script
`packaging/deb/etc/systemd/user/1bit-agent.service` — systemd unit for deb
Update `packaging/deb/DEBIAN/control` — add nodejs, systemd deps
Update `packaging/install.sh` — install CLI, service, and enable
