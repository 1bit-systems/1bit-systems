# Implementation Tasks: 1bit = pi Clone

> Proposal: `openspec/changes/1bit-pi-clone/proposal.md`
> Plan: `docs/superpowers/plans/2026-07-02-1bit-pi-clone.md`

## Status

- [x] **Task 1**: Scaffold — package.json, tsconfig, CLI bin, shim
- [x] **Task 2**: Branding — default settings, TUI theme, system prompt
- [x] **Task 3**: Commands — up, down, status (full implementation)
- [x] **Task 4**: Commands — chat, interactive agent session (full implementation)
- [x] **Task 5**: Commands — packages (install, remove, list — full implementation)
- [x] **Task 6**: Commands — update, config (full implementation)
- [x] **Task 7**: Extensions + Skills — NPU integration
- [x] **Task 8**: Systemd service + CLI install
- [x] **Task 9**: .pi settings for 1bit dev + pi manifest
- [x] **Task 10**: Packaging — install script includes 1bit CLI + service

## Verification Gates

| Gate | Cmd | Expected | Status |
|------|-----|----------|--------|
| CLI routes | `1bit help` | Prints 1bit brand help | ✅ |
| NPU status | `1bit status` | Shows NPU stack health | ✅ |
| Config | `1bit config` | Shows NPU as default provider | ✅ |
| Packages | `1bit list` | Shows installed packages | ✅ |
| Config set | `1bit config theme=dark` | Saves setting | ✅ |
| Service | `systemctl --user status 1bit-agent` | Active/running | 🔧 (requires install) |

## Files Created

| File | Purpose |
|------|---------|
| `src/commands/chat.ts` | Full chat: banner, NPU query, /commands, readline |
| `src/commands/up.ts` | Start NPU stack + build engine |
| `src/commands/down.ts` | Kill NPU stack processes |
| `src/commands/status.ts` | Port/API health check with fetch |
| `src/commands/packages.ts` | npm: & git: package install/remove/list |
| `src/commands/update.ts` | git pull + npm install + build |
| `src/commands/config.ts` | Get/set config keys |
| `src/branding/config.ts` | ~/.1bit/ settings with NPU defaults |
| `extensions/1bit-npu/index.js` | 4 tools: npu_status, npu_up, npu_down, npu_query |
| `skills/1bit-npu/SKILL.md` | NPU stack management skill |
| `skills/1bit-agent/SKILL.md` | Agent architecture skill |
| `services/1bit-agent.service` | systemd --user unit |
| `services/install-service.sh` | install/uninstall/status commands |
| `.pi/settings.json` | NPU provider config for pi |
| `themes/1bit.json` | 1bit color theme (dark, green/cyan/pink) |
| `prompts/1bit.md` | System prompt (NPU-native agent) |
