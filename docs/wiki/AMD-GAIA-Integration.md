---
phase: analysis
owner: magistrate
---

# AMD GAIA integration (and naming-conflict note)

## What AMD GAIA is

**AMD GAIA** (https://amd-gaia.ai) is AMD's official "desktop interface for running AI agents 100% locally on your AMD hardware." Electron-based, MIT-licensed, Windows `.exe` + Ubuntu `.deb` installers. Uses Lemonade Server on port 8000 for model dispatch across NPU + iGPU on Strix Halo. Tested model: Qwen3.5-35B-A3B-GGUF.

Key features:
- Local-only — no cloud round-trip
- Multi-format document analysis (PDF, Word, Excel, code)
- Session persistence CLI ↔ UI
- Built-in MCP server integration
- File-system search + browsing
- Electron UI on localhost:4200

## Naming conflict: AMD GAIA vs our `halo-gaia` (resolved)

> **Resolved 2026-04-20.** Our crate was renamed `halo-gaia` → `1bit-helm`.
> This section is preserved as the historical record and rename rationale.
> Every other reference in the tree now says `1bit-helm`; `packages.toml`
> keeps `[component.gaia]` as an alias for back-compat with old shell
> history. See `Crate-1bit-helm.md` for the live spec.


Our scaffold crate `halo-gaia` (phase: analysis, see `Crate-halo-gaia.md`) is a native-Rust desktop client for 1bit systems — the SAME category as AMD GAIA. Both:
- Target Ryzen AI MAX+ 395 / Strix Halo
- Local-only inference
- Desktop-first UI
- MIT licensed

**This is a direct naming overlap.** We're a small project; AMD is AMD. Two options:

1. **Rename our crate.** Candidate names: `1bit-helm` (captain's wheel — navigation + control metaphor), `halo-aura`, `halo-pulse`, `halo-cockpit`, `halo-pane`, `halo-orbit`. Rename is cheap because `halo-gaia` is still scaffold-only — no external callers.

2. **Keep the name, own the overlap.** Argue that `halo-gaia` is narrowly the 1bit systems desktop pane (Greek primordial of Earth, 1bit-family framing), not a competing AI framework. AMD GAIA is an AI-agent framework; we're a pane. Different scopes despite the word.

**Recommendation: rename.** Conflict risk > branding cost. Pick `1bit-helm` or equivalent during the next halo-gaia promotion-from-analysis review.

## Integration path: AMD GAIA as external client over 1bit-proxy

AMD GAIA drives inference through a Lemonade/OpenAI-compatible endpoint. In the
repair path, point it at `http://127.0.0.1:13306/api/v1` so GAIA can keep one
local base URL while the backend is toolbox `llama-server`, native Lemonade, or
optional FastFlowLM.

Same pattern as Hermes Agent (see `Hermes-Integration.md`): Electron + Node lives on the user's laptop (external-client surface, Rule A unaffected) while our native-HIP kernels stay on strixhalo.

Setup snippet for a user who wants AMD GAIA → 1bit systems:

```bash
# On the user's Linux box
sudo dpkg -i gaia-agent-ui.deb        # AMD GAIA install
# Edit gaia's Lemonade endpoint config:
#   (exact config path TBD — check their docs/configuration page)
# Point it at:
#   http://100.64.0.1:13306/api/v1     # 1bit-proxy over the mesh
```

Windows users run `gaia-agent-ui.exe`; same config.

Bearer: AMD GAIA may not send an `Authorization` header by default. Caddy's `/lemon/*` route (or a new `/gaia/*` alias if we add one) needs to handle pre-auth appropriately — either require the user to set a bearer in GAIA's request headers config OR whitelist mesh IPs without bearer for this specific route. TBD when we actually wire a user.

## What AMD GAIA won't replace

- **1bit-cli** — our power-user terminal stays. GAIA is the glanceable surface; CLI is the dispatcher.
- **/studio/voice/ mobile PWA** — browser-accessible from phones. GAIA is desktop-only.
- **1bit-landing dashboard** — LAN-only service status. GAIA is user-focused, not ops.

## Cross-refs

- `Crate-1bit-helm.md` — our desktop client spec (formerly Crate-halo-gaia.md)
- `Development.md` — the repair path and review rules AMD GAIA sits on top of
- `Hermes-Integration.md` — parallel external-client pattern for Hermes Agent
- `VPN-Only-API.md` — mesh + bearer posture that AMD GAIA sits behind

## Phase: analysis

Promote to `solutioning` once:
- Decision on rename (halo-gaia → 1bit-helm or kept)
- Caddy route shape for AMD GAIA decided (reuse `/lemon/*` or add `/gaia/*`)
- A real AMD GAIA install on a mesh peer verifies the integration path end-to-end
