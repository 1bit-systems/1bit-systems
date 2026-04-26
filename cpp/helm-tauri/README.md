# 1bit-helm (Tauri)

The dynamic-tiling desktop shell for the 1bit-systems stack. Tauri 2 +
React 19 + `react-mosaic-component`. Each tile is one product surface
(halo / MCP / bench / runbook). Drag the dividers to resize, click the
maximize button (or double-click the title bar) to expand a tile to
full window.

This is a sibling of `cpp/helm/` (Qt6) and `cpp/helm-tui/` (FTXUI). The
three coexist as alternate front-ends over the same lemonade :8180 +
halo-agent backend. None of them are the canonical desktop yet — Qt6
is the default, this Tauri shell is the mosaic experiment.

## Build

```bash
# Opt-in target — not in the default release-strix build.
cmake --preset release-strix -DONEBIT_BUILD_HELM_TAURI=ON
cmake --build --preset release-strix --target helm-tauri
```

The CMake target shells out to `npm install && npm run tauri build`.
First run requires Node 20+, npm, and a Rust toolchain (cargo) with
`tauri-cli` installed (`cargo install tauri-cli@^2`).

## Develop

```bash
cd cpp/helm-tauri
npm install
npm run tauri dev
```

## Initial layout

```
+--------+--------+
| halo   | MCP    |
+--------+--------+
| bench  | runbook|
+--------+--------+
```

## Tile contracts

- **HaloTile** — recent halo agent transcript via `/v1/chat/completions`
  SSE on lemonade :8180, model `1bit-systems/halo-1bit-2b`. Push-to-talk
  button placeholder; the real WebRTC bridge wires to `cpp/echo` later.
  Compact list when tiled, full transcript + tool-call viewer when
  maximized.
- **MCPTile** — GET halo-agent's tool registry; render as a card grid.
  Click a card to expand to an invocation form with arg fields.
- **BenchTile** — polled metrics from lemonade `/health` +
  `/system-stats`. Plain CSS sparklines (no chart lib — bundle stays
  small). Shows tok/s, prompt tok/s, GPU mem, system RAM, KV-cache
  size.
- **RunbookTile** — markdown viewer over `cpp/agent/configs/runbooks/`
  via `markdown-it`. Pinned ToC on the left, content on the right.

## Future tiles

- **EchoTile** — voice-loop status + waveform.
- **TraceTile** — rocprof / perf-counter live view.
- **InstallerTile** — `1bit install` GUI on top of `packages.toml`.
- **PowerTile** — RyzenAdj curves + GFXOFF state.

## Voice and UX rules

Calm-technical. No memes, no 80s movie quotes — those live on the GH
README and the wiki. Dark theme by default. Pure CSS, no Tailwind.

## Tauri allowlist

Only `http://127.0.0.1:8180/*` (lemonade) and `tauri://localhost` are
allowed by the IPC + fetch CSP. No telemetry, no remote analytics.
