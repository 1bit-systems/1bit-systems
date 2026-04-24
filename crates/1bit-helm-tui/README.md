# 1bit-helm-tui

Interactive tmux-style operator TUI for the 1bit stack. Pairs with
`1bit-helm` (egui/eframe desktop) as the headless / SSH surface.

## Status

**v0 skeleton.** Three hardcoded panes, Ctrl-q / Ctrl-c quit, no
interactivity yet. Ratatui + crossterm event loop is live; the pane
tree + layout persistence are staged but not yet driving the render.

Full `cargo test -p onebit-helm-tui` green. Clippy -D warnings clean.

## Vision

The endgame is a **fully dynamic scalable window system** that an
operator can rearrange on the fly. Goals, in priority order:

1. **Move panes anywhere**, any time. Drag borders with the mouse,
   split and merge with tmux bindings (`|` `-` `h j k l H J K L z`).
   No modal dialogs. No stop-the-world rearrange.

2. **Content adapts to pane size.** A pane the size of a postage
   stamp shows only the traffic-light indicator (green / yellow /
   red). Drag it bigger and more detail fills in — last 3 log lines,
   decode tok/s number, kernel time breakdown. Drag even bigger and
   you get the full live chart. Same widget, progressive disclosure.

3. **One widget per service / agent.** Every running thing
   (`1bit-server`, `1bit-watch-discord`, each `1bit-agents` specialist,
   `halo-whisper`, `halo-kokoro`, etc.) gets its own little card. You
   open the ones you care about, close the rest. Shared status bar
   across all of them at the bottom.

4. **Fun, not just informative.** Agents should feel alive. A small
   Conway / particle animation per agent pane while idle — wakes up
   into real activity as the agent dispatches. Same aesthetic as the
   turquoise + magenta theme on `1bit.systems`, not boring green-on-
   black mainframe vibe.

5. **Save + restore layouts.** `~/.config/1bit/tui-layout.json`.
   Named presets (Ctrl-b 1..9) so you can flip between "full monitor
   dashboard", "just the training run", "pre-deploy checklist" etc.

## Contributors welcome

**We are looking for a GUI / UX specialist** to turn this skeleton
into that full vision. Skills that map well:

- Rust + ratatui (or willingness to pick up — the API is small)
- Terminal UI / TUI animation experience (any language — we'll port)
- Strong opinions about pane behavior (ergonomics > novelty)
- Game-loop thinking — the 30 Hz render cadence is already set up

Open an issue on the repo or drop a line on the Discord
(https://discord.gg/EhQgmNePg) in `#ideas` or `#general` with a
prototype sketch — we'd rather see a gif of a moving pane than a
paragraph about intentions.

## Layout

```
crates/1bit-helm-tui/
├── Cargo.toml           # ratatui + crossterm + tokio + serde
└── src/
    ├── main.rs           # event loop, crossterm setup
    ├── pane.rs           # split tree (Node) + serde JSON
    ├── layout.rs          # load/save ~/.config/1bit/tui-layout.json
    ├── theme.rs           # turquoise + magenta palette
    └── widgets/
        └── mod.rs         # status / logs / gpu / power / kv / bench / repl stubs
```

## Running (v0)

```sh
cargo run -p onebit-helm-tui
```

Keys today: **Ctrl-q** / **Ctrl-c** quit. Everything else is scaffolding.

## Deep dive

See `docs/wiki/Helm-TUI.md` for the design brief: key bindings, pane
tree JSON schema, widget list, responsive-detail rules, animation
guidelines.
