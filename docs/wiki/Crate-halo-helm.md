---
phase: implementation
owner: cartograph
renamed_from: halo-gaia
rename_date: 2026-04-20
---

# Crate: 1bit-helm

> Renamed from `halo-gaia` on 2026-04-20 to avoid clash with AMD GAIA
> (amd-gaia.ai). See `docs/wiki/AMD-GAIA-Integration.md`. The old name is
> retained as an install alias in `packages.toml` (`[component.gaia]` →
> `helm`) but no source tree reference to `halo-gaia` / `halo_gaia` should
> remain outside the AMD-GAIA comparison doc.

## Problem

Operator + end-user desktop client for 1bit systems. Today the paths are (a) browser at `/studio/voice/` (Apple watch to Android phone to laptop), (b) `halo` CLI (terse, power-user). Nothing pulls everything into one pane: live metrics + voice chat + skill + memory inspection + remote mesh peers. 1bit-helm is that pane.

Name: a ship's helm is the single surface where the pilot reads instruments and applies control. Fits the role better than the original "halo-gaia" once AMD shipped their own GAIA.

## Invariants

1. **Single binary, no runtime deps.** Native Rust GUI on egui/eframe (glow backend). Ships as one ELF, runs on Linux first, macOS later. No Electron, no web-view shell. No Python, no node. Rule A applies all the way to the user's desktop.
2. **Works offline.** Must render + let the user manage local skills / memory / models even when 1bit-server is down.
3. **Mesh-aware.** The same binary running on any mesh peer (laptop, second mini-PC, Pi) should see the same 1bit systems state by pointing at the mesh IP. No per-device state drift.
4. **Bearer secrets never land in files.** Tokens live in the system keyring (secret-service / kwallet / libsecret). No plaintext token on disk.
5. **One keyboard trip equals one user intention.** No popup dialogs for routine ops. CLI power-user can drive every pane with keyboard alone.

## Non-goals

- Not a web-browser shell (use `/studio/voice/` in Firefox if you want a browser — this is the native client lane).
- Not a model trainer — 1bit-helm never spawns training jobs. Training happens on Battlemage or a rented box; 1bit-helm consumes the resulting weights via `halo install`.
- Not a chat-logging product — conversation history lives in the server-side FTS5 sessions DB (1bit-agents::sessions), NOT in 1bit-helm. 1bit-helm is a view.
- Not a web-facing product. Distribution is via `halo install helm` (aliased from legacy `halo install gaia`) or `cargo install 1bit-helm` — no signed installers for Windows / macOS ahead of Linux.
- Not a multi-tenant app — each user runs their own mesh peer.

## Interface

### Binary

```text
1bit-helm                # opens the window
# env:
HALO_HELM_URL    # default http://127.0.0.1:8180
HALO_HELM_MODEL  # default halo-1bit-2b
HALO_HELM_TOKEN  # optional bearer
# Legacy HALO_GAIA_* env vars are honored as fallbacks.
```

### Public API (crate)

```rust
pub struct HelmApp { /* ...config, panes, transport, runtime handle... */ }
impl HelmApp {
    pub fn new(cfg: SessionConfig) -> Self;                                    // runtime-free; test-safe
    pub fn from_cc(cc: &eframe::CreationContext, cfg: SessionConfig) -> Self;
    pub fn attach_runtime(&mut self, handle: tokio::runtime::Handle,
                          http: reqwest::Client);                              // called once from main
    pub fn refresh_skills(&mut self);                                          // reads ~/.halo/skills
    pub fn refresh_memory(&mut self);                                          // reads ~/.halo/memories
    pub fn refresh_models(&mut self);                                          // spawns GET /v1/models
    pub fn send_chat(&mut self);                                               // spawns SSE POST
}

#[derive(Serialize, Deserialize, Copy, Clone, Eq, PartialEq)]
pub enum Pane { Status, Chat, Skills, Memory, Models, Settings }
```

`HelmApp` implements `eframe::App`. Panes:

| Pane     | Source of truth                                                 | Write path                            |
| -------- | --------------------------------------------------------------- | ------------------------------------- |
| Status   | SSE subscription to `http://127.0.0.1:8190/_live/stats`         | read-only                             |
| Chat     | `POST /v1/chat/completions` (gateway :8200), `stream: true`     | stream worker → `UiMsg::ChatDelta`    |
| Skills   | `onebit_agents::SkillStore::list` + body on row-click            | `$EDITOR` spawn on "Edit"             |
| Memory   | `onebit_agents::MemoryStore::list(Memory \| User)`               | inline "Add" → `store.add(kind, ...)` |
| Models   | `GET /v1/models` (gateway :8200)                                | Load button → toast (model-swap TBD)  |
| Settings | `keyring::Entry` / XDG file via `Bearer`                        | Save/Reset bearer                     |

Top bar: 6-button pane switcher. Left side panel: the same 6 panes as a radio list, plus a small config summary (default model + landing URL). Bottom strip: live dot + brand + loaded model + tok/s + GPU temp/util, or an error/toast row.

### Persistence

eframe's `persistence` feature stores state in the platform config dir (`~/.config/1bit-helm/` on Linux, via eframe's default). Today we persist the last-open pane under key `helm.current_pane`; window size is handled by eframe's own viewport save. Schema is `serde_json`-compatible so we can change the backing ron later without a migration.

## Test matrix

| Invariant                                               | Test                                                          |
| ------------------------------------------------------- | ------------------------------------------------------------- |
| App constructs without window / tokio / FS access       | `app::tests::new_constructs_without_panic_...`                |
| Status-first default (invariant 5 keyboard-trip)        | `app::tests::default_pane_is_status`                          |
| Pane-switcher coverage (tripwire on new variants)       | `app::tests::pane_all_covers_six_panes...`                    |
| Stable pane labels (top-bar text contract)              | `app::tests::pane_labels_are_stable_strings`                  |
| Persistence round-trip (serde stays wired)              | `app::tests::pane_round_trips_through_serde`                  |
| Skills reads from tempdir-seeded root                   | `app::tests::refresh_skills_picks_up_tempdir_seed`            |
| Memory reads from tempdir-seeded root                   | `app::tests::refresh_memory_reads_tempdir_seed`               |
| Chat `stream: true` body shape                          | `app::tests::build_chat_body_sets_stream_true_and_includes_turns` |
| SSE delta → assistant-turn state machine                | `app::tests::apply_ui_msg_chat_delta_then_done_lands_assistant_turn` |
| Telemetry disconnect flips `live_connected` flag        | `app::tests::apply_ui_msg_telemetry_disconnected_flips_state` |
| Conversation log roundtrip (write + read)               | `app::tests::conversation_log_flushes_on_save_path` + `conv_log::tests::roundtrip_write_then_read_preserves_turns` |
| Keyring fallback writes 0600 + reloads                  | `bearer::tests::file_fallback_roundtrip_creates_0600_and_reloads` |
| Bearer clear nukes on-disk copy                         | `bearer::tests::clear_removes_on_disk_copy`                   |
| Models pane parses OpenAI `/v1/models` shape            | `models::tests::parses_standard_openai_shape`                 |
| Landing SSE parse (full + minimal + garbage)            | `telemetry::tests::parse_stats_full_shape` + `..._tolerates_missing_fields` + `..._rejects_garbage` |

Transport modules (`client`, `conversation`, `session`, `stream`) keep their pre-rename test coverage.

## Phase: implementation

Promoted from `solutioning` on 2026-04-20 when all five panes went live against real transports. The scaffolded "not wired yet" placeholders are gone; everything renders from real SSE / REST / filesystem sources.

Closed since the solutioning cut:

- [x] **Async plumbing for network panes.** A multi-thread tokio runtime is spun up on the side in `main`; `HelmApp::attach_runtime(handle, http)` hands the app a `Handle` + shared `reqwest::Client`, and workers post back via `mpsc::UnboundedSender<UiMsg>`. `HelmApp::new` stays runtime-free — tests construct it without touching tokio (see `app::tests::new_constructs_without_panic_and_defaults_are_sane`).
- [x] **Keyring backend selection.** `keyring` 3.x on `sync-secret-service` + `crypto-rust` features. Linux gets `dbus-secret-service` (pure Rust, no libsecret C dep). File fallback: `~/.config/1bit-helm/bearer.txt` chmod 0600. Backend on the current session is surfaced in the Settings pane via `Bearer::backend()`.
- [x] **Status-pane live probe.** Subscribes to `http://127.0.0.1:8190/_live/stats` (1bit-landing SSE). Long-lived connection with exp-backoff reconnect (500 ms → 5 s cap). Cadence matches the server's 1.5 s SSE_INTERVAL.
- [x] **Chat pane streaming.** `POST /v1/chat/completions` with `stream: true`, reuses the hand-rolled `parse_sse_line` parser from `stream.rs`. Token-by-token append to the streaming bubble; `stick_to_bottom(true)` ScrollArea.
- [x] **Skills + Memory pane reads.** Direct calls to `onebit_agents::SkillStore::list` + `onebit_agents::MemoryStore::list(MemoryKind)`. Both accept `..._root_override: Option<PathBuf>` so tests seed a tempdir.
- [x] **Conversation log on close.** Flushed from `eframe::App::save` into `~/.halo/helm/conversations/<ts>.jsonl` (roundtrip tested).
- [x] **Desktop entry + icon.** `cpp/helm/assets/1bit-helm.desktop` + `cpp/helm/assets/icon.svg` (copy of the `1b` cyan monogram from `1bit-site/assets/logo.svg`).
- [x] **Brand surfacing.** Top bar shows "1bit-helm — 1bit monster"; bottom strip and Settings → About show `1bit monster` and `1bit.systems`.

Remaining for `verified` (needs user manual-run screenshot):

- [ ] Manual run on strixhalo: screenshot of Status pane with live telemetry + Chat pane streaming a real reply + Models pane populated from the gateway.
- [ ] Model-swap API. Today the "Load" button in the Models pane emits a toast ("halo-server loads models at startup today; model-swap API TBD"). Future hook: a `/v1/models/load` endpoint on lemonade.
- [ ] egui theming. Shipped with egui's default `Visuals::dark`. A future polish pass can read the system prefers-colour-scheme hint.
- [ ] 1bit-server lifecycle (observe-only today; the `systemctl --user start/stop` actions live in `halo` CLI — helm does not shell out).

## Runtime system deps

eframe's `glow` backend is OpenGL over `winit`. On a stock CachyOS / Arch box that means the existing `mesa` + `libxkbcommon` + `wayland-client` + `libxcb` packages are sufficient — nothing new to add to `packages.toml`. Noting here so a future PKGBUILD author doesn't need to re-trace:

- `libxkbcommon` (winit keyboard)
- `libwayland-client` / `libwayland-egl` (Wayland path)
- `libxcb`, `libx11` (X11 fallback)
- `libgl` (mesa / swrast)
- `fontconfig` (default-fonts feature still bundles its own fonts, but fontconfig is required to render into them on Linux)

None of these are Python; all ship by default on the supported distros. We do NOT add them to PKGBUILD runtime deps until the first user on a slim Wayland-only machine hits a missing library.

## Cross-refs

- CLAUDE.md — Rule A (no Python) applies.
- `docs/wiki/SDD-Workflow.md` — phase gate + review specialist mapping.
- `docs/wiki/VPN-Only-API.md` — mesh-awareness requirement (invariant 3).
- `docs/wiki/AMD-GAIA-Integration.md` — rename rationale + AMD GAIA integration story.
- `cpp/helm/` — implementation.
