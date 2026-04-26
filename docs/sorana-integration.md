# Sorana → 1bit-server integration

Research date: 2026-04-19. Scope: pointing the Sorana Visual AI Workspace
(https://tetramatrix.github.io/Sorana/) at our local `1bit-server`
(127.0.0.1:8180, OpenAI-compat) without installing Sorana on this box.

## Summary

**Sorana does not work out-of-the-box with 1bit-server.** Three blockers
stacked on top of each other, in order of severity:

1. **Linux is not a supported OS.** The `Tetramatrix/Sorana` GitHub repo
   has **zero releases** and the landing page states "Operating System:
   Windows 11 (64-bit)". The github.io page is a marketing site, not a
   distribution channel for us. There is no tarball, AppImage, `.deb`,
   or Flatpak to install.
2. **Sorana's backend layer is Lemonade-first.** The sister project
   `Tetramatrix/lemonade-python-sdk` is explicitly "powering the Sorana
   AI workspace." The SDK auto-discovers Lemonade by port-scanning
   `[8000, 8020, 8040, 8060, 8080, 9000, 13305, 11434]` and hitting
   `/api/v1/health`. Our 1bit-server is on **:8180** and serves `/v1/*`
   (no `/api/v1` prefix). So even if a Linux build existed, default
   discovery would not find us.
3. **No documented base-URL override.** Neither the Sorana README nor the
   landing page mentions a config-file override for a custom OpenAI base
   URL. The SDK's `LemonadeClient(base_url=...)` takes one, but Sorana
   itself is closed-source — we can't confirm the desktop app exposes
   that knob.

The good news: Lemonade Server (`lemonade-server.ai`) is itself
**OpenAI-compatible** at `http://localhost:13305/api/v1`. That means the
schema Sorana's SDK expects is the same schema 1bit-server already
emits. The only thing missing is a route/port fixup. A 60-line axum or
Bun proxy ("halo-masquerade") that listens on `127.0.0.1:13305`, exposes
`/api/v1/health` + `/api/v1/models`, and forwards `/api/v1/chat/completions`
to `127.0.0.1:8180/v1/chat/completions` makes Sorana happy — **on
Windows**. On Linux you would additionally need Wine or wait for a Linux
build from Tetramatrix.

Bottom line: this is option **(b)** — Lemonade-only, shim required —
gated behind a bigger option **(c)** problem: **no Linux binary
exists**. For a 1bit systems user today, Sorana is not a reachable frontend.

## Shim decision (a/b/c)

- **(a) Sorana accepts base_url → zero work.** Not supported. No
  documentation of such an override exists in any public Tetramatrix
  surface area.
- **(b) Sorana is Lemonade-only → write a small proxy.** This is our
  situation *if and when* a Linux build ships. The shim is trivial
  because both ends already speak OpenAI schema.
- **(c) Sorana bundles its own inference.** Partially true — the
  Sorana README lists a bundled ~806 MB offline model. That does not
  block us; it just means a user can run Sorana without any backend at
  all if they choose. For 1bit systems users who want their strix halo to
  serve tokens, they would still need the shim.

## Setup steps (hypothetical — Windows-only today)

Once a Linux build of Sorana ships (or a user is running it from
Windows over Headscale against the strix halo), the integration path
is:

### 1. Run the halo-masquerade shim

Bind `127.0.0.1:13305` (the default Lemonade port that Sorana's SDK
discovers first). Forward everything to 1bit-server.

Sketch of the axum shim (do **not** build this sprint — wait until a
real Sorana client exists to test against):

```rust
// crates/halo-masquerade/src/main.rs — SKETCH ONLY
// binds :13305, speaks /api/v1/* (Lemonade), forwards to :8180/v1/*
async fn main() -> anyhow::Result<()> {
    let app = Router::new()
        .route("/api/v1/health",          get(|| async { Json(json!({"status":"ok"})) }))
        .route("/api/v1/models",          get(proxy_models))
        .route("/api/v1/chat/completions", post(proxy_chat))
        .route("/api/v1/completions",      post(proxy_completions))
        .route("/api/v1/embeddings",       post(proxy_embeddings))
        .with_state(reqwest::Client::new());
    let listener = TcpListener::bind("127.0.0.1:13305").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
// proxy_* strip "/api" and forward to "http://127.0.0.1:8180/v1/..."
// Add Authorization: Bearer sk-halo-... on the outbound if Caddy-gated.
```

Same shape as the existing `1bit-lemonade` crate — in fact we should
likely **fold this into `1bit-lemonade`** rather than spawn a new
crate. `1bit-lemonade` already binds a gateway port and already has the
"proxy to 1bit-server" intent documented in its `lib.rs`. Add the
`/api/v1/*` routes next to the `/v1/*` routes it already serves; switch
the default bind to `127.0.0.1:13305` for Sorana mode.

### 2. Point Sorana at it (if a config exists)

Sorana's README documents a `.sorana/` data folder but **no config
file schema**. If and when Tetramatrix publishes one, the expected
shape — based on the SDK's `LemonadeClient(base_url=...)` signature —
would be something like:

```json
// ~/.config/sorana/config.json  (HYPOTHETICAL — format unconfirmed)
{
  "backend": "lemonade",
  "lemonade": {
    "base_url": "http://127.0.0.1:13305",
    "auto_discover": false
  }
}
```

If no config file exists, just running the shim on `:13305` is enough:
the SDK's port-scan will find it as the first hit.

## Bearer token note

1bit-server behind Caddy at `https://strixhalo.local/v2/chat/completions`
requires `Authorization: Bearer sk-halo-...`. The **local** 1bit-server
on `127.0.0.1:8180` does not — Caddy is the bearer gate. Two options:

- **Local mode (recommended).** Run the shim alongside 1bit-server on
  the same box; shim → `127.0.0.1:8180/v1/*` directly; no bearer
  needed. This is the "closet PC" case.
- **Remote mode.** Shim on the user's laptop → `https://strixhalo.local/v2/*`
  over Headscale. Shim must inject `Authorization: Bearer sk-halo-...`
  on every outbound request. Store the token at
  `~/.config/halo-masquerade/token` (0600). Do **not** ask Sorana to
  add it — Sorana does not expose an auth-header knob for Lemonade
  backends (the SDK has no auth parameter).

## Known limitations

- **Streaming (SSE).** 1bit-server emits `text/event-stream` when
  `stream=true`. The shim must `flush_interval: -1` its proxy (same
  trick as Caddy). Sorana's SDK behavior with streaming is
  undocumented — assume it works because the Lemonade server ships
  streaming too.
- **Function calling / tools.** 1bit-server's tool-call surface is
  currently a TODO per `crates/1bit-server/src/routes.rs`. Sorana
  advertises "no-code agent pipelines" which likely rely on tool
  calls. Expect agent-builder workflows to misbehave until 1bit-server
  gets real tool dispatch.
- **MCP.** Sorana does not advertise MCP-client support. Our
  `1bit-mcp` bridge is not relevant to this integration.
- **Embeddings / audio / image.** The Lemonade SDK mentions embeddings,
  TTS, STT, rerank, and image endpoints. 1bit-server only serves
  `/v1/chat/completions` + `/v1/completions` + `/v1/models`. Any Sorana
  workflow touching embeddings or audio will 404 against the shim
  unless we forward to a different backend (sd-server for images
  already lives at `127.0.0.1:8081`; embeddings have no home yet).
- **Model list.** 1bit-server reports whatever `lemond` has
  loaded. Sorana may display only one model (the current halo model).
  That is fine for text chat; it is a confusing UX for a "workspace"
  that expects to route across models.
- **Closed-source desktop app.** We cannot patch Sorana. If it decides
  to reject a health response that does not match a hard-coded shape,
  we find out at runtime. There is no source to grep.

## Next steps (Monday morning, mics arriving)

Do **not** spend the morning on Sorana. The desktop client does not
ship on Linux. Instead:

1. Confirm the voice loop (`whisper.cpp` STT live, `halo-kokoro` TTS
   pending) is the day's focus — that runs on our stack natively.
2. If a user *insists* on a polished desktop workspace pointed at
   1bit-server, recommend **open-webui** or **LibreChat** instead.
   Both accept an explicit OpenAI base URL + bearer. One `docker run`
   or one systemd unit away. Zero shim required.
3. Park Sorana. Re-evaluate when Tetramatrix ships a Linux build **and**
   publishes a config schema. Watch `github.com/Tetramatrix/Sorana/releases`;
   the tag to wait for is the first non-Windows asset.

If the shim-in-`1bit-lemonade` work lands anyway (because we wanted it
for other Lemonade-native clients like AnythingLLM), Sorana benefits
automatically on whatever day its Linux build drops.

## Sources

- Sorana landing — https://tetramatrix.github.io/Sorana/
- Sorana repo — https://github.com/Tetramatrix/Sorana (README only,
  no source, 0 releases as of 2026-04-19)
- Lemonade Python SDK — https://github.com/Tetramatrix/lemonade-python-sdk
  (port-scans `[8000, 8020, 8040, 8060, 8080, 9000, 13305, 11434]`,
  hits `/api/v1/health`)
- Lemonade Server (upstream, AMD) — https://lemonade-server.ai/
  (OpenAI-compatible at `http://localhost:13305/api/v1`)
- 1bit-server routes — `crates/1bit-server/src/routes.rs` (OpenAI
  `/v1/*` paths, no `/api/v1` prefix, bound on `:8180`)
- 1bit-lemonade scaffold — `crates/1bit-lemonade/src/lib.rs` (future
  home for the shim; currently serves only `/v1/models` + `/_health`
  on `:8200`)
