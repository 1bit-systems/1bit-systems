---
phase: implementation
owner: scribe
---

# Crate: 1bit-landing

## Problem

Our LAN-side dashboard. Marketing page + live `/metrics` probe + per-service health on strixhalo.local:8190 (Caddy proxies this behind `/` + `/_health` + `/_live/*`). The operator opens a browser at `https://strixhalo.local`, sees the halo landing page, can eyeball tok/s / GPU temp / NPU uptime / model loaded. No bearer — LAN-only via hostname (mesh-gated by Headscale).

1bit-landing is the "glanceable" surface, distinct from `/studio/` (marketing for the public CF Pages deploy) and distinct from `/studio/voice/` (interactive voice). One box, one view.

## Invariants

1. **LAN-only. No bearer, no public exposure.** Caddy proxies `/` and `/_health` + `/_live/*` here WITHOUT the bearer gate. Only mesh peers can reach strixhalo.local to get to it.
2. **Read-only.** Surface never writes state. All it does is scrape 1bit-server `/v1/models` + rocm-smi + xrt-smi + systemctl output.
3. **Zero external assets.** Fonts, CSS, JS all inline or bundled. Works offline, loads instantly on a 4G phone.
4. **Updates live without page reload.** Server-Sent Events on `/_live/stats` push fresh metrics every 1–2 seconds. Browser renders with vanilla JS.
5. **Survives backend outages.** If 1bit-server is down, 1bit-landing still serves the static shell with a red status bar. It does NOT proxy through 1bit-server.

## Non-goals

- Not a configuration UI. Operators use `halo` CLI + `systemctl --user` + edit config files directly. 1bit-landing is glanceable, not mutable.
- Not a public marketing page (that's `1bit.systems/` served by Cloudflare Pages).
- Not a multi-tenant dashboard. One box, one view.
- Not a historical metrics store. Use `halo burnin` for time-series analysis.

## Interface

```
GET  /                    → HTML dashboard (inline CSS + JS)
GET  /_health             → plain "ok\n" if the binary is up, 200
GET  /_live/stats         → SSE stream, 1 event per ~1.5s, JSON { tok_s, gpu_temp_c, npu_up, loaded_model, shadow_burn_exact_pct }
GET  /_live/services      → SSE stream, service-status delta events
GET  /metrics             → Prometheus scrape endpoint (text format)
```

Binds `:8190` loopback by default. Overridable via `HALO_LANDING_BIND` env.

## Test matrix

| Invariant | Test |
|---|---|
| 2 (read-only) | `no_POST_routes_registered` |
| 3 (zero external assets) | `html_renders_without_outbound_fetch` — grep response body for `src="http`, assert none |
| 4 (live updates) | `sse_stats_emits_one_event_per_interval` |
| 5 (survives outages) | `onebit_server_down_still_serves_shell` — mock 127.0.0.1:8180 returning 500 |

## TODO

- [x] `/` HTML shell ships today (via `strix-landing.service` on :8190)
- [x] `/_health` simple liveness
- [x] `/metrics` Prometheus endpoint (shape verified by external scrapers)
- [x] `/_live/stats` SSE stream wired to real sources (1bit-server `/v1/models` + `/metrics`, `rocm-smi`, `xrt-smi`, shadow-burnin jsonl, `systemctl --user`). 1 s cache TTL, 1.5 s SSE cadence. 2026-04-20.
- [x] `/_live/services` SSE deltas on tracked `strix-*` units — first frame is a full snapshot, subsequent frames only flips.
- [x] Tok/s live gauge pulled from 1bit-server telemetry (was static; now `tok_s_decode` from `/metrics` via `/_live/stats`).

## Spec cross-ref

| Spec section | Code file |
|---|---|
| Interface / routes | `cpp/landing/src/main.rs` |
| Invariant 3 (zero external) | `cpp/landing/assets/` — all inline, no CDN refs |
| Invariant 4 (SSE) | `cpp/landing/src/main.rs` `live_stats_sse` + `live_services_sse` |
| Telemetry sources | `cpp/landing/src/telemetry.rs` (cache + `Sources` struct) |

## Phase: implementation

Promote to `verified` once:
- SSE `/_live/stats` pushes real 1bit-server telemetry (not static values)
- Zero-external-asset test (grep body) added + green
- Outage smoke: kill strix-server, confirm landing shell still renders within 100 ms

## Cross-refs

- `docs/wiki/SDD-Workflow.md` — phase framework
- `docs/wiki/VPN-Only-API.md` — LAN-only policy
- `cpp/landing/` — code
- `strixhalo/systemd/strix-landing.service` — user unit
- Caddy wires it at `/` and `/_health` via the `@landing` matcher
