# VPN-only API — 10-seat private beta testing ground

**Policy (2026-04-20):** the 1bit systems test API does not expose a public endpoint. Every client — laptop, mobile, browser, agent — reaches the box over a private Headscale mesh. No Cloudflare Tunnel, no public ACME cert, no port-forward. The website is a marketing surface; access to compute is a separate, authenticated, mesh-only path.

## Two fences, not one

1. **Fence 1 — Mesh membership.** You're not talking to strixhalo unless your device is on the Headscale mesh. Headscale runs on strixhalo itself and uses Tailscale clients (macOS, Linux, Windows, iOS, Android) to peer in. We hand out a **single-use, 24-hour pre-auth key** per invitee.
2. **Fence 2 — Per-user bearer token.** Even on the mesh, every `/v2/*` / `/lemon/*` / `/sd/*` Caddy route requires a `Authorization: Bearer sk-halo-...` header. Tokens are per-user, stored in `/etc/caddy/bearers.txt` (root:caddy 0640), one per line. Caddy matches any line at request time; we can revoke one user without affecting the other nine.

Every bearer is also time-boxed. Each line carries an `expires <ISO>` comment stamp written by `1bit-mesh-invite.sh`, and a nightly sweeper auto-revokes once the stamp is in the past. See [Beta-10-Day-TTL](Beta-10-Day-TTL.md).

Bearer line format (post 2026-04-20):

```
sk-halo-XXXX  # handle  # issued 2026-04-20T14:00Z  # expires 2026-04-30T14:00Z
```

Both fences matter. Mesh-only without bearers = lateral movement risk between peers. Bearers without mesh = public endpoint, which we explicitly don't want today.

## 10 seats, why

We cap the beta at **10 users** because:

- Shadow-burnin parity passes today at 95.55% byte-exact. We don't want more users hitting the API than we can observe in logs.
- Per-user metrics + quota logic are not yet wired in the proxy/control layer. 10 users × single stream keeps us inside the local stack's capacity without queueing pathologies.
- One box, one drive, one operator. 10 is the number where we can still manually reconcile a mis-issued token, not 100.

Raise the cap in `strixhalo/bin/1bit-mesh-invite.sh` (`MAX_USERS=10`) once parity + quotas land.

## Onboarding flow

**Operator side** — as bcloud on strixhalo, one command per invitee:

```bash
strixhalo/bin/1bit-mesh-invite.sh <handle>
```

Outputs a ready-to-send message containing:
- single-use `--authkey` (24-hour TTL, Headscale-issued)
- per-user `sk-halo-<hex>` bearer token (random, 32 hex chars)
- the exact `tailscale up` command
- a `curl` smoke test
- mobile + browser notes

Script refuses the 11th invite (cap enforcement).

**Invitee side** — three commands:

```bash
# Linux (Arch / CachyOS / any distro with a pkgmgr that has tailscale)
sudo pacman -S tailscale && sudo systemctl enable --now tailscaled
sudo tailscale up --login-server https://headscale.1bit.systems --authkey <paste>

# Verify
curl https://strixhalo.local:8443/v2/v1/models -H "Authorization: Bearer <paste>"
```

Mobile: Tailscale app → Settings → "Use alternate coordination server" → paste `https://headscale.1bit.systems` → paste authkey.

Browser: install our internal root CA (Caddy-issued) then open `https://strixhalo.local:8443/studio/`.

## Revocation

```bash
strixhalo/bin/1bit-mesh-revoke.sh <handle>
```

Expires the Headscale authkey, drops the bearer line from `/etc/caddy/bearers.txt`, reloads Caddy. Sub-second.

Automated revoke on TTL expiry is covered by [Beta-10-Day-TTL](Beta-10-Day-TTL.md) — the same script gets called under the hood.

## Security posture (what the attacker model is)

| Threat | Defense |
|---|---|
| Drive-by on public endpoint | No public endpoint exists. `api.1bit.systems` does not resolve to a public IP. |
| Mesh peer compromise | Per-user bearer on every compute route. Compromised mesh peer gets one user's tokens, not all ten. |
| Token in git or a log | Tokens rotate per-invitee; one revoke + reissue, seconds. Tokens should never be logged by proxy/control services. |
| Replay across subdomains | `host` + bearer match in Caddy; a bearer for `/v2/` does not authorize `/sd/` unless we explicitly issue it. |
| Lost phone | Pre-auth key is single-use + 24-hour TTL; after onboarding the device has its own node key which we revoke via Headscale. |
| Nation-state-in-the-middle on WebPKI | We don't use WebPKI. Caddy's internal CA rules out any public CA compromise vector. |
| Lateral movement across peers inside mesh | Headscale ACL locks peers to the allowed 1bit proxy/service ports only. No peer-to-peer SSH/NFS/etc unless explicitly allowed. |

## What's on the public website vs the mesh

| Surface | Reach | Auth |
|---|---|---|
| `1bit.systems/` (Hugo marketing) | public, Cloudflare Pages | none |
| `1bit.systems/join/` | public | Discord invite to get onto mesh |
| `1bit.systems/docs/` | public | none |
| `1bit.systems/audio/` | public marketing page | none |
| `strixhalo.local:8443/studio/` | mesh-only (hostname resolves via tailnet DNS) | none — LAN-style |
| `strixhalo.local:8443/v2/*` | mesh-only | bearer token |
| `strixhalo.local:8443/lemon/*` | mesh-only | bearer token |
| `strixhalo.local:8443/sd/*` | mesh-only | bearer token |
| `wss://strixhalo.local:8443/audio/ws` | mesh-only | bearer token |

Public = brochure. Mesh = the product.

## Planned evolution

- **Auto-invite via Discord bot** — once 10 manual invites stabilize, a bot reads `/request-invite` slash commands and calls `1bit-mesh-invite.sh` under the hood. Keeps the cap-enforcement in one place.
- **Per-user quotas** (daily token limit, RPS cap) — server-side, not Caddy.
- **OIDC on Headscale** — today it's pre-auth keys. If we grow past 50 users we flip to OIDC via a hosted Keycloak or Pocket-ID.
- **Audit log** — shipped to the pi archive nightly (already have the rsync path).

## Cross-refs

- `strixhalo/bin/1bit-mesh-invite.sh` — the canonical invite script
- `strixhalo/bin/1bit-mesh-revoke.sh` — the canonical revoke script
- `strixhalo/bin/1bit-beta-expire.sh` — nightly 10-day TTL sweeper (see [Beta-10-Day-TTL](Beta-10-Day-TTL.md))
- `project_halo_network.md` memory — mesh topology + node IPs
- `docs/wiki/Cloudflare-Tunnel-Setup.md` — the *other* approach, currently NOT in use (template deliberately disabled)
- `studio-site/join/index.html` — the public-facing explanation + invite request form
- `/etc/caddy/Caddyfile` — where the bearer matching actually happens
