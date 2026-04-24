# CUTOVER.md — flipping `/v1/*` from gen-1 C++ to gen-2 Rust

> **Status: completed at v0.1.0 (2026-04-24).** This document is retained as a historical runbook; the cutover described below has already happened. `/v1/*` now terminates at `127.0.0.1:8180` (`1bit-server`, Rust). The gen-1 `bitnet_decode` (C++) service on `:8080` is disabled and the `/v1-legacy/*` escape hatch has not been needed. The gates and procedure are preserved in case a future backend swap (e.g. NPU-backed serving) needs the same pattern.

When to run: after the 72-hour shadow burn-in clears the gates below.
What happens: Caddy's `/v1/*` route moves from `127.0.0.1:8080`
(`bitnet_decode`, C++) to `127.0.0.1:8180` (`1bit-server`, Rust). Gen-1
stays running on `:8080` for rollback; `/v1-legacy/*` is added as an
escape hatch.

## Cutover gates (all must pass)

| gate | target | current | status |
|---|---|---|---|
| PPL on wikitext 1024 tokens | `9.1607 ± 0.05` | 9.1805 (+0.0198) | ✓ |
| Shadow burnin exact-match rate | ≥ 90% | 96.66% | ✓ |
| Shadow burnin total rounds | ≥ 50,000 | 1,500 | ⌛ ~48h to go |
| v2 unreachable count | < 0.1% of rounds | 0% | ✓ |
| v1/v2 p95 latency gap | < 100 ms | 10 ms | ✓ |
| Median prefix-match length | ≥ 40 chars | 80 chars | ✓ |
| 1bit-server uptime since KV-reset fix | ≥ 72 h | ~10 h | ⌛ |
| CI on main | green | green | ✓ |

Re-run check:

```bash
1bit ppl
1bit bench
1bit doctor
1bit status
```

## Rollback safety

Gen-1 C++ bitnet_decode stays installed + running on `:8080`. If gen-2
misbehaves after cutover:

```bash
sudo caddy stop
sudo sed -i '/# CUTOVER:flipped/,/# CUTOVER:end/d' /etc/caddy/Caddyfile
sudo systemctl restart caddy        # restores pre-cutover routing
```

Takes under 10 seconds.

## Procedure

1. Verify gates via `1bit doctor && 1bit ppl && 1bit bench | head -15`.
   Abort if any fail.
2. Snapshot the current Caddyfile:
   `sudo cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.pre-cutover-$(date +%Y%m%d)`
3. Edit `/etc/caddy/Caddyfile`:
   - Change the `@authorized` catch-all upstream from `127.0.0.1:8080`
     to `127.0.0.1:8180`.
   - Add a new `handle_path /v1-legacy/*` block that still points at
     `127.0.0.1:8080` with the same bearer gate — this is the rollback
     lane for clients that need gen-1 specifically.
   - Wrap both changes between `# CUTOVER:flipped` / `# CUTOVER:end`
     comments so the rollback sed above knows what to strip.
4. `sudo caddy validate --config /etc/caddy/Caddyfile`
5. `sudo systemctl reload caddy` (admin is on since 2026-04-19, so
   reload works without a full restart).
6. Smoke:
   ```bash
   curl -sS -H "Authorization: Bearer $TOKEN" \
        https://strixhalo.local/v1/chat/completions \
        -d '{"model":"1bit-monster-2b","messages":[{"role":"user","content":"Hi"}],"max_tokens":8}' \
        | jq -c '.choices[0].message.content'
   ```
7. Post to #changelog via halo-anvil: "cutover done, /v1 now gen-2".

## What breaks (known)

- Clients that depend on C++-specific response quirks (field ordering
  in the JSON, non-deterministic `id` length) may regress. The Rust
  server matches the OpenAI shape exactly; gen-1 had minor drift.
  Fix: update client to use `jq` / a JSON parser, not string matching.
- Clients that used `stream: true` with SSE **must** handle the
  `data: [DONE]\n\n` sentinel. 1bit-server emits it; some gen-1 builds
  didn't. DSPy, OpenAI SDK, our 1bit-helm all handle it; ad-hoc curl
  consumers should check.

## Post-cutover (within 7 days)

- Remove `halo-bitnet.service` from `1bit install --list` default set
  (keep component in `packages.toml`, drop from `core` deps).
- Drop gen-1 from shadow-burnin (no longer useful as baseline; replace
  with 1bit-router itself on a second GPU once we have one).
- Update README.md's "Quickstart" to point at `:8180` by default.
- Tag a `pre-cutover` snapshot on `bong-water-water-bong/rocm-cpp` so
  we can roll back to the gen-1-compatible kernel set if needed.

## Fire-drill test (recommended before the real cutover)

Run the cutover against a second Caddy instance on a non-standard port:

```bash
CADDY_ADMIN=localhost:2020 caddy run --config /tmp/Caddyfile.drill
curl -H "Authorization: Bearer $TOKEN" https://strixhalo.local:8443/v1/...
```

Confirms the config reload + rollback sed both work before touching
the real box.
