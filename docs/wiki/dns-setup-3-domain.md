# DNS setup for 1bit.music + 1bit.video + 1bit.stream

Three domains registered 2026-04-23, all pointed at the same Cloudflare Pages origin as `1bit.systems`. Pages routes by Host header via `_redirects`.

## Registrar → Cloudflare

For each of the three new domains at Porkbun (or whichever registrar you used):

1. Porkbun dash → Domain Management → `1bit.music`
2. Authoritative Nameservers → set to Cloudflare's:
   - `ns1.cloudflare.com`
   - `ns2.cloudflare.com`
   (or whichever pair Cloudflare assigns — Cloudflare dash shows the exact two)
3. Wait for propagation (5–60 min)

Repeat for `1bit.video` and `1bit.stream`.

## Cloudflare zone records

For each of the three zones, add a single CNAME to the existing Pages project (`1bit-systems.pages.dev`):

```
Type   Name   Target                         Proxy   TTL
CNAME  @      1bit-systems.pages.dev         Proxied Auto
CNAME  www    1bit-systems.pages.dev         Proxied Auto
```

Pages project custom-hostnames panel: add `1bit.music`, `1bit.video`, `1bit.stream`, `www.1bit.music`, `www.1bit.video`, `www.1bit.stream`. Cloudflare auto-provisions SSL via Universal SSL within minutes.

## Verification checklist

After DNS propagates:

- `curl -I https://1bit.music` → 200, returns the `/music/index.html` content
- `curl -I https://1bit.video` → 200, returns `/video/index.html`
- `curl -I https://1bit.stream` → 200, returns `/stream/index.html`
- `curl -I https://1bit.systems` → 200, returns the existing `index.html` (unchanged)
- `curl -I https://www.1bit.music` → 301 → `https://1bit.music`

## Future routing

When the tier-gated streaming endpoints go live, route pattern becomes:

```
1bit.music/browse/*        → Pages (static catalog index)
1bit.music/api/*           → 1bit-stream server (tunneled from sliger)
1bit.music/player/*        → Pages (WASM player)
```

`1bit-stream` backend runs on sliger; Cloudflare Tunnel exposes it through `api.1bit.music` (subdomain) without opening firewall ports. Current `strix-cloudflared.service` pattern already covers this; add a second config entry once the stream backend is deployed.

## Recovery

If any domain fails to validate TLS, most common causes:

- Nameserver not propagated yet — wait
- CNAME flattening disabled on Cloudflare — ensure "CNAME flattening" is on for apex records
- Pages custom hostname not added — manual add in Pages → Custom domains

Trademark note: owning the domain ≠ owning the mark. USPTO TEAS Plus filing on `1bit.music` / `1bit.video` / `1bit.stream` as word marks is the separate next step (~$350/class × 3 or bundle under the umbrella `1bit.systems` application).
