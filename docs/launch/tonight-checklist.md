# Tonight's soft-launch checklist

Target: flip all six domains live + collect first waitlist signups in ~45 min.

## 0. Prep (5 min)

- [ ] Cloudflare dashboard open (left tab)
- [ ] Namecheap dashboard open (right tab)
- [ ] Coffee / spliff

## 1. Namecheap → Cloudflare NS swap (20 min)

For each of `1bit.music`, `1bit.video`, `1bit.stream`, `1bit.audio`, `waterbon.me`:

1. Cloudflare → **Add a site** → enter domain → Free plan.
2. Note the two NS Cloudflare assigns (e.g. `lars.ns.cloudflare.com` / `maya.ns.cloudflare.com`).
3. Namecheap → Domain List → Manage → **Custom DNS** (dropdown) → paste both Cloudflare NS → green tick.
4. Repeat.

(`1bit.systems` is already there, skip.)

## 2. Cloudflare DNS per zone (8 min)

Each zone, add:

```
Type   Name   Target                      Proxy      TTL
CNAME  @      1bit-systems.pages.dev      Proxied    Auto
CNAME  www    1bit-systems.pages.dev      Proxied    Auto
```

## 3. Pages custom hostnames (6 min)

Pages project `1bit-systems` → **Custom domains** → add each:
- `1bit.music`, `www.1bit.music`
- `1bit.video`, `www.1bit.video`
- `1bit.stream`, `www.1bit.stream`
- `1bit.audio`, `www.1bit.audio`
- `waterbon.me`, `www.waterbon.me`

SSL provisions in ~2 min per domain.

## 4. Deploy waitlist Worker (5 min)

```bash
cd 1bit-site/workers
wrangler publish waitlist.js \
  --name waitlist \
  --route "*1bit.music/api/waitlist*" \
  --kv-namespace WAITLIST
```

Do once. The `--route` matcher is wildcard so the same worker serves every apex hostname.

## 5. Verify (3 min)

```bash
curl -I https://1bit.music           # expect 200
curl -I https://1bit.video           # expect 200
curl -I https://1bit.stream          # expect 200
curl -I https://1bit.audio           # expect 200 (→ /compress/)
curl -I https://waterbon.me          # expect 200 (→ /me/)
curl -sX POST https://1bit.music/api/waitlist \
  -H 'content-type: application/json' \
  -d '{"email":"test@1bit.systems"}' | jq .
```

## 6. Announce (varies)

Copy-paste posts in `docs/launch/social-posts.md`:
- [ ] Discord (halo-ai server)
- [ ] X / Twitter thread
- [ ] Reddit r/LocalLLaMA
- [ ] Reddit r/audiophile
- [ ] Reddit r/amd
- [ ] Hacker News (Show HN — save for post-Run-8 for "launch launch")
- [ ] Patreon announcement post

## 7. (Optional tonight) USPTO TEAS Plus (~$500)

- [ ] `1bit.systems` — Class 9 software
- [ ] `1bit.systems` — Class 42 SaaS
- [ ] `1bit-ac` — Class 9 software
- Pay by card, done in 20 min online.
