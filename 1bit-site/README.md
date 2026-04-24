# 1bit-site

Static site for **1bit.systems** — vanilla HTML + inline CSS + vanilla ES
modules. Zero CDN, zero analytics, zero framework. Total page weight
under 60 KB per page including styles.

## Layout

```
1bit-site/
  index.html              — landing (hero, proof band, APU play, metrics, install, stack, crates)
  install.sh              — the curl-pipe-bash payload (do not rewrite content — link only)
  docs/                   — docs landing + install/quickstart/architecture/models/api
                            desktop/troubleshooting/faq/contributing/changelog
                          — plus rendered wiki pages via docs/build.sh (rustc-only md→html)
  blog/                   — stack-only engineering posts
  assets/                 — logo.svg, style.css, docs.css
  _headers                — CSP + cache rules (CF Pages convention)
  _redirects              — short permalinks + trailing-slash fixes
  robots.txt              — crawler policy (parked surfaces disallowed)
  wrangler.toml           — CF Pages project config
```

## Rebuild the wiki pages

```bash
bash 1bit-site/docs/build.sh        # rustc -O builds the renderer, then renders docs/wiki/*.md
```

## Deploy to Cloudflare Pages

**Preview deploy (no production domain changes):**

```bash
cd 1bit-site
npx wrangler pages deploy . \
  --project-name=1bit-systems \
  --branch=preview-$(date +%Y-%m-%d) \
  --commit-dirty=true
```

Wrangler prints a preview URL like `https://preview-2026-04-21.1bit-systems.pages.dev/`.
This is the URL to share for review. It does **not** move the production
domain.

**Production promote (only after user says ship):**

```bash
cd 1bit-site
npx wrangler pages deploy . \
  --project-name=1bit-systems \
  --branch=main \
  --commit-dirty=true
```

Custom domain `1bit.systems` is bound to the `main` branch in the CF
dashboard (Pages → Custom domains). `_headers` is applied automatically.

## Local preview

```bash
cd 1bit-site
python3 -m http.server 8765
# open http://localhost:8765
```

Or any other static file server — there's no build step.

## Constraints (codified)

- **No framework.** No React, no Vue, no Svelte, no Astro, no Next. Vanilla.
- **No CDN.** No Google Fonts. All assets live in this repo.
- **No analytics.** Rule A-vibe: zero third-party fetches.
- **Under 300 KB per page.** Budget holds with inline CSS.
- **Mobile-first.** 320 px min. Sticky nav collapses to hamburger below 820 px.
- **`prefers-reduced-motion`** respected. Hero glow disables.
- **Semantic HTML.** Every interactive element keyboard-reachable.
- **Inline SVG logos.** No image fetches.
- **Calm technical voice.** No memes, no movie quotes. GH README carries personality.

## Domain + DNS notes

- Root domain `1bit.systems` lives at Cloudflare, nameservers on CF.
- Pages project name: `1bit-systems` (hyphenated; CF project slugs don't allow spaces).
- `install.sh` is served from this Pages project at `/install.sh`. `_headers` sets the MIME to `text/plain`.
