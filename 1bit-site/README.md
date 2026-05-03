# 1bit-site

Static Cloudflare Pages site for `1bit.systems`.

The site is deliberately simple: vanilla HTML/CSS/JS, no framework, no CDN, no analytics. The landing page is the canonical public docs surface; legacy `/docs/*` paths redirect back to anchors on `/` through `_redirects`.

## Layout

```text
1bit-site/
  index.html       clean landing/docs page for the current stack
  install.sh       curl-pipe bootstrap that hands off to repo install.sh
  _headers         Cloudflare Pages headers
  _redirects       docs and short-link redirects
  robots.txt       crawler policy
  wrangler.toml    Cloudflare Pages project config
  assets/          local fonts, logos, CSS, JS, app screenshots
  blog/            static blog index
```

Current public architecture:

```text
Apps / SDKs -> 1bit-proxy :13306/v1 or :13306/api/v1
                 -> Lemonade :13305/api/v1
                 -> FastFlowLM :52625/v1

Open WebUI :3000 -> 1bit-proxy :13306/v1
Control plane    -> 1bit CLI + GAIA + systemd target
```

## Local Preview

```sh
cd 1bit-site
python3 -m http.server 8765
```

Open `http://127.0.0.1:8765`.

## Cloudflare Pages Deploy

Preview deploy:

```sh
cd 1bit-site
npx wrangler pages deploy . \
  --project-name=1bit-systems \
  --branch=preview-$(date +%Y-%m-%d) \
  --commit-dirty=true
```

Production deploy:

```sh
cd 1bit-site
npx wrangler pages deploy . \
  --project-name=1bit-systems \
  --branch=main \
  --commit-dirty=true
```

The custom domain `1bit.systems` is bound to the `main` branch in Cloudflare Pages. `_headers` is applied automatically.

## Rules

- Keep it static.
- Keep copy aligned with `README.md`.
- Do not publish local inference ports as internet-accessible services.
- Keep the inference endpoint contract primary.
- Keep the single control plane secondary: `1bit` CLI, GAIA, Open WebUI, and systemd.
- Keep Lemonade canonical, FLM on `:52625`, proxy on `:13306`, Open WebUI secondary.
