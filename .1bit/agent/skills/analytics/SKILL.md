---
name: analytics
description: Pull GitHub network traffic (clones, views, stars, referrers, paths) and Cloudflare Analytics (page views, unique visitors, bandwidth, top pages, countries) for the 1bit.systems project. Use when checking site traffic, GitHub popularity, or before a release to gauge interest.
---

# Analytics — GitHub + Cloudflare Traffic Dashboard

Pulls live analytics from both **GitHub** (clones, views, stars, referrers) and
**Cloudflare** (page views, unique visitors, bandwidth, geo breakdown) for the
`1bit.systems` project.

## Quick Usage

```bash
# Full dashboard (default: last 14 days)
bash ~/.pi/agent/skills/analytics/analytics.sh

# GitHub-only
bash ~/.pi/agent/skills/analytics/github-analytics.sh

# Cloudflare-only
bash ~/.pi/agent/skills/analytics/cf-analytics.sh

# Custom date range
bash ~/.pi/agent/skills/analytics/analytics.sh --since 2026-06-01 --until 2026-07-07

# Just today's numbers
bash ~/.pi/agent/skills/analytics/analytics.sh --today

# Update badges in 1bit-site/
bash ~/.pi/agent/skills/analytics/analytics.sh --badges
```

## Prerequisites

### GitHub: Zero-Setup (gh CLI)

**No token needed** — just authenticate `gh` once with your GitHub account:

```bash
gh auth login
# Follow the device auth flow (open URL, paste code)
```

Once authenticated, `gh` stores the token in the system keyring automatically.
All GitHub analytics scripts use `gh api` and work immediately — no env vars, no
manual token creation, no scopes to configure.

```bash
# Verify:
gh auth status
# Should show: ✓ Logged in to github.com account <you> (keyring)
```

For SSH-key-only setups (no `gh` CLI): add your SSH public key to GitHub,
then run `gh auth login` and choose the SSH flow. The scripts auto-detect
`gh` and use it.

### GitHub: Manual Token (fallback)

If `gh` CLI isn't available, the scripts fall back to the keyring → env var chain:

```bash
# Store in keyring:
secret-tool store --label="GitHub analytics token" service github-analytics token

# Or set env var:
export GITHUB_TOKEN="ghp_..."
```

Classic token, no scopes needed for public repo traffic data.

### Cloudflare API Token

| Method | Command |
|--------|---------|
| Keyring (recommended) | `bash ~/.pi/agent/skills/analytics/setup-keyring.sh` |
| Manual keyring | `secret-tool store --label="Cloudflare API token" service cloudflare-analytics token` |
| Env var | `export CLOUDFLARE_API_TOKEN="..."` |

Create a token at [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens)
with **Analytics:Read** permission.

### Cloudflare Zone ID (optional)

```bash
# Store in keyring:
secret-tool store --label="Cloudflare zone ID" service cloudflare-analytics zone
# Or set env var:
export CLOUDFLARE_ZONE_ID="abc123..."
```

Find it at Cloudflare dashboard → 1bit.systems → Overview → Zone ID (right sidebar).
If not set, the script auto-discovers it via `curl /zones?name=1bit.systems`.

## Data Sources

### GitHub Traffic API (free, included with all repos)

| Endpoint | What it Returns |
|----------|----------------|
| `/repos/{owner}/{repo}/traffic/clones` | Daily clone counts + unique cloners (14 days) |
| `/repos/{owner}/{repo}/traffic/views` | Daily page views + unique visitors (14 days) |
| `/repos/{owner}/{repo}/traffic/popular/referrers` | Top 10 referrer sources |
| `/repos/{owner}/{repo}/traffic/popular/paths` | Top 10 pages visited |
| `/repos/{owner}/{repo}` | Stars, forks, watchers, open issues |

**Limitations:** GitHub only retains 14 days of traffic data. The script saves
daily snapshots to `~/.local/share/analytics/` so you can build long-term trends.

### Cloudflare GraphQL Analytics

| Dataset | Resolution | Fields |
|---------|------------|--------|
| `httpRequests1dGroups` | Daily | Page views, unique visitors, requests, bandwidth, countries, paths, status codes, threats |
| `httpRequests1hGroups` | Hourly | Same as above but hourly granularity |

**Filters available:** date range, path prefix, country, status code, bot/cached flags.

## Output Format

```
═══ 1bit.systems Analytics ═══
Period: 2026-06-23 → 2026-07-07 (14 days)

┌─ GitHub: bong-water-water-bong/1bit ─────────────────────┐
│ Stars:     72 (3 new)                                     │
│ Forks:     14                                             │
│                                                           │
│ Traffic (14 days):                                        │
│   Clones:   3,647 total · 723 unique cloners              │
│   Views:    12,842 total · 3,201 unique visitors          │
│                                                           │
│ Top Referrers:                                            │
│   google.com             1,204                            │
│   github.com               892                            │
│   reddit.com               456                            │
│   twitter.com              234                            │
│                                                           │
│ Top Paths:                                                │
│   /bong-water-water-bong/1bit                  8,442      │
│   /bong-water-water-bong/1bit/wiki             2,100      │
│   /bong-water-water-bong/1bit/blob/main/...    1,300      │
└───────────────────────────────────────────────────────────┘

┌─ Cloudflare: 1bit.systems ───────────────────────────────┐
│ Page Views:       89,432                                  │
│ Unique Visitors:  24,187                                  │
│ Total Requests:   342,891                                 │
│ Bandwidth:        18.4 GB                                 │
│ Cached:           78.2%                                   │
│ Threats Blocked:  1,247                                   │
│                                                           │
│ Top Pages:                                                │
│   /                               34,221 (38.3%)          │
│   /blog/                          12,834 (14.4%)          │
│   /demo/                           7,421  (8.3%)          │
│   /benchmarks/                     5,892  (6.6%)          │
│   /docs/wiki/performance           4,331  (4.8%)          │
│                                                           │
│ Top Countries:                                            │
│   US         28,432 (31.8%)                               │
│   CN         12,847 (14.4%)                               │
│   DE          7,234  (8.1%)                               │
│   GB          5,891  (6.6%)                               │
│   JP          4,201  (4.7%)                               │
└───────────────────────────────────────────────────────────┘
```

## Files

| Path | Purpose |
|------|---------|
| `analytics.sh` | Main orchestrator — calls both GitHub + Cloudflare, formats output |
| `github-analytics.sh` | GitHub traffic only (clones, views, referrers, paths, stars) |
| `cf-analytics.sh` | Cloudflare GraphQL analytics only (views, bandwidth, geo, pages) |
| `badges.sh` | Update the badge JSON files in 1bit-site/ |

## Cache & History

Daily snapshots are saved to `~/.local/share/analytics/`:

```
~/.local/share/analytics/
├── github-clones-2026-07-07.json
├── github-views-2026-07-07.json
├── cf-summary-2026-07-07.json
├── history-clones.csv
├── history-views.csv
└── history-pageviews.csv
```

The CSV files accumulate daily so you can plot long-term trends with any
spreadsheet tool.

## Auto-update Badges

The `--badges` flag updates the badge JSON files in `~/1bit-site/`:

- `clones-stars.json` — clone count badge
- `daily-clones.json` — today's clone count
- `traffic-badge.json` — combined traffic badge
- `validated-badge.json` — ghost traffic validated flag

These badges are served on `1bit.systems` and embedded in the README.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `GITHUB_TOKEN not found` | Store in keyring: `bash ~/.pi/agent/skills/analytics/setup-keyring.sh` or `secret-tool store --label="GitHub analytics token" service github-analytics token` |
| `CLOUDFLARE_API_TOKEN not found` | Store in keyring: `secret-tool store --label="Cloudflare API token" service cloudflare-analytics token` |
| Cannot connect to keyring | Ensure daemon is running: `pgrep -a gnome-keyring` — if not, logout/login restarts it |
| Cloudflare returns "not found" | Verify zone ID: `secret-tool lookup service cloudflare-analytics zone` or set via setup script |
| GitHub returns 403 | Public repo traffic is free. Check token isn't expired. Classic token, no scopes. |
| No data for date range | GitHub only retains 14 days. Cloudflare retains 30 days on free plan. |

## Related Skills

- **jarvis** — JARVIS server at localhost:8080, can receive analytics push
- **benchmark-audit** — cross-reference performance numbers across the site
- **auto-commits** — commit badge updates after analytics refresh
