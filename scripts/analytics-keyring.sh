#!/usr/bin/env bash
# analytics-keyring.sh — Run the analytics workflow locally using keyring secrets
#
# Loads Cloudflare API token + account ID from GNOME Keyring (via secret-tool)
# and runs the full analytics collection pipeline, outputting site/analytics-data.json.
#
# Usage: ./scripts/analytics-keyring.sh

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*"; }

# ── load secrets from keyring ──────────────────────────────────────────────
log "Loading secrets from GNOME Keyring..."

CF_TOKEN=$(secret-tool lookup service cloudflare-analytics account api-token 2>/dev/null || true)
CF_ACCOUNT=$(secret-tool lookup service cloudflare-analytics account account-id 2>/dev/null || true)
GH_TOKEN=$(secret-tool lookup service github-analytics account pat 2>/dev/null || true)

if [ -z "$CF_TOKEN" ] || [ -z "$CF_ACCOUNT" ]; then
  err "Missing Cloudflare secrets in keyring."
  echo ""
  echo "  Store them first:"
  echo "    secret-tool store --label='Cloudflare API Token' service cloudflare-analytics account api-token"
  echo "    secret-tool store --label='Cloudflare Account ID' service cloudflare-analytics account account-id"
  echo ""
  echo "  See docs/analytics-keyring.md for details."
  exit 1
fi
log "Cloudflare secrets loaded ✓"
[ -n "$GH_TOKEN" ] && log "GitHub token loaded ✓" || warn "No GitHub token in keyring (traffic stats will be skipped)"

T="$CF_TOKEN"; A="$CF_ACCOUNT"
OUT="site/analytics-data.json"
REPO="bong-water-water-bong/1bit-systems"

# ── collect data ───────────────────────────────────────────────────────────
log "Collecting analytics data..."

echo "{" > "$OUT"

# 1. Cloudflare Pages Project Info
log "  Fetching Pages project info..."
echo '"pages":' >> "$OUT"
curl -sf -H "Authorization: Bearer $T" \
  "https://api.cloudflare.com/client/v4/accounts/$A/pages/projects/1bit-systems" | \
  jq '{ name: .result.name, domains: .result.domains, canonical_deployment: .result.canonical_deployment | { url, created_on, aliases } }' >> "$OUT" 2>/dev/null || echo '"error"' >> "$OUT"
echo ',' >> "$OUT"

# 2. Web Analytics
log "  Fetching Web Analytics..."
echo '"web_analytics":' >> "$OUT"
WA=$(curl -sf -H "Authorization: Bearer $T" \
  "https://api.cloudflare.com/client/v4/accounts/$A/analytics/websites" 2>/dev/null || echo '{"result":[]}')
echo "$WA" | jq '[.result[]? | select(.domain == "1bit.systems") | { domain, id, auto_install, token_public }]' >> "$OUT"
echo ',' >> "$OUT"

# 3. Zones
log "  Fetching zones..."
echo '"zones":' >> "$OUT"
curl -sf -H "Authorization: Bearer $T" \
  "https://api.cloudflare.com/client/v4/zones?per_page=50" 2>/dev/null | \
  jq '[.result[]? | { name, id, status, plan: .plan.name }]' >> "$OUT" || echo '[]' >> "$OUT"
echo ',' >> "$OUT"

# 4. Deployments (last 10)
log "  Fetching deployments..."
echo '"deployments":' >> "$OUT"
curl -sf -H "Authorization: Bearer $T" \
  "https://api.cloudflare.com/client/v4/accounts/$A/pages/projects/1bit-systems/deployments?per_page=10" 2>/dev/null | \
  jq '[.result[]? | { created_on: .created_on, url: .url, aliases: .aliases, latest_stage: ([.stages[]? | select(.ended_on == null) | .name] | first // "done"), env: .env }]' >> "$OUT" || echo '[]' >> "$OUT"
echo ',' >> "$OUT"

# 5. Account project summary
log "  Fetching account projects..."
echo '"account_projects":' >> "$OUT"
curl -sf -H "Authorization: Bearer $T" \
  "https://api.cloudflare.com/client/v4/accounts/$A/pages/projects" 2>/dev/null | \
  jq '[.result[]? | { name, subdomain }]' >> "$OUT" || echo '[]' >> "$OUT"

# ── GitHub Traffic (if token available) ────────────────────────────────────
if [ -n "$GH_TOKEN" ]; then
  log "  Fetching GitHub traffic..."
  echo ',' >> "$OUT"
  echo '"github_traffic":' >> "$OUT"

  # We'll build a sub-object
  GH_DATA=$(mktemp)
  echo "{" > "$GH_DATA"

  CLONES=$(curl -sf -H "Authorization: Bearer $GH_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/traffic/clones" 2>/dev/null || echo '{}')
  echo '"clones":' >> "$GH_DATA"
  echo "$CLONES" | jq '{ count: .count, uniques: .uniques, last_24h: .clones[-1] }' >> "$GH_DATA"
  echo ',' >> "$GH_DATA"

  VIEWS=$(curl -sf -H "Authorization: Bearer $GH_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/traffic/views" 2>/dev/null || echo '{}')
  echo '"views":' >> "$GH_DATA"
  echo "$VIEWS" | jq '{ count: .count, uniques: .uniques }' >> "$GH_DATA"
  echo ',' >> "$GH_DATA"

  REFERRERS=$(curl -sf -H "Authorization: Bearer $GH_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/traffic/popular/referrers" 2>/dev/null || echo '[]')
  echo '"referrers":' >> "$GH_DATA"
  echo "$REFERRERS" | jq '.' >> "$GH_DATA"
  echo ',' >> "$GH_DATA"

  PATHS=$(curl -sf -H "Authorization: Bearer $GH_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO/traffic/popular/paths" 2>/dev/null || echo '[]')
  echo '"paths":' >> "$GH_DATA"
  echo "$PATHS" | jq '.' >> "$GH_DATA"
  echo ',' >> "$GH_DATA"

  REPO_INFO=$(curl -sf -H "Authorization: Bearer $GH_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/$REPO" 2>/dev/null || echo '{}')
  echo '"repo":' >> "$GH_DATA"
  echo "$REPO_INFO" | jq '{ stars: .stargazers_count, forks: .forks_count, open_issues: .open_issues_count }' >> "$GH_DATA"

  echo "}" >> "$GH_DATA"
  cat "$GH_DATA" >> "$OUT"
  rm -f "$GH_DATA"
fi

echo "" >> "$OUT"
echo "}" >> "$OUT"

log "Analytics data written to $OUT"

# ── print summary ──────────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo "  Analytics Summary"
echo "=============================================="
cat "$OUT" | jq '{
  pages_domains: .pages.domains,
  web_analytics_sites: (.web_analytics | length),
  zones: (.zones | length),
  deployments: (.deployments | length),
  account_projects: (.account_projects | length)
}' 2>/dev/null || true

if [ -n "$GH_TOKEN" ]; then
  echo ""
  echo "GitHub Traffic:"
  cat "$OUT" | jq '.github_traffic | {
    clones: .clones.count,
    unique_cloners: .clones.uniques,
    views: .views.count,
    unique_visitors: .views.uniques,
    top_referrer: .referrers[0].referrer,
    top_path: .paths[0].path,
    stars: .repo.stars,
    forks: .repo.forks
  }' 2>/dev/null || true
fi

echo ""
log "Done."
