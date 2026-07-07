#!/usr/bin/env bash
# badges.sh — Update badge JSON files in ~/1bit-site/ from latest analytics
# Usage: bash badges.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_credentials.sh"

SITE_DIR="${HOME}/1bit-site"
CACHE_DIR="${HOME}/.local/share/analytics"

# GitHub clone counts for badges — uses gh CLI (preferred) or token
REPO="bong-water-water-bong/1bit-systems"

if command -v gh &>/dev/null && gh auth status &>/dev/null 2>&1; then
    TOTAL_CLONES=$(gh api "repos/$REPO/traffic/clones" --jq '.count // 0')
    UNIQUE_CLONES=$(gh api "repos/$REPO/traffic/clones" --jq '.uniques // 0')
    STARS=$(gh api "repos/$REPO" --jq '.stargazers_count // 0')
    TODAY=$(date +%Y-%m-%d)
    TODAY_CLONES=$(gh api "repos/$REPO/traffic/clones" --jq ".clones[] | select(.timestamp[:10] == \"$TODAY\") | .count // 0")
else
    GH_TOKEN=$(get_gh_token) || { echo "GitHub token not available — skipping GitHub badges"; exit 0; }
    TOTAL_CLONES=$(curl -sf -H "Authorization: token $GH_TOKEN" \
        "https://api.github.com/repos/$REPO/traffic/clones" | jq -r '.count // 0')
    UNIQUE_CLONES=$(curl -sf -H "Authorization: token $GH_TOKEN" \
        "https://api.github.com/repos/$REPO/traffic/clones" | jq -r '.uniques // 0')
    STARS=$(curl -sf -H "Authorization: token $GH_TOKEN" \
        "https://api.github.com/repos/$REPO" | jq -r '.stargazers_count // 0')
    TODAY=$(date +%Y-%m-%d)
    TODAY_CLONES=$(curl -sf -H "Authorization: token $GH_TOKEN" \
        "https://api.github.com/repos/$REPO/traffic/clones" | jq -r ".clones[] | select(.timestamp[:10] == \"$TODAY\") | .count // 0")
fi

# Traffic badge (clones)
cat > "$SITE_DIR/traffic-badge.json" << EOF
{
  "schemaVersion": 1,
  "label": "clones",
  "message": "$TOTAL_CLONES ($UNIQUE_CLONES unique)",
  "color": "brightgreen",
  "cacheSeconds": 86400
}
EOF

# Clones:stars badge
cat > "$SITE_DIR/clones-stars.json" << EOF
{
  "schemaVersion": 1,
  "label": "clones:stars",
  "message": "$TOTAL_CLONES:$STARS ($UNIQUE_CLONES:$STARS unique)",
  "color": "brightgreen",
  "cacheSeconds": 86400
}
EOF

# Daily clones badge
cat > "$SITE_DIR/daily-clones.json" << EOF
{
  "schemaVersion": 1,
  "label": "clones today",
  "message": "$TODAY_CLONES",
  "color": "brightgreen",
  "cacheSeconds": 86400
}
EOF

# Validated badge — marks that these numbers come from live analytics
cat > "$SITE_DIR/validated-badge.json" << EOF
{
  "schemaVersion": 1,
  "label": "analytics",
  "message": "live $(date +%Y-%m-%d)",
  "color": "brightgreen",
  "cacheSeconds": 86400
}
EOF

# Cloudflare page views badge (if available)
CF_TOKEN=$(get_cf_token) || true
if [[ -n "$CF_TOKEN" ]]; then
    ZONE_ID=$(get_cf_zone 2>/dev/null) || true
    if [[ -z "$ZONE_ID" ]]; then
        ZONE_ID=$(curl -sf -H "Authorization: Bearer $CF_TOKEN" \
            "https://api.cloudflare.com/client/v4/zones?name=1bit.systems" \
            | jq -r '.result[0].id // empty')
    fi
    if [[ -n "$ZONE_ID" ]]; then
        PV_QUERY=$(cat <<GQL
{"query":"{viewer{zones(filter:{zoneTag:\"$ZONE_ID\"}){httpRequests1dGroups(filter:{date_geq:\"$TODAY\",date_leq:\"$TODAY\"}){sum{pageViews}}}}}"}
GQL
)
        PV_TODAY=$(curl -sf -X POST "https://api.cloudflare.com/client/v4/graphql" \
            -H "Authorization: Bearer $CF_TOKEN" \
            -H "Content-Type: application/json" \
            -d "$PV_QUERY" \
            | jq -r '.data.viewer.zones[0].httpRequests1dGroups[0].sum.pageViews // 0')

        cat > "$SITE_DIR/pageviews-badge.json" << EOF
{
  "schemaVersion": 1,
  "label": "page views today",
  "message": "$PV_TODAY",
  "color": "brightgreen",
  "cacheSeconds": 86400
}
EOF
    fi
fi

echo "Updated badges:"
ls -la "$SITE_DIR"/*-badge.json "$SITE_DIR"/clones-stars.json "$SITE_DIR"/daily-clones.json "$SITE_DIR"/validated-badge.json 2>/dev/null
