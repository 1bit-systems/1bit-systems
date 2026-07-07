#!/usr/bin/env bash
# github-analytics.sh — Pull GitHub traffic data for a repo
# Default: bong-water-water-bong/1bit-systems (primary discovery surface)
# Usage: bash github-analytics.sh [--repo owner/repo] [--today] [--since DATE] [--until DATE]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_credentials.sh"

OWNER="bong-water-water-bong"
REPO="1bit-systems"
CACHE_DIR="${HOME}/.local/share/analytics"
mkdir -p "$CACHE_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ---- Date handling ----
TODAY=$(date +%Y-%m-%d)
SINCE=""
UNTIL=""
SHORT_MODE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo) OWNER="${2%/*}"; REPO="${2#*/}"; shift 2 ;;
        --today) SHORT_MODE=true; SINCE="$TODAY"; UNTIL="$TODAY"; shift ;;
        --since) SINCE="$2"; shift 2 ;;
        --until) UNTIL="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "$SINCE" ]]; then SINCE=$(date -d "14 days ago" +%Y-%m-%d); fi
if [[ -z "$UNTIL" ]]; then UNTIL="$TODAY"; fi

# ---- Auth & API caller ----
# Prefer gh CLI (already authenticated, zero config). Fall back to curl + token.
if command -v gh &>/dev/null && gh auth status &>/dev/null 2>&1; then
    api_call() {
        gh api "$1" --jq '.' 2>/dev/null
    }
else
    GH_TOKEN=$(get_gh_token) || { echo -e "${RED}✗ GitHub not authenticated. Run: gh auth login${NC}"; exit 1; }
    api_call() {
        curl -sf -H "Authorization: token $GH_TOKEN" \
             -H "Accept: application/vnd.github+json" \
             "https://api.github.com$1" 2>/dev/null
    }
fi

# ---- Fetch ----
echo -e "${BOLD}${CYAN}═══ GitHub Analytics: $OWNER/$REPO ═══${NC}"
echo -e "Period: $SINCE → $UNTIL\n"

# Stars + forks + watchers
repo_info=$(api_call "/repos/$OWNER/$REPO")
STARS=$(echo "$repo_info" | jq -r '.stargazers_count // 0')
FORKS=$(echo "$repo_info" | jq -r '.forks_count // 0')
WATCHERS=$(echo "$repo_info" | jq -r '.subscribers_count // 0')
OPEN_ISSUES=$(echo "$repo_info" | jq -r '.open_issues_count // 0')

# Traffic: clones
clones_json=$(api_call "/repos/$OWNER/$REPO/traffic/clones")
TOTAL_CLONES=$(echo "$clones_json" | jq -r '.count // 0')
UNIQUE_CLONES=$(echo "$clones_json" | jq -r '.uniques // 0')

# Traffic: views
views_json=$(api_call "/repos/$OWNER/$REPO/traffic/views")
TOTAL_VIEWS=$(echo "$views_json" | jq -r '.count // 0')
UNIQUE_VIEWS=$(echo "$views_json" | jq -r '.uniques // 0')

# Top referrers
referrers_json=$(api_call "/repos/$OWNER/$REPO/traffic/popular/referrers")

# Top paths
paths_json=$(api_call "/repos/$OWNER/$REPO/traffic/popular/paths")

# ---- Display ----
echo -e "${GREEN}┌─ Repository Stats ───────────────────────────────────┐${NC}"
printf "  %-20s ${BOLD}%s${NC}\n" "Stars:" "$STARS"
printf "  %-20s %s\n" "Forks:" "$FORKS"
printf "  %-20s %s\n" "Watchers:" "$WATCHERS"
printf "  %-20s %s\n" "Open Issues:" "$OPEN_ISSUES"
echo

echo -e "${GREEN}┌─ Traffic (14-day window) ────────────────────────────┐${NC}"
printf "  %-20s ${BOLD}%'d${NC} total · ${BOLD}%'d${NC} unique\n" "Clones:" "$TOTAL_CLONES" "$UNIQUE_CLONES"
printf "  %-20s ${BOLD}%'d${NC} total · ${BOLD}%'d${NC} unique\n" "Page Views:" "$TOTAL_VIEWS" "$UNIQUE_VIEWS"
echo

# Daily breakdown
if ! $SHORT_MODE; then
    echo -e "${GREEN}┌─ Daily Clones ───────────────────────────────────────┐${NC}"
    if [[ -n "$clones_json" && "$clones_json" != "null" ]]; then
        echo "$clones_json" | jq -r '
            .clones // [] | reverse | .[] |
            "  \(.timestamp[0:10])  ░  \(.count) clones (\(.uniques) unique)"
        '
    else
        echo "  (no data)"
    fi

    echo
    echo -e "${GREEN}┌─ Daily Views ────────────────────────────────────────┐${NC}"
    if [[ -n "$views_json" && "$views_json" != "null" ]]; then
        echo "$views_json" | jq -r '
            .views // [] | reverse | .[] |
            "  \(.timestamp[0:10])  ░  \(.count) views (\(.uniques) unique)"
        '
    else
        echo "  (no data)"
    fi
fi

# Top referrers
echo
echo -e "${GREEN}┌─ Top Referrers ──────────────────────────────────────┐${NC}"
if [[ -n "$referrers_json" && "$referrers_json" != "null" ]]; then
    echo "$referrers_json" | jq -r '
        (.[] // []) | 
        "  \(.referrer // "unknown")  \(.count)  (\(.uniques) unique)"
    ' 2>/dev/null || echo "  (no data)"
else
    echo "  (no data)"
fi

# Top paths
echo
echo -e "${GREEN}┌─ Top Paths ──────────────────────────────────────────┐${NC}"
if [[ -n "$paths_json" && "$paths_json" != "null" ]]; then
    echo "$paths_json" | jq -r '
        (.[] // []) | 
        "  \(.path)  \(.count)  (\(.uniques) unique)"
    ' 2>/dev/null || echo "  (no data)"
else
    echo "  (no data)"
fi

# ---- Cache ----
HISTORY_CLONES="$CACHE_DIR/history-clones-$REPO.csv"
HISTORY_VIEWS="$CACHE_DIR/history-views-$REPO.csv"
HISTORY_STARS="$CACHE_DIR/history-stars-$REPO.csv"
echo "$TODAY,$TOTAL_CLONES,$UNIQUE_CLONES" >> "$HISTORY_CLONES"
echo "$TODAY,$TOTAL_VIEWS,$UNIQUE_VIEWS" >> "$HISTORY_VIEWS"
echo "$TODAY,$STARS,$FORKS" >> "$HISTORY_STARS"
echo "$clones_json" > "$CACHE_DIR/github-clones-$REPO-$TODAY.json"
echo "$views_json" > "$CACHE_DIR/github-views-$REPO-$TODAY.json"

echo -e "\n${CYAN}History saved to: $CACHE_DIR/${NC}"
