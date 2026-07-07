#!/usr/bin/env bash
# cf-analytics.sh — Pull Cloudflare GraphQL Analytics for 1bit.systems
# Usage: bash cf-analytics.sh [--today] [--since DATE] [--until DATE]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_credentials.sh"

DOMAIN="1bit.systems"
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
        --today) SHORT_MODE=true; SINCE="$TODAY"; UNTIL="$TODAY"; shift ;;
        --since) SINCE="$2"; shift 2 ;;
        --until) UNTIL="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "$SINCE" ]]; then SINCE=$(date -d "14 days ago" +%Y-%m-%d); fi
if [[ -z "$UNTIL" ]]; then UNTIL="$TODAY"; fi

# ---- Auth ----
CF_TOKEN=$(get_cf_token) || { echo -e "${RED}✗ CLOUDFLARE_API_TOKEN not found. Run: bash $SCRIPT_DIR/setup-keyring.sh${NC}"; exit 1; }
ZONE_ID=$(get_cf_zone) || { echo -e "${RED}✗ Zone ID not found. Run: bash $SCRIPT_DIR/setup-keyring.sh${NC}"; exit 1; }

cf_gql() {
    local query="$1"
    curl -sf -X POST "https://api.cloudflare.com/client/v4/graphql" \
        -H "Authorization: Bearer $CF_TOKEN" \
        -H "Content-Type: application/json" \
        -d "$query" 2>/dev/null
}

# ---- Fetch daily summary ----
SUMMARY_QUERY=$(jq -n --arg zone "$ZONE_ID" --arg since "$SINCE" --arg until "$UNTIL" '{
  query: "{
    viewer {
      zones(filter: {zoneTag: \"\($zone)\"}) {
        httpRequests1dGroups(
          filter: {date_geq: \"\($since)\", date_leq: \"\($until)\"}
          orderBy: [date_ASC]
          limit: 100
        ) {
          dimensions { date }
          sum { pageViews requests bytes cachedRequests cachedBytes threats }
          uniq { uniques }
        }
      }
    }
  }"
}')

SUMMARY=$(cf_gql "$SUMMARY_QUERY")

# Parse totals
TOTAL_PV=$(echo "$SUMMARY" | jq -r '[.data.viewer.zones[0].httpRequests1dGroups[].sum.pageViews // 0] | add // 0')
TOTAL_REQUESTS=$(echo "$SUMMARY" | jq -r '[.data.viewer.zones[0].httpRequests1dGroups[].sum.requests // 0] | add // 0')
TOTAL_BYTES=$(echo "$SUMMARY" | jq -r '[.data.viewer.zones[0].httpRequests1dGroups[].sum.bytes // 0] | add // 0')
TOTAL_CACHED=$(echo "$SUMMARY" | jq -r '[.data.viewer.zones[0].httpRequests1dGroups[].sum.cachedRequests // 0] | add // 0')
TOTAL_THREATS=$(echo "$SUMMARY" | jq -r '[.data.viewer.zones[0].httpRequests1dGroups[].sum.threats // 0] | add // 0')
TOTAL_UNIQUES=$(echo "$SUMMARY" | jq -r '[.data.viewer.zones[0].httpRequests1dGroups[].uniq.uniques // 0] | add // 0')

format_bytes() {
    local bytes=$1
    if (( bytes > 1073741824 )); then
        awk "BEGIN {printf \"%.1f GB\", $bytes/1073741824}"
    elif (( bytes > 1048576 )); then
        awk "BEGIN {printf \"%.1f MB\", $bytes/1048576}"
    elif (( bytes > 1024 )); then
        awk "BEGIN {printf \"%.1f KB\", $bytes/1024}"
    else
        echo "${bytes} B"
    fi
}

if (( TOTAL_REQUESTS > 0 )); then
    CACHE_PCT=$(awk "BEGIN {printf \"%.1f\", ($TOTAL_CACHED/$TOTAL_REQUESTS)*100}")
else
    CACHE_PCT="0"
fi

# ---- Display ----
echo -e "${BOLD}${CYAN}═══ Cloudflare Analytics: $DOMAIN ═══${NC}"
echo -e "Period: $SINCE → $UNTIL\n"

echo -e "${GREEN}┌─ Summary ────────────────────────────────────────────┐${NC}"
printf "  %-22s ${BOLD}%'d${NC}\n" "Page Views:" "$TOTAL_PV"
printf "  %-22s %'d\n" "Unique Visitors:" "$TOTAL_UNIQUES"
printf "  %-22s %'d\n" "Total Requests:" "$TOTAL_REQUESTS"
printf "  %-22s %s\n" "Bandwidth:" "$(format_bytes $TOTAL_BYTES)"
printf "  %-22s %s%%\n" "Cached:" "$CACHE_PCT"
printf "  %-22s %'d\n" "Threats Blocked:" "$TOTAL_THREATS"
echo

# Daily breakdown
if ! $SHORT_MODE; then
    echo -e "${GREEN}┌─ Daily ──────────────────────────────────────────────┐${NC}"
    echo "$SUMMARY" | jq -r '
        .data.viewer.zones[0].httpRequests1dGroups[]? // empty |
        "  \(.dimensions.date)  ░  \(.sum.pageViews) pv · \(.uniq.uniques) uniq · \(.sum.requests) req · \(.sum.bytes | tonumber / 1048576 * 100 | round / 100) MB"
    '
    echo
fi

# ---- Status codes ----
STATUS_QUERY=$(jq -n --arg zone "$ZONE_ID" --arg since "$SINCE" --arg until "$UNTIL" '{
  query: "{
    viewer {
      zones(filter: {zoneTag: \"\($zone)\"}) {
        httpRequests1dGroups(
          filter: {date_geq: \"\($since)\", date_leq: \"\($until)\"}
          limit: 100
        ) {
          sum { responseStatusMap { edgeResponseStatus requests } }
        }
      }
    }
  }"
}')
STATUS_DATA=$(cf_gql "$STATUS_QUERY")
echo -e "${GREEN}┌─ Status Codes ───────────────────────────────────────┐${NC}"
echo "$STATUS_DATA" | jq -r '
    [.data.viewer.zones[0].httpRequests1dGroups[].sum.responseStatusMap[]?]
    | group_by(.edgeResponseStatus) 
    | map({status: .[0].edgeResponseStatus, req: (map(.requests) | add)})
    | sort_by(-.req) | .[0:8]
    | .[] | "  \(.status)  \(.req) requests"
' 2>/dev/null || echo "  (no data)"
echo

# ---- Countries ----
COUNTRY_QUERY=$(jq -n --arg zone "$ZONE_ID" --arg since "$SINCE" --arg until "$UNTIL" '{
  query: "{
    viewer {
      zones(filter: {zoneTag: \"\($zone)\"}) {
        httpRequests1dGroups(
          filter: {date_geq: \"\($since)\", date_leq: \"\($until)\"}
          limit: 100
        ) {
          sum { countryMap { clientCountryName requests bytes threats } }
        }
      }
    }
  }"
}')
COUNTRY_DATA=$(cf_gql "$COUNTRY_QUERY")
echo -e "${GREEN}┌─ Top Countries (by requests) ────────────────────────┐${NC}"
echo "$COUNTRY_DATA" | jq -r '
    [.data.viewer.zones[0].httpRequests1dGroups[].sum.countryMap[]?]
    | group_by(.clientCountryName) 
    | map({country: .[0].clientCountryName, req: (map(.requests) | add), bytes: (map(.bytes) | add)})
    | sort_by(-.req) | .[0:10]
    | .[] | "  \(.country)  \(.req) req  \((.bytes/1048576*100|round/100) // 0) MB"
' 2>/dev/null || echo "  (no data)"
echo

# ---- Browsers ----
BROWSER_QUERY=$(jq -n --arg zone "$ZONE_ID" --arg since "$SINCE" --arg until "$UNTIL" '{
  query: "{
    viewer {
      zones(filter: {zoneTag: \"\($zone)\"}) {
        httpRequests1dGroups(
          filter: {date_geq: \"\($since)\", date_leq: \"\($until)\"}
          limit: 100
        ) {
          sum { browserMap { uaBrowserFamily pageViews } }
        }
      }
    }
  }"
}')
BROWSER_DATA=$(cf_gql "$BROWSER_QUERY")
echo -e "${GREEN}┌─ Top Browsers (by page views) ───────────────────────┐${NC}"
echo "$BROWSER_DATA" | jq -r '
    [.data.viewer.zones[0].httpRequests1dGroups[].sum.browserMap[]?]
    | group_by(.uaBrowserFamily) 
    | map({browser: .[0].uaBrowserFamily, pv: (map(.pageViews) | add)})
    | sort_by(-.pv) | .[0:8]
    | .[] | "  \(.browser // "Unknown")  \(.pv) pv"
' 2>/dev/null || echo "  (no data)"

# ---- Cache ----
HISTORY_PV="$CACHE_DIR/history-pageviews.csv"
echo "$TODAY,$TOTAL_PV,$TOTAL_UNIQUES,$TOTAL_REQUESTS,$TOTAL_BYTES" >> "$HISTORY_PV"
echo "$SUMMARY" > "$CACHE_DIR/cf-summary-$TODAY.json"

echo -e "\n${CYAN}History saved to: $CACHE_DIR/${NC}"
