#!/usr/bin/env bash
# setup-keyring.sh — Store GitHub + Cloudflare tokens in GNOME Keyring
# Usage: bash setup-keyring.sh
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_keyring() {
    if ! command -v secret-tool &>/dev/null; then
        echo -e "${RED}✗ secret-tool not found. Install: sudo apt install libsecret-tools${NC}"
        exit 1
    fi
    if ! pgrep -x gnome-keyring-daemon &>/dev/null; then
        echo -e "${YELLOW}⚠ gnome-keyring-daemon not running. Starting it...${NC}"
        gnome-keyring-daemon --start --components=secrets 2>/dev/null || true
    fi
    echo -e "${GREEN}✓ Keyring available${NC}\n"
}

store_secret() {
    local service="$1"
    local key="$2"
    local label="$3"

    # Check if already stored
    if secret-tool lookup service "$service" "$key" &>/dev/null; then
        local existing
        existing=$(secret-tool lookup service "$service" "$key")
        echo -e "${YELLOW}⚠ $label already stored (${existing:0:8}...). Overwrite? [y/N] ${NC}"
        read -r answer
        if [[ ! "$answer" =~ ^[Yy]$ ]]; then
            echo "  Skipped."
            return
        fi
        secret-tool clear service "$service" "$key" 2>/dev/null || true
    fi

    echo -n "  Enter $label: "
    read -rs value
    echo

    if [[ -z "$value" ]]; then
        echo -e "${YELLOW}  Empty value, skipped.${NC}"
        return
    fi

    echo -n "$value" | secret-tool store --label="$label" service "$service" "$key"
    echo -e "${GREEN}  ✓ Stored${NC}"
}

show_status() {
    echo -e "\n${GREEN}═══ Keyring Status ═══${NC}"
    if secret-tool lookup service github-analytics token &>/dev/null; then
        local gh_token
        gh_token=$(secret-tool lookup service github-analytics token)
        echo -e "  GitHub:      ${GREEN}✓${NC} ${gh_token:0:8}..."
    else
        echo -e "  GitHub:      ${RED}✗ not stored${NC}"
    fi

    if secret-tool lookup service cloudflare-analytics token &>/dev/null; then
        local cf_token
        cf_token=$(secret-tool lookup service cloudflare-analytics token)
        echo -e "  Cloudflare:  ${GREEN}✓${NC} ${cf_token:0:8}..."
    else
        echo -e "  Cloudflare:  ${RED}✗ not stored${NC}"
    fi

    if secret-tool lookup service cloudflare-analytics zone &>/dev/null; then
        local cf_zone
        cf_zone=$(secret-tool lookup service cloudflare-analytics zone)
        echo -e "  CF Zone ID: ${GREEN}✓${NC} $cf_zone"
    else
        echo -e "  CF Zone ID: ${YELLOW}⚠ not stored (will auto-discover)${NC}"
    fi
}

main() {
    echo -e "${GREEN}═══ Analytics Keyring Setup ═══${NC}\n"
    check_keyring

    echo -e "${YELLOW}GitHub Token${NC}"
    echo "  Go to: https://github.com/settings/tokens"
    echo "  Create a classic token — no scopes needed for public repos.\n"
    store_secret "github-analytics" "token" "GitHub token"

    echo -e "\n${YELLOW}Cloudflare API Token${NC}"
    echo "  Go to: https://dash.cloudflare.com/profile/api-tokens"
    echo "  Create token with Analytics:Read permission.\n"
    store_secret "cloudflare-analytics" "token" "Cloudflare API token"

    echo -e "\n${YELLOW}Cloudflare Zone ID (optional)${NC}"
    echo "  Find in Cloudflare dashboard → 1bit.systems → Overview → Zone ID (right sidebar)"
    echo "  Leave empty to auto-discover (slower, needs zone:read on token).\n"
    store_secret "cloudflare-analytics" "zone" "Cloudflare zone ID"

    show_status
    echo -e "\n${GREEN}Done. Run: bash ~/.pi/agent/skills/analytics/analytics.sh${NC}"
}

main
