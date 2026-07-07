#!/usr/bin/env bash
# analytics.sh — Full GitHub + Cloudflare analytics dashboard for 1bit.systems
# Usage: bash analytics.sh [--today] [--since DATE] [--until DATE] [--badges] [--github] [--cloudflare]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

DO_GITHUB=false
DO_CLOUDFLARE=false
DO_BADGES=false
PASSTHRU_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --github) DO_GITHUB=true; shift ;;
        --cloudflare) DO_CLOUDFLARE=true; shift ;;
        --badges) DO_BADGES=true; shift ;;
        *) PASSTHRU_ARGS+=("$1"); shift ;;
    esac
done

# Default: both
if ! $DO_GITHUB && ! $DO_CLOUDFLARE; then
    DO_GITHUB=true
    DO_CLOUDFLARE=true
fi

echo -e "${BOLD}${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║        1bit.systems Analytics Dashboard                  ║${NC}"
echo -e "${BOLD}${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo

# ---- GitHub ----
if $DO_GITHUB; then
    bash "$SCRIPT_DIR/github-analytics.sh" "${PASSTHRU_ARGS[@]}" 2>&1 || {
        echo -e "${YELLOW}⚠ GitHub analytics failed (check token). Continuing...${NC}"
    }
    echo
fi

# ---- Cloudflare ----
if $DO_CLOUDFLARE; then
    bash "$SCRIPT_DIR/cf-analytics.sh" "${PASSTHRU_ARGS[@]}" 2>&1 || {
        echo -e "${YELLOW}⚠ Cloudflare analytics failed (check token/zone). Continuing...${NC}"
    }
    echo
fi

# ---- Badges ----
if $DO_BADGES; then
    echo -e "${GREEN}Updating badges in ~/1bit-site/ ...${NC}"
    bash "$SCRIPT_DIR/badges.sh" 2>&1 || {
        echo -e "${RED}✗ Badge update failed${NC}"
    }
fi

echo -e "${CYAN}Done.${NC}"
