#!/usr/bin/env bash
# check-remote-main.sh — Poll origin/main for new commits and signal agents
#
# Run this periodically (e.g., via cron, or after `git fetch`) to detect
# pushes to main that happened on the remote. Checks multiple repos.
#
# Usage:
#   ./check-remote-main.sh                    # Check all known repos
#   ./check-remote-main.sh /path/to/repo      # Check specific repo
#   ./check-remote-main.sh --init             # Initialize tracking files
#
# Setup as a cron job (every 5 minutes):
#   */5 * * * * ${HOME}/scripts/check-remote-main.sh

set -euo pipefail

SIGNAL_SCRIPT="${HOME}/scripts/signal-agent-awareness.sh"
RECORD_SCRIPT="${HOME}/scripts/record-agent-change.sh"

# Repos to monitor — add more as needed
REPOS="${1:-${HOME}/1bit-systems ${HOME}/colibri}"

for REPO in $REPOS; do
    if [ ! -d "$REPO/.git" ] && [ ! -f "$REPO/.git" ]; then
        continue
    fi

    TRACKING_FILE="${HOME}/.1bit/agent/.remote-head-$(echo "$REPO" | tr '/' '_')"

    cd "$REPO"

    # Fetch latest remote info (lightweight, no merge)
    git fetch origin main --depth=10 2>/dev/null || true

    REMOTE_HEAD="$(git rev-parse origin/main 2>/dev/null || echo '')"
    LOCAL_HEAD="$(git rev-parse main 2>/dev/null || echo '')"

    if [ -z "$REMOTE_HEAD" ]; then
        continue
    fi

    PREVIOUS_HEAD=""
    if [ -f "$TRACKING_FILE" ]; then
        PREVIOUS_HEAD="$(cat "$TRACKING_FILE")"
    fi

    # Check if there's something new
    if [ "$REMOTE_HEAD" != "$PREVIOUS_HEAD" ]; then
        NEW_COMMITS="$(git log --oneline "${PREVIOUS_HEAD:-$LOCAL_HEAD}..origin/main" 2>/dev/null | head -5 || true)"

        if [ -n "$NEW_COMMITS" ]; then
            echo "[check-remote] $REPO: new commits on origin/main detected:"
            echo "$NEW_COMMITS"

            # Record in awareness file
            if [ -x "$RECORD_SCRIPT" ]; then
                "$RECORD_SCRIPT" commit main
            fi

            # Signal running agents
            FIRST_LINE="$(echo "$NEW_COMMITS" | head -1)"
            if [ -x "$SIGNAL_SCRIPT" ]; then
                "$SIGNAL_SCRIPT" "$(basename $REPO): $FIRST_LINE"
            fi
        fi

        # Update tracking
        echo "$REMOTE_HEAD" > "$TRACKING_FILE"
    fi
done
