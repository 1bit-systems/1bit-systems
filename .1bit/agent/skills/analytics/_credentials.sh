#!/usr/bin/env bash
# _credentials.sh — Shared credential loader for analytics scripts
# Tries keyring first, then environment variables.
# Sources: secret-tool (GNOME Keyring) → env vars
#
# Usage: source this file, then use get_gh_token, get_cf_token, get_cf_zone

get_gh_token() {
    # 1. Keyring
    if command -v secret-tool &>/dev/null; then
        local tok
        tok=$(secret-tool lookup service github-analytics token 2>/dev/null) || true
        if [[ -n "$tok" ]]; then
            echo "$tok"
            return 0
        fi
    fi
    # 2. Env var
    if [[ -n "${GITHUB_TOKEN:-}" ]]; then
        echo "$GITHUB_TOKEN"
        return 0
    fi
    # 3. gh CLI (already authenticated)
    if command -v gh &>/dev/null && gh auth status &>/dev/null 2>&1; then
        gh auth token 2>/dev/null && return 0
    fi
    return 1
}

# Check if gh CLI can be used directly (preferred — no token needed)
use_gh_cli() {
    command -v gh &>/dev/null && gh auth status &>/dev/null 2>&1
}

get_cf_token() {
    if command -v secret-tool &>/dev/null; then
        local tok
        tok=$(secret-tool lookup service cloudflare-analytics key token 2>/dev/null) || true
        if [[ -n "$tok" ]]; then
            echo "$tok"
            return 0
        fi
    fi
    if [[ -n "${CLOUDFLARE_API_TOKEN:-}" ]]; then
        echo "$CLOUDFLARE_API_TOKEN"
        return 0
    fi
    return 1
}

get_cf_zone() {
    if command -v secret-tool &>/dev/null; then
        local zone
        zone=$(secret-tool lookup service cloudflare-analytics key zone 2>/dev/null) || true
        if [[ -n "$zone" ]]; then
            echo "$zone"
            return 0
        fi
    fi
    if [[ -n "${CLOUDFLARE_ZONE_ID:-}" ]]; then
        echo "$CLOUDFLARE_ZONE_ID"
        return 0
    fi
    return 1
}
