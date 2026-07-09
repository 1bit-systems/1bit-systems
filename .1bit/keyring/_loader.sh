#!/usr/bin/env bash
# _loader.sh — Source this in scripts to access the 1bit keyring
#
# Usage:
#   source ~/.1bit/keyring/_loader.sh
#   TOKEN=$(get_secret github token)
#   export GITHUB_TOKEN=$(get_secret github token)
#
# Priority: keyring → env var → config file

KR_DIR="${KR_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
KR_FALLBACK_DIR="${HOME}/.config/1bit/secrets"

# Ensure fallback dir exists
[[ -d "$KR_FALLBACK_DIR" ]] || mkdir -p "$KR_FALLBACK_DIR"

# ── get_secret: retrieve a secret by namespace and key ──
# Usage: get_secret <namespace> <key>
# Returns the secret value on stdout, or empty string + non-zero exit
get_secret() {
    local ns="${1:-}"; local key="${2:-}"
    [[ -z "$ns" || -z "$key" ]] && return 1

    # 1. Try keyring (secret-tool) — try all attribute patterns
    if command -v secret-tool &>/dev/null; then
        local val=""
        val=$(secret-tool lookup service "$ns" key "$key" 2>/dev/null) || true
        if [[ -z "$val" ]]; then
            val=$(secret-tool lookup service "$ns" account "$key" 2>/dev/null) || true
        fi
        if [[ -z "$val" ]]; then
            val=$(secret-tool lookup service "$ns" "$key" 2>/dev/null) || true
        fi
        if [[ -n "$val" ]]; then
            echo "$val"
            return 0
        fi
    fi

    # 2. Try env var (namespace_key, uppercased)
    local env_name="${ns}_${key}"
    env_name="${env_name//-/_}"
    env_name="${env_name^^}"
    if [[ -n "${!env_name:-}" ]]; then
        echo "${!env_name}"
        return 0
    fi

    # 3. Try flat file fallback
    local fallback="$KR_FALLBACK_DIR/${ns}-${key}"
    if [[ -f "$fallback" ]]; then
        cat "$fallback"
        return 0
    fi

    return 1
}

# ── has_secret: check if a secret exists ──
has_secret() {
    local ns="${1:-}"; local key="${2:-}"
    get_secret "$ns" "$key" >/dev/null 2>&1
}

# ── require_secret: get a secret or fail with message ──
require_secret() {
    local ns="${1:-}"; local key="${2:-}"
    local val
    val=$(get_secret "$ns" "$key") || {
        echo "[keyring] ERROR: ${ns}/${key} not found. Set it with: kr set $ns $key" >&2
        return 1
    }
    echo "$val"
}

# ── Convenience aliases ──
get_gh_token()    { get_secret github token; }
get_cf_token()    { get_secret cloudflare token; }
get_cf_zone()     { get_secret cloudflare zone; }
get_cf_account()  { get_secret cloudflare account-id; }
get_ai_key()      { get_secret ai "${1:-}"; }
get_hf_token()    { get_secret huggingface token; }

# ── Export as functions for subshell use ──
export -f get_secret has_secret require_secret
export -f get_gh_token get_cf_token get_cf_zone get_cf_account get_ai_key get_hf_token
