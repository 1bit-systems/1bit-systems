#!/usr/bin/env bash
#
# bench-npu-ioctl-budget.sh — regression guard for the FastFlowLM NPU lane
#
# Runs strace -f -e trace=ioctl on `flm serve <small-model>` during a single
# OpenAI-compat decode call and asserts the per-token ioctl count stays below
# a threshold. Detects regressions where FLM (or amdxdna) starts firing more
# DRM round-trips per token than the baseline.
#
# Baseline measured 2026-04-28 (FLM 0.9.39, amdxdna 0.6, FW 1.1.2.65, AIE2P):
#   ~96 ioctls/token on qwen3:0.6b — see benchmarks/RESULTS-flm-strace-2026-04-28.md
#
# After the upstream BO-pool fix lands (per-stream free-list of CMD + small-DEV
# BOs), this should collapse to ~5-10 ioctls/token — bump THRESHOLD down then
# and re-add the pre-fix value as a comment for the historical record.
#
# Usage:
#   benchmarks/bench-npu-ioctl-budget.sh [model]      # default qwen3:0.6b
#   benchmarks/bench-npu-ioctl-budget.sh qwen3:1.7b
#
# Exit:
#   0 — per-token ioctl count under THRESHOLD
#   1 — over THRESHOLD (regression)
#   2 — environment problem (no NPU, no flm, no strace, etc.)

set -euo pipefail

# --- Config ----------------------------------------------------------------

MODEL="${1:-qwen3:0.6b}"
FLM_PORT="${FLM_PORT:-8042}"          # avoid colliding with lemond's flm proxy on :8001
MAX_TOKENS="${MAX_TOKENS:-20}"
THRESHOLD="${THRESHOLD:-250}"          # ioctls/token (incl. prefill); baseline ~215, fail above 250
WARN_THRESHOLD="${WARN_THRESHOLD:-200}"  # warn if creeping above 200/token
WORKDIR="${WORKDIR:-/tmp/npu-ioctl-budget}"

# --- Output helpers --------------------------------------------------------

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
say()  { printf '%b▸%b %s\n' "$CYAN" "$NC" "$*"; }
ok()   { printf '%b✓%b %s\n' "$GREEN" "$NC" "$*"; }
warn() { printf '%b!%b %s\n' "$YELLOW" "$NC" "$*"; }
fail() { printf '%b✗%b %s\n' "$RED" "$NC" "$*" >&2; }
die_env() { fail "$*"; exit 2; }

# --- Pre-flight ------------------------------------------------------------

require() { command -v "$1" >/dev/null 2>&1 || die_env "$1 not on PATH"; }
require flm
require strace
require jq
require curl
require pgrep
[[ -e /dev/accel/accel0 ]] || die_env "/dev/accel/accel0 missing — no XDNA NPU on this box"

mkdir -p "$WORKDIR"

# --- State management ------------------------------------------------------

LEMOND_WAS_ACTIVE=0
FLM_PID=""
STRACE_PID=""

cleanup() {
    [[ -n "$STRACE_PID" ]] && sudo kill "$STRACE_PID" 2>/dev/null || true
    [[ -n "$FLM_PID" ]] && kill "$FLM_PID" 2>/dev/null || true
    sleep 1
    [[ -n "$FLM_PID" ]] && kill -9 "$FLM_PID" 2>/dev/null || true
    if (( LEMOND_WAS_ACTIVE )); then
        say "restoring lemond.service (was active before probe)"
        systemctl --user start lemond 2>&1 | head -2
    fi
}
trap cleanup EXIT

# --- Main ------------------------------------------------------------------

if systemctl --user is-active lemond >/dev/null 2>&1; then
    LEMOND_WAS_ACTIVE=1
    say "stopping lemond.service to free the NPU for the probe"
    systemctl --user stop lemond
    sleep 2
fi

# Some boxes have a stale flm process — kill it first
pkill -f "^flm serve" 2>/dev/null || true
sleep 1

say "starting flm serve $MODEL on :$FLM_PORT"
flm serve "$MODEL" --port "$FLM_PORT" >"$WORKDIR/flm-server.log" 2>&1 &
FLM_PID=$!
disown 2>/dev/null || true

# Wait for it to come up
for _ in $(seq 1 60); do
    if curl -sS -m 1 "http://127.0.0.1:$FLM_PORT/v1/models" >/dev/null 2>&1; then break; fi
    sleep 1
done
curl -sS -m 1 "http://127.0.0.1:$FLM_PORT/v1/models" >/dev/null 2>&1 \
    || die_env "flm serve never became reachable on :$FLM_PORT (see $WORKDIR/flm-server.log)"
ok "flm serve up — PID $FLM_PID"

say "warm-up call (load weights into NPU)"
curl -sS -m 60 "http://127.0.0.1:$FLM_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$MODEL" '{model:$m,messages:[{role:"user",content:"ok"}],max_tokens:5,temperature:0}')" \
    > "$WORKDIR/warmup.json"
ok "warm"

say "attaching strace -f -e trace=ioctl -c to flm PID $FLM_PID"
sudo strace -f -p "$FLM_PID" -e trace=ioctl -c -o "$WORKDIR/strace.txt" >/dev/null 2>&1 &
STRACE_PID=$!
sleep 1

say "firing $MAX_TOKENS-token decode at temperature=0"
RESP=$(curl -sS -m 120 "http://127.0.0.1:$FLM_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg m "$MODEL" --argjson n "$MAX_TOKENS" \
        '{model:$m,messages:[{role:"user",content:"Write five short words separated by spaces."}],max_tokens:$n,temperature:0}')")
echo "$RESP" > "$WORKDIR/decode.json"

# Detach strace cleanly so it dumps the summary
sudo kill -INT "$STRACE_PID" 2>/dev/null || true
sleep 2
STRACE_PID=""

# --- Parse + assert --------------------------------------------------------

if [[ ! -s "$WORKDIR/strace.txt" ]]; then
    die_env "strace produced no output — check ptrace policy (kernel.yama.ptrace_scope) or sudo"
fi

# strace -c output: a single line for ioctl with the count in column 4 (or last)
TOTAL_IOCTLS=$(awk '/ioctl/ {print $4}' "$WORKDIR/strace.txt" | head -1)
TOTAL_IOCTLS=${TOTAL_IOCTLS:-0}

# Tokens actually decoded (FLM honors max_tokens but may stop earlier on EOS)
DECODED=$(echo "$RESP" | jq -r '.usage.completion_tokens // .timings.predicted_n // 0')
if [[ -z "$DECODED" || "$DECODED" == "0" || "$DECODED" == "null" ]]; then
    die_env "couldn't parse decoded-token count from response (see $WORKDIR/decode.json)"
fi

PER_TOKEN=$(( TOTAL_IOCTLS / DECODED ))

echo
printf '%b═══ NPU ioctl budget ═══════════════════════════════════════%b\n' "$BOLD" "$NC"
printf '  Model:           %s\n' "$MODEL"
printf '  Decoded tokens:  %s\n' "$DECODED"
printf '  Total ioctls:    %s\n' "$TOTAL_IOCTLS"
printf '  Per token:       %b%s%b   (threshold %s, warn at %s)\n' \
    "$BOLD" "$PER_TOKEN" "$NC" "$THRESHOLD" "$WARN_THRESHOLD"
printf '  Decode tps:      %s\n' "$(echo "$RESP" | jq -r '.usage.decoding_speed_tps // .timings.predicted_per_second // "?"' | awk '{printf "%.1f\n", $1}')"
printf '%b════════════════════════════════════════════════════════════%b\n' "$BOLD" "$NC"
echo

# --- Pass / fail -----------------------------------------------------------

if (( PER_TOKEN > THRESHOLD )); then
    fail "REGRESSION: $PER_TOKEN ioctls/token > $THRESHOLD threshold"
    fail "raw strace: $WORKDIR/strace.txt"
    exit 1
fi

if (( PER_TOKEN > WARN_THRESHOLD )); then
    warn "$PER_TOKEN ioctls/token is above warn threshold $WARN_THRESHOLD (still under fail $THRESHOLD)"
fi

ok "$PER_TOKEN ioctls/token — under fail threshold $THRESHOLD"
exit 0
