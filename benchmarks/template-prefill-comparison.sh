#!/usr/bin/env bash
# template-prefill-comparison.sh — A/B end-to-end tok/s bench for
# HALO_CHAT_TEMPLATE=llama3 vs HALO_CHAT_TEMPLATE=short.
#
# PURPOSE
#
#   Measure the HTTP-end-to-end tok/s gap between the default Llama-3
#   chat template (9-ish extra prefill tokens per single-turn request)
#   and the minimal `short` template (0 extra framing tokens after BOS).
#
#   Documented target: on halo-1bit-2b.h1b running at ~80.8 tok/s at
#   the kernel level, the `llama3` template prefill stalls the first
#   decode step and the observed HTTP tok/s drops to ~65.7 at
#   64-token generation. The `short` template should recover most of
#   that gap (expected improvement ~10-15 tok/s at 64-tok ctx,
#   shrinking as context grows and prefill becomes a smaller fraction
#   of wall-clock).
#
# HARNESS DESIGN
#
#   Same pattern as ppl-kv-dtype.sh: spin up 1bit-server-real twice on
#   a scratch port (default 8182), once per HALO_CHAT_TEMPLATE value.
#   Fire N requests at /v1/chat/completions, take the p50 of reported
#   server-side tok/s plus the wall-clock tok/s measured by this
#   script. Emit JSON to /home/bcloud/claude output/.
#
#   This script is DOCUMENTATION and is intentionally not run as part
#   of CI — it needs a real GPU, a loaded .h1b model, and the
#   real-backend feature compiled in. Run by hand on strixhalo when
#   you want to verify a template change.
#
# RUN
#
#   benchmarks/template-prefill-comparison.sh
#
# OVERRIDES (env)
#
#   HALO_SERVER_BIN   path to the real-backend binary (default
#                     $HOME/.local/bin/1bit-halo-server-real)
#   HALO_MODEL        .h1b model path
#   TEMPLATE_PORT     scratch port (default 8182)
#   PROMPT            text content sent as the single user message.
#                     Default: a short greeting — the shorter the
#                     prompt, the larger the relative prefill overhead
#                     and the bigger the `short` vs `llama3` delta.
#   MAX_TOKENS        generation length (default 64 to match the
#                     diagnosis; also try 256 / 1024 to see the delta
#                     shrink).
#   REQ_COUNT         number of samples per template (default 20)
#
# EXIT
#
#   0 — ran both arms, wrote JSON, `short` tok/s was >= `llama3`
#   1 — `short` tok/s was lower than `llama3` (regression — investigate)
#   2 — harness / binary / dataset / port missing

set -euo pipefail

OUT_DIR="/home/bcloud/claude output"
OUT_JSON="${OUT_DIR}/template-prefill-comparison.json"

SERVER_BIN="${HALO_SERVER_BIN:-$HOME/.local/bin/1bit-halo-server-real}"
MODEL="${HALO_MODEL:-$HOME/1bit-halo-models/models/halo-1bit-2b.h1b}"
PORT="${TEMPLATE_PORT:-8182}"
PROMPT="${PROMPT:-Hi, what's up?}"
MAX_TOKENS="${MAX_TOKENS:-64}"
REQ_COUNT="${REQ_COUNT:-20}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-60}"

have() { command -v "$1" >/dev/null 2>&1; }
have jq   || { echo "template-bench: need jq"   >&2; exit 2; }
have curl || { echo "template-bench: need curl" >&2; exit 2; }

[[ -x "$SERVER_BIN" ]] || { echo "template-bench: missing $SERVER_BIN" >&2; exit 2; }
[[ -r "$MODEL"      ]] || { echo "template-bench: missing $MODEL"      >&2; exit 2; }

mkdir -p "$OUT_DIR"

# Fire one request, return tok/s = (completion_tokens / wall_seconds).
# Relies on the /v1/chat/completions non-streaming envelope.
fire_one() {
  local start end body
  start=$(date +%s.%N)
  body=$(curl -fsS --max-time 120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"halo-1bit-2b\",\"stream\":false,\"max_tokens\":${MAX_TOKENS},\"messages\":[{\"role\":\"user\",\"content\":$(jq -Rn --arg s "$PROMPT" '$s')}]}") || return 1
  end=$(date +%s.%N)
  local ctoks
  ctoks=$(echo "$body" | jq -r '.usage.completion_tokens // 0')
  awk -v c="$ctoks" -v s="$start" -v e="$end" 'BEGIN { if (e - s > 0) printf "%.4f\n", c / (e - s); else print "0.0000" }'
}

run_arm() {
  local template="$1"
  local log=/tmp/template-bench-${template}.log
  echo "=== arm: HALO_CHAT_TEMPLATE=${template} ==="

  HALO_CHAT_TEMPLATE="$template" "$SERVER_BIN" \
    --bind "127.0.0.1:${PORT}" \
    --model "$MODEL" \
    --rate-limit-rpm 0 \
    >"$log" 2>&1 &
  local pid=$!
  trap 'kill -TERM '"$pid"' 2>/dev/null; wait '"$pid"' 2>/dev/null || true' EXIT INT TERM

  # Wait for /healthz
  local t=0
  until curl -fsS "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; do
    sleep 1; t=$((t+1))
    [[ $t -ge $BOOT_TIMEOUT ]] && { echo "server didn't come up" >&2; kill -TERM "$pid" 2>/dev/null; return 2; }
  done

  # Warm-up (one request, discarded).
  fire_one >/dev/null 2>&1 || true

  # Collect.
  local samples=()
  for _ in $(seq 1 "$REQ_COUNT"); do
    samples+=("$(fire_one || echo 0.0)")
  done

  kill -TERM "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  trap - EXIT INT TERM

  # p50 of samples.
  printf '%s\n' "${samples[@]}" | sort -n | awk 'BEGIN{c=0}{a[c++]=$1}END{if(c%2==1)print a[int(c/2)]; else printf "%.4f\n",(a[c/2-1]+a[c/2])/2}'
}

llama3_p50=$(run_arm llama3)
short_p50=$(run_arm  short)

# delta (%) = (short - llama3) / llama3 * 100
delta_pct=$(awk -v l="$llama3_p50" -v s="$short_p50" 'BEGIN { if (l > 0) printf "%.2f\n", (s - l) / l * 100; else print "0.00" }')

cat >"$OUT_JSON" <<EOF
{
  "prompt": $(jq -Rn --arg s "$PROMPT" '$s'),
  "max_tokens": $MAX_TOKENS,
  "samples_per_arm": $REQ_COUNT,
  "llama3_p50_tokps": $llama3_p50,
  "short_p50_tokps":  $short_p50,
  "delta_pct": $delta_pct
}
EOF

echo "wrote $OUT_JSON"
cat "$OUT_JSON"

# Expected: short_p50 >= llama3_p50 for short single-turn prompts.
awk -v l="$llama3_p50" -v s="$short_p50" 'BEGIN { exit !(s >= l) }'
