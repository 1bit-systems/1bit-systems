#!/usr/bin/env bash
# ppl-kv-dtype.sh — A/B perplexity bench for int8 vs fp16 KV cache.
#
# Spins up 1bit-halo-server-real twice on a scratch port: once with
# HALO_KV_DTYPE=i8 (default), once with HALO_KV_DTYPE=f16. Posts the same
# wikitext-103 slice to /ppl each time, captures PPL, emits delta.
#
# Quality cutover rule per project_i8_kv_unwired.md: int8 KV with per-(pos,
# kv_head) fp16 scales should hold PPL within ~0.5 % of fp16. A delta
# larger than HALO_KV_TOLERANCE (default 0.5 %, expressed as abs PPL) fails
# the bench.
#
# Does NOT touch whatever server is already running on :8180 (production
# serve port). Uses HALO_KV_PORT (default 8181) scratch.
#
# Exit codes:
#   0 — int8 PPL within tolerance of fp16
#   1 — int8 PPL drift exceeded tolerance
#   2 — harness / binary / dataset / port missing
#
# Output: /home/bcloud/claude output/ppl-kv-dtype.json

set -euo pipefail

OUT_DIR="/home/bcloud/claude output"
OUT_JSON="${OUT_DIR}/ppl-kv-dtype.json"

SERVER_BIN="${HALO_SERVER_BIN:-$HOME/.local/bin/1bit-halo-server-real}"
MODEL="${HALO_MODEL:-$HOME/1bit-halo-models/models/halo-1bit-2b.h1b}"
WIKITEXT="${WIKITEXT:-$HOME/1bit-halo-models/datasets/wikitext-103-test.txt}"
PORT="${HALO_KV_PORT:-8181}"
CHARS="${CHARS:-6000}"
STRIDE="${STRIDE:-1024}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TOLERANCE_PCT="${HALO_KV_TOLERANCE_PCT:-0.5}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-60}"
PPL_TIMEOUT="${PPL_TIMEOUT:-600}"

have() { command -v "$1" >/dev/null 2>&1; }
have jq   || { echo "ppl-kv-dtype: need jq"   >&2; exit 2; }
have curl || { echo "ppl-kv-dtype: need curl" >&2; exit 2; }

[[ -x "$SERVER_BIN" ]] || { echo "ppl-kv-dtype: server binary not found or not executable: $SERVER_BIN" >&2; exit 2; }
[[ -f "$MODEL" ]]      || { echo "ppl-kv-dtype: model missing: $MODEL" >&2; exit 2; }
[[ -f "$WIKITEXT" ]]   || { echo "ppl-kv-dtype: wikitext missing: $WIKITEXT" >&2; exit 2; }

if ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}$"; then
    echo "ppl-kv-dtype: port ${PORT} already bound — pick another via HALO_KV_PORT=" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"

TEXT=$(dd if="$WIKITEXT" bs=1 count="$CHARS" status=none)
[[ -n $TEXT ]] || { echo "ppl-kv-dtype: wikitext slice empty" >&2; exit 2; }

BODY=$(jq -n --arg text "$TEXT" \
             --argjson stride "$STRIDE" \
             --argjson max_tokens "$MAX_TOKENS" \
             '{text:$text, stride:$stride, max_tokens:$max_tokens}')

SERVER_PID=
cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        # Give it 3 s to drain then force.
        for _ in 1 2 3; do
            sleep 1
            kill -0 "$SERVER_PID" 2>/dev/null || break
        done
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

run_ppl_for_dtype() {
    local dtype="$1"
    local log_file
    log_file=$(mktemp -t "ppl-kv-${dtype}.log.XXXXXX")

    echo "─── $dtype ─────────────────────────────────────────────────"
    echo "launching server on :${PORT} with HALO_KV_DTYPE=${dtype} (log: ${log_file})"

    HALO_KV_DTYPE="$dtype" "$SERVER_BIN" \
        --bind "127.0.0.1:${PORT}" \
        --model "$MODEL" \
        >"$log_file" 2>&1 &
    SERVER_PID=$!

    local deadline=$(( $(date +%s) + BOOT_TIMEOUT ))
    while ! curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ppl-kv-dtype: server died during boot ($dtype) — tail of log:" >&2
            tail -40 "$log_file" >&2
            return 2
        fi
        if [[ $(date +%s) -ge $deadline ]]; then
            echo "ppl-kv-dtype: server boot timeout ($dtype, ${BOOT_TIMEOUT}s)" >&2
            return 2
        fi
        sleep 0.5
    done

    local raw
    raw=$(curl -fsS --max-time "$PPL_TIMEOUT" \
               -H 'Content-Type: application/json' \
               -d "$BODY" \
               "http://127.0.0.1:${PORT}/ppl") || {
        echo "ppl-kv-dtype: /ppl request failed ($dtype)" >&2
        return 2
    }

    local ppl
    ppl=$(echo "$raw" | jq -r '.ppl // empty')
    [[ -n "$ppl" ]] || {
        echo "ppl-kv-dtype: /ppl response missing .ppl ($dtype): $raw" >&2
        return 2
    }

    # Stop the server between runs so the next one comes up clean and
    # binds the port fresh.
    cleanup
    SERVER_PID=
    # Short settle window to let the port drop out of TIME_WAIT-ish state.
    sleep 1

    echo "$ppl"
}

PPL_I8=$(run_ppl_for_dtype i8)  || exit 2
PPL_F16=$(run_ppl_for_dtype f16) || exit 2

# Δ = |ppl_i8 − ppl_f16|, pct = Δ / ppl_f16 * 100.
DELTA=$(awk -v a="$PPL_I8" -v b="$PPL_F16" 'BEGIN { d=a-b; if (d<0) d=-d; printf "%.6f", d }')
DELTA_PCT=$(awk -v d="$DELTA" -v b="$PPL_F16" 'BEGIN { printf "%.4f", (d/b)*100 }')
TOL_ABS=$(awk -v b="$PPL_F16" -v t="$TOLERANCE_PCT" 'BEGIN { printf "%.6f", b*t/100 }')

STATUS="pass"
if awk -v d="$DELTA" -v t="$TOL_ABS" 'BEGIN { exit !(d>t) }'; then
    STATUS="fail"
fi

cat > "$OUT_JSON" <<JSON
{
  "harness": "1bit-halo-server-real /ppl",
  "timestamp": "$(date -Is)",
  "server_bin": "${SERVER_BIN}",
  "model": "${MODEL}",
  "dataset": "${WIKITEXT}",
  "chars": ${CHARS},
  "stride": ${STRIDE},
  "max_tokens": ${MAX_TOKENS},
  "port": ${PORT},
  "ppl_i8": ${PPL_I8},
  "ppl_f16": ${PPL_F16},
  "delta_abs": ${DELTA},
  "delta_pct": ${DELTA_PCT},
  "tolerance_pct": ${TOLERANCE_PCT},
  "tolerance_abs": ${TOL_ABS},
  "status": "${STATUS}"
}
JSON

echo
echo "ppl_i8       = ${PPL_I8}"
echo "ppl_f16      = ${PPL_F16}"
echo "Δ abs        = ${DELTA}"
echo "Δ %          = ${DELTA_PCT}%"
echo "tolerance %  = ${TOLERANCE_PCT}%"
echo "status       = ${STATUS}"
echo "json         = ${OUT_JSON}"

[[ "$STATUS" == "pass" ]] && exit 0 || exit 1
