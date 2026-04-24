#!/usr/bin/env bash
# greedy-fast-path.sh — A/B decode-throughput bench for the dedicated
# `forward_token_greedy` fast path vs the legacy `forward_token` +
# `HALO_SKIP_LOGITS_COPY` skip-copy path vs the full copy+reconcile
# path.
#
# What the dedicated greedy path skips per decode step:
#   * The 512 KB fp32 logits D→H memcpy (vocab × 4 B = 128256 × 4 B).
#   * The O(vocab) host-argmax reconcile scan (~50-80 µs on Zen5).
#   * The `Vec<f32>` resize dance that keeps the scratch buffer alive.
#
# Expected delta on halo-1bit-2b @ 64-tok decode (projection only — the
# bench here is the harness, actually running it needs ROCm + model):
#
#   full-copy+reconcile (HALO_SKIP_LOGITS_COPY=0) :  baseline, say 66 tok/s
#   skip-copy (HALO_SKIP_LOGITS_COPY=1, empty Vec):  +~1-3 tok/s
#   forward_token_greedy (this feature)           :  +~2-5 tok/s total vs
#                                                    baseline
#
# The spread narrows at long context (L=1024+) because the 512 KB copy
# is a fixed per-step cost but per-step walltime grows with L — the
# skip looks bigger at short context where each step is already fast.
#
# Usage (once the server binary supports selecting between these
# paths, or when invoked directly against the backend from a standalone
# bench driver):
#
#   benchmarks/greedy-fast-path.sh
#
# Env knobs:
#   HALO_SERVER_BIN   default: ~/.local/bin/1bit-halo-server-real
#   HALO_MODEL        default: ~/1bit-halo-models/models/halo-1bit-2b.h1b
#   PROMPT            default: "The capital of France is"
#   MAX_TOKENS        default: 64       — the L in "64-tok decode"
#   ITERS             default: 5        — runs per mode; median reported
#   PORT              default: 8182     — scratch port, not 8180 prod
#
# Exit codes:
#   0 — ran all three modes, emitted JSON
#   2 — harness precondition missing (binary, model, port, jq)
#
# Output: /home/bcloud/claude output/greedy-fast-path.json
#
# NOTE: the greedy path is selected automatically by the router when
# `temperature == 0.0` and no streaming callback is set — see
# crates/1bit-router/src/lib.rs. To A/B against the skip-copy path,
# flip `HALO_SKIP_LOGITS_COPY` below; the router still enters
# `forward_token_greedy` because the temperature gate dominates, so
# this bench actually A/Bs against a temperature=0.01 sampled run
# (one step through the sampler for a single-logit dispatch). The
# sampled mode is included as the backward-compat control.

set -euo pipefail

OUT_DIR="/home/bcloud/claude output"
OUT_JSON="${OUT_DIR}/greedy-fast-path.json"

SERVER_BIN="${HALO_SERVER_BIN:-$HOME/.local/bin/1bit-halo-server-real}"
MODEL="${HALO_MODEL:-$HOME/1bit-halo-models/models/halo-1bit-2b.h1b}"
PROMPT="${PROMPT:-The capital of France is}"
MAX_TOKENS="${MAX_TOKENS:-64}"
ITERS="${ITERS:-5}"
PORT="${PORT:-8182}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-60}"

have() { command -v "$1" >/dev/null 2>&1; }
have jq   || { echo "greedy-fast-path: need jq"   >&2; exit 2; }
have curl || { echo "greedy-fast-path: need curl" >&2; exit 2; }

[[ -x "$SERVER_BIN" ]] || {
    echo "greedy-fast-path: server binary not found or not executable: $SERVER_BIN" >&2
    exit 2
}
[[ -f "$MODEL" ]] || {
    echo "greedy-fast-path: model missing: $MODEL" >&2
    exit 2
}

if ss -tln 2>/dev/null | awk '{print $4}' | grep -q ":${PORT}$"; then
    echo "greedy-fast-path: port ${PORT} already bound — pick another via PORT=" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"

# Median of $1..$N numeric args to stdout.
median() {
    printf '%s\n' "$@" | sort -n | awk '
        { a[NR] = $1 }
        END {
            if (NR == 0) { print 0; exit }
            if (NR % 2) { print a[(NR + 1) / 2] }
            else        { printf "%.6f\n", (a[NR/2] + a[NR/2 + 1]) / 2 }
        }'
}

# Boot a one-shot server with the env of the calling shell + whatever
# overrides the caller wired in via `env`. Caller is responsible for
# cleanup on exit.
boot_server() {
    local log="$1"
    HALO_MODEL_PATH="$MODEL" \
        "$SERVER_BIN" --port "$PORT" >"$log" 2>&1 &
    SERVER_PID=$!
    local deadline=$((SECONDS + BOOT_TIMEOUT))
    until curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )) || ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "greedy-fast-path: server failed to come up within ${BOOT_TIMEOUT}s" >&2
            echo "--- server log ---" >&2
            cat "$log" >&2 || true
            exit 2
        fi
        sleep 0.5
    done
}

# Run one /v1/chat/completions request, report tok/s = completion_tokens / elapsed_ms * 1000.
one_run() {
    local temperature="$1"
    local t_start t_end elapsed_ms body tok
    t_start=$(date +%s%N)
    body=$(curl -fsS "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(jq -cn \
            --arg prompt "$PROMPT" \
            --argjson max "$MAX_TOKENS" \
            --argjson temp "$temperature" \
            '{ messages: [{ role: "user", content: $prompt }],
               max_tokens: $max, temperature: $temp, stream: false }')")
    t_end=$(date +%s%N)
    elapsed_ms=$(( (t_end - t_start) / 1000000 ))
    tok=$(printf '%s' "$body" | jq '.usage.completion_tokens // 0')
    if (( tok <= 0 || elapsed_ms <= 0 )); then
        echo "0"
        return
    fi
    # tok/s
    awk -v t="$tok" -v ms="$elapsed_ms" 'BEGIN { printf "%.3f", t * 1000 / ms }'
}

run_mode() {
    local label="$1" tps rates=()
    echo "==> $label" >&2
    for i in $(seq 1 "$ITERS"); do
        tps=$(one_run "${TEMP:-0.0}")
        echo "    iter $i: $tps tok/s" >&2
        rates+=("$tps")
    done
    median "${rates[@]}"
}

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ---- Mode A: dedicated greedy path (forward_token_greedy). ----
# Selected automatically by the router when temperature=0.0.
# HALO_SKIP_LOGITS_COPY is irrelevant here (greedy path never
# touches the flag), pinned to 0 to make the separation from Mode B
# unambiguous.
LOG_A=$(mktemp)
HALO_SKIP_LOGITS_COPY=0 boot_server "$LOG_A"
GREEDY_TPS=$(TEMP=0.0 run_mode "forward_token_greedy (temp=0.0)")
cleanup
unset SERVER_PID

# ---- Mode B: legacy skip-copy path (forward_token + empty Vec + env flag). ----
# Router still enters forward_token_greedy on temp=0.0 now, so this
# mode is exercised at temp=0.01 — a single-token sampled run that
# goes through the legacy `forward_token` path.
LOG_B=$(mktemp)
HALO_SKIP_LOGITS_COPY=1 boot_server "$LOG_B"
SKIP_TPS=$(TEMP=0.01 run_mode "forward_token + HALO_SKIP_LOGITS_COPY=1 (temp=0.01)")
cleanup
unset SERVER_PID

# ---- Mode C: full-copy + host reconcile (legacy baseline). ----
LOG_C=$(mktemp)
HALO_SKIP_LOGITS_COPY=0 boot_server "$LOG_C"
FULL_TPS=$(TEMP=0.01 run_mode "forward_token + HALO_SKIP_LOGITS_COPY=0 (temp=0.01)")
cleanup
unset SERVER_PID

# ---- Emit JSON summary. ----
jq -n \
    --arg prompt "$PROMPT" \
    --argjson max "$MAX_TOKENS" \
    --argjson iters "$ITERS" \
    --arg greedy "$GREEDY_TPS" \
    --arg skip "$SKIP_TPS" \
    --arg full "$FULL_TPS" \
    '{ prompt: $prompt,
       max_tokens: $max,
       iters: $iters,
       modes: {
         "forward_token_greedy": ($greedy | tonumber),
         "forward_token_skip_copy": ($skip | tonumber),
         "forward_token_full_copy": ($full | tonumber)
       },
       delta_tok_s: {
         greedy_vs_full: (($greedy | tonumber) - ($full | tonumber)),
         greedy_vs_skip: (($greedy | tonumber) - ($skip | tonumber)),
         skip_vs_full:   (($skip   | tonumber) - ($full | tonumber))
       }
     }' | tee "$OUT_JSON"

echo "wrote $OUT_JSON" >&2
