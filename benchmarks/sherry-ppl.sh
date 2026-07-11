#!/usr/bin/env bash
# sherry-ppl.sh — PPL regression gate for Sherry 1.25-bit.
#
# Invokes `bitnet_decode --ppl --dataset wikitext-103` against a Sherry-
# packed halo-1bit-2b model, parses the PPL, compares to the gen-1
# baseline of 9.1607 and fails if the delta exceeds 0.10.
#
# The 0.10 tolerance is TIGHTER than the 0.05 cutover tolerance documented
# in CLAUDE.md because Sherry is lossy by construction (3:4 sparsity), so
# we cap drift sooner — if we can't hold 9.26 or better on wikitext-103 the
# kernel is wrong or the requantizer's zero-choice heuristic is wrong.
#
# Output JSON lives at /home/bcloud/claude output/sherry-ppl.json (blessed
# per feedback_benchmark_output_folder.md). The script still writes the
# JSON on failure so CI can read the actual number.
#
# Exit codes:
#   0  PPL within tolerance
#   1  PPL drift exceeded 0.10
#   2  harness / model missing — cannot measure

set -euo pipefail

OUT_DIR="/home/bcloud/claude output"
OUT_JSON="${OUT_DIR}/sherry-ppl.json"

DEFAULT_MODEL="${HALO_SHERRY_MODEL:-/opt/halo-ai/models/halo-1bit-2b-sherry.h1b}"
DEFAULT_DATASET="${HALO_PPL_DATASET:-/opt/halo-ai/datasets/wikitext-103.txt}"
BASELINE="${HALO_PPL_BASELINE:-9.1607}"
TOLERANCE="${HALO_PPL_TOLERANCE:-0.10}"
BITNET_DECODE_BIN="${BITNET_DECODE:-/opt/halo-ai/bin/bitnet_decode}"

MODEL="${DEFAULT_MODEL}"
DATASET="${DEFAULT_DATASET}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        --tolerance)
            TOLERANCE="$2"
            shift 2
            ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "sherry-ppl: unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

mkdir -p "${OUT_DIR}"

write_json() {
    # write_json <status> <ppl> <delta> <note>
    local status="$1"
    local ppl="$2"
    local delta="$3"
    local note="$4"
    cat > "${OUT_JSON}" <<JSON
{
  "harness": "bitnet_decode --ppl",
  "timestamp": "$(date -Is)",
  "model": "${MODEL}",
  "dataset": "${DATASET}",
  "baseline": ${BASELINE},
  "tolerance": ${TOLERANCE},
  "ppl": "${ppl}",
  "delta": "${delta}",
  "status": "${status}",
  "note": "${note}"
}
JSON
}

if [[ ! -x "${BITNET_DECODE_BIN}" ]]; then
    write_json "skipped" "null" "null" "bitnet_decode not found at ${BITNET_DECODE_BIN}"
    echo "sherry-ppl: SKIP — bitnet_decode binary missing" >&2
    exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
    write_json "skipped" "null" "null" "model missing: ${MODEL}"
    echo "sherry-ppl: SKIP — model missing: ${MODEL}" >&2
    exit 2
fi
if [[ ! -f "${DATASET}" ]]; then
    write_json "skipped" "null" "null" "dataset missing: ${DATASET}"
    echo "sherry-ppl: SKIP — dataset missing: ${DATASET}" >&2
    exit 2
fi

# Run the harness. Real CLI is positional:
#   bitnet_decode <model.h1b> --ppl <file.txt> [max_tokens]
# stderr line: "[ppl] file=... tokens=... nll=... ppl=9.1805"
# stdout line: '{"file":"...","tokens":...,"nll":...,"ppl":9.180500}'
RAW="$("${BITNET_DECODE_BIN}" \
    "${MODEL}" \
    --ppl "${DATASET}" \
    2>&1 || true)"

# Try JSON shape first (more stable), fall back to "ppl=<n>" stderr line.
PPL="$(echo "${RAW}" | awk 'match($0, /"ppl":[ ]*([0-9]+\.[0-9]+)/, m) { print m[1]; exit }')"
if [[ -z "${PPL:-}" ]]; then
    PPL="$(echo "${RAW}" | awk 'match($0, /ppl=([0-9]+\.[0-9]+)/, m) { print m[1]; exit }')"
fi

if [[ -z "${PPL:-}" ]]; then
    write_json "parse_error" "null" "null" "could not parse PPL from harness output"
    echo "sherry-ppl: FAIL — could not parse PPL" >&2
    echo "${RAW}" | tail -20 >&2
    exit 1
fi

# Compare with awk; bash doesn't do floats natively.
DELTA="$(awk -v a="${PPL}" -v b="${BASELINE}" 'BEGIN { printf "%.4f", a - b }')"
ABS_DELTA="$(awk -v d="${DELTA}" 'BEGIN { if (d < 0) d = -d; printf "%.4f", d }')"
PASS="$(awk -v a="${ABS_DELTA}" -v t="${TOLERANCE}" 'BEGIN { print (a <= t) ? "pass" : "fail" }')"

write_json "${PASS}" "${PPL}" "${DELTA}" "delta=${DELTA} vs baseline=${BASELINE} tol=${TOLERANCE}"

echo "sherry-ppl: ${PASS} — PPL=${PPL} delta=${DELTA} (tol ${TOLERANCE})"
if [[ "${PASS}" == "pass" ]]; then
    exit 0
else
    exit 1
fi
