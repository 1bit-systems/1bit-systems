#!/usr/bin/env bash
# Monitor for OPTC CRTC hangs during a combined workload soak.
#
# Usage:
#   scripts/optc_soak_2hr.sh [seconds]
#
# Optional workload hooks:
#   OPTC_AI_CMD='...'   command to run while monitoring, e.g. an inference loop
#   OPTC_GPU_CMD='...'  command to run while monitoring, e.g. a graphics load

set -euo pipefail

DURATION="${1:-7200}"
CADENCE="${OPTC_CADENCE:-10}"
OUT_DIR="${OPTC_OUT_DIR:-$HOME/claude output/optc-soak-$(date +%Y%m%d-%H%M%S)}"
PATTERN='REG_WAIT.*optc35_disable_crtc|optc35_disable_crtc'

mkdir -p "$OUT_DIR"

cleanup() {
    [[ -n "${AI_PID:-}" ]] && kill "$AI_PID" 2>/dev/null || true
    [[ -n "${GPU_PID:-}" ]] && kill "$GPU_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "output: $OUT_DIR"
journalctl -k -b --no-pager >"$OUT_DIR/dmesg-pre.log" 2>/dev/null || true
baseline=$(grep -Eci "$PATTERN" "$OUT_DIR/dmesg-pre.log" || true)

if [[ -n "${OPTC_AI_CMD:-}" ]]; then
    bash -lc "$OPTC_AI_CMD" >"$OUT_DIR/ai.log" 2>&1 &
    AI_PID=$!
fi

if [[ -n "${OPTC_GPU_CMD:-}" ]]; then
    bash -lc "$OPTC_GPU_CMD" >"$OUT_DIR/gpu-worker.log" 2>&1 &
    GPU_PID=$!
fi

{
    printf 'ts,optc_count,optc_delta'
    for f in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
        [[ -e "$f" ]] || continue
        card=${f#/sys/class/drm/}
        card=${card%%/*}
        printf ',%s_power_dpm' "$card"
    done
    printf '\n'
} >"$OUT_DIR/gpu.csv"

end=$((SECONDS + DURATION))
while (( SECONDS < end )); do
    journalctl -k -b --no-pager >"$OUT_DIR/dmesg-current.log" 2>/dev/null || true
    count=$(grep -Eci "$PATTERN" "$OUT_DIR/dmesg-current.log" || true)
    {
        printf '%s,%s,%s' "$(date -Is)" "$count" "$((count - baseline))"
        for f in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
            [[ -e "$f" ]] || continue
            printf ',%s' "$(cat "$f")"
        done
        printf '\n'
    } >>"$OUT_DIR/gpu.csv"
    sleep "$CADENCE"
done

journalctl -k -b --no-pager >"$OUT_DIR/dmesg-post.log" 2>/dev/null || true
final=$(grep -Eci "$PATTERN" "$OUT_DIR/dmesg-post.log" || true)
delta=$((final - baseline))

{
    echo "duration_seconds=$DURATION"
    echo "cadence_seconds=$CADENCE"
    echo "baseline_optc_count=$baseline"
    echo "final_optc_count=$final"
    echo "new_optc_events=$delta"
    if (( delta == 0 )); then
        echo "verdict=PASS"
    else
        echo "verdict=FAIL"
    fi
} | tee "$OUT_DIR/VERDICT.txt"

(( delta == 0 ))
