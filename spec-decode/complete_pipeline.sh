#!/usr/bin/env bash
# Complete spec decode pipeline: monitor training, build, benchmark
set -ueo pipefail

TRAINING_LOG=/tmp/training_output.log
CKPT=/home/bcloud/spec-decode/checkpoints/eagle3_draft.bin
BUILD_DIR=/home/bcloud/spec-decode/build
SPEC_BIN=$BUILD_DIR/npu_spec_decode
RESULTS_FILE=/tmp/spec_decode_results.txt

notify() {
    local msg="$1"
    local title="${2:-1bit Spec Decode}"
    echo "[notify] $(date) $msg"
    notify-send -u critical "$title" "$msg" 2>/dev/null || true
    wall "$(date) 🔔 1bit SpecDecode: $msg" 2>/dev/null || true
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ⚡ $msg"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

notify "Pipeline started — monitoring training every 30s"

echo "[pipeline] $(date) Starting completion pipeline"
echo "[pipeline] Monitoring training at $TRAINING_LOG"

# Wait for training to finish
PREV_SIZE=0
STALL_COUNT=0
while true; do
  # Check if process exists
  PROC_COUNT=$(ps aux | grep train_from_cache | grep -v grep | wc -l)
  if [ "$PROC_COUNT" -eq 0 ]; then
    notify "Training process exited — checking checkpoint"
    break
  fi

  # Check if log stopped growing (stall detection)
  if [ -f "$TRAINING_LOG" ]; then
    CUR_SIZE=$(stat --format="%s" "$TRAINING_LOG" 2>/dev/null || echo 0)
    if [ "$CUR_SIZE" -eq "$PREV_SIZE" ] && [ "$PREV_SIZE" -gt 0 ]; then
      STALL_COUNT=$((STALL_COUNT + 1))
    else
      STALL_COUNT=0
    fi
    PREV_SIZE=$CUR_SIZE
  fi

  # If stalled for 10 checks (5 min), assume done/crashed
  if [ "$STALL_COUNT" -ge 10 ]; then
    notify "Training stalled (no output for 5 min)"
    break
  fi

  # Show latest loss
  LOSS=$(tail -3 "$TRAINING_LOG" 2>/dev/null | grep -oP 'loss=[\d.]+' | tail -1 || echo "?")
  echo "[pipeline] $(date) Training in progress: loss=$LOSS (check every 30s)"
  sleep 30
done

notify "Training complete or stopped"

# Verify checkpoint
if [ -f "$CKPT" ]; then
  CKPT_SIZE=$(stat --format="%s" "$CKPT")
  CKPT_TIME=$(stat --format="%y" "$CKPT")
  notify "Checkpoint found: $(numfmt --to=iec $CKPT_SIZE) (modified $CKPT_TIME)"
  echo "[pipeline] Checkpoint: $CKPT ($(numfmt --to=iec $CKPT_SIZE))"
  echo "[pipeline] Last modified: $CKPT_TIME"
else
  # Try epoch checkpoints
  EPOCH_CKPT=$(ls -t /home/bcloud/spec-decode/checkpoints/eagle3_draft_epoch*.bin 2>/dev/null | head -1)
  if [ -n "$EPOCH_CKPT" ]; then
    notify "No final checkpoint — using latest epoch checkpoint: $(basename $EPOCH_CKPT)"
    cp "$EPOCH_CKPT" "$CKPT"
  else
    notify "WARNING: No checkpoint found! Training may not have completed."
  fi
fi

# Check final training output
echo "[pipeline] Final training log (last 20 lines):"
tail -20 "$TRAINING_LOG" 2>/dev/null

# Rebuild
echo ""
notify "Rebuilding npu_spec_decode..."
echo "[pipeline] $(date) Rebuilding..."
cd "$BUILD_DIR"
cmake .. -DENABLE_NPU=ON -DXRT_DIR=/home/bcloud/torch2aie/toolchain/xrt 2>&1 | tail -3
make -j32 npu_spec_decode 2>&1 | tail -3

# Run benchmark
echo ""
notify "Running spec decode benchmark — this will take ~30s..."
export XILINX_XRT=/opt/xilinx/xrt
export LD_LIBRARY_PATH=$XILINX_XRT/lib64
export OMP_NUM_THREADS=32

$SPEC_BIN 9 64 2>&1 | tee "$RESULTS_FILE"
EXIT_CODE=$?

echo ""
echo "[pipeline] $(date) === PIPELINE COMPLETE (exit=$EXIT_CODE) ==="

if [ $EXIT_CODE -eq 0 ]; then
  # Parse results
  TOK_S=$(grep -oP '[0-9.]+ tok/s' "$RESULTS_FILE" | tail -1 || echo "?")
  ACCEPT=$(grep -oP '[0-9.]+%' "$RESULTS_FILE" | tail -1 || echo "?")
  SPEEDUP=$(grep -oP '[0-9.]+x' "$RESULTS_FILE" | tail -1 || echo "?")

  SUMMARY="Results: ${TOK_S:-N/A} | Acceptance: ${ACCEPT:-N/A} | Speedup: ${SPEEDUP:-N/A}"
  notify "$SUMMARY"
  echo ""
  echo "═══════════════════════════════════════════"
  echo "  $SUMMARY"
  echo "═══════════════════════════════════════════"
  echo ""
  echo "Full results saved to: $RESULTS_FILE"
else
  notify "Benchmark failed (exit=$EXIT_CODE) — check $RESULTS_FILE for errors"
fi

exit $EXIT_CODE
