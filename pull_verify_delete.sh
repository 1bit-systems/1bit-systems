#!/usr/bin/env bash
# Pull → verify → delete each model, one batch at a time
# Keeps disk free while proving every model is decodable

BATCH="${1:-4}"
LOG_DIR="/tmp/flm-pull-verify"
mkdir -p "$LOG_DIR"

# All models to pull (ordered small → large)
MODELS=(
  "embed-gemma:300m"
  "whisper-v3:turbo"
  "gemma3:1b"
  "llama3.2:1b"
  "lfm2:1.2b"
  "lfm2.5-it:1.2b"
  "lfm2.5-tk:1.2b"
  "qwen3.5:0.8b"
  "qwen3:1.7b"
  "nanbeige4.1:3b"
  "lfm2:2.6b"
  "lfm2-trans:2.6b"
  "llama3.2:3b"
  "qwen2.5-it:3b"
  "qwen2.5vl-it:3b"
  "qwen3:4b"
  "qwen3-it:4b"
  "qwen3-tk:4b"
  "phi4-mini-it:4b"
  "gemma3:4b"
  "qwen3.5:2b"
  "qwen3.5:4b"
  "qwen3vl-it:4b"
  "translategemma:4b"
  "medgemma:4b"
  "medgemma1.5:4b"
  "qwen3:8b"
  "llama3.1:8b"
  "deepseek-r1:8b"
  "deepseek-r1-0528:8b"
  "gemma4-it:e2b"
  "gemma4-it:e4b"
  "qwen3.5:9b"
  "gpt-oss:20b"
  "gpt-oss-sg:20b"
)

TOTAL=${#MODELS[@]}
PULLED=0
VERIFIED=0
FAILED=0
START=$(date +%s)

process_model() {
    local model="$1"
    local id="$(echo "$model" | tr ':.' '__')"
    local log="$LOG_DIR/$id.log"
    
    echo "[$(date +%H:%M:%S)] ▶ Pulling: $model"
    
    sudo flm pull "$model" --quiet > "$log" 2>&1
    local pull_rc=$?
    
    if [ $pull_rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] ✗ $model — PULL FAILED (exit $pull_rc)"
        return 1
    fi
    
    sudo flm check "$model" > "$LOG_DIR/${id}_check.log" 2>&1
    local check_rc=$?
    
    if [ $check_rc -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] ✗ $model — VERIFY FAILED"
        sudo flm remove "$model" --quiet > /dev/null 2>&1 || true
        return 1
    fi
    
    # Get size before delete
    local model_dir=$(sudo find /root/.config/flm/models -maxdepth 1 -type d -name "*${model//:/*}*" 2>/dev/null | head -1)
    local size="?"
    if [ -n "$model_dir" ]; then
        size=$(sudo du -sh "$model_dir" 2>/dev/null | cut -f1)
    fi
    
    echo "[$(date +%H:%M:%S)] ✓ $model — VERIFIED (~${size})"
    
    sudo flm remove "$model" --quiet > /dev/null 2>&1
    echo "[$(date +%H:%M:%S)] 🗑 $model — DELETED"
    
    return 0
}

echo "═══════════════════════════════════════════"
echo " Pull → Verify → Delete — $TOTAL models"
echo " Started: $(date)"
echo " Free:    $(df -h / | tail -1 | awk '{print $4}')"
echo "═══════════════════════════════════════════"

for ((i=0; i<TOTAL; i+=BATCH)); do
    BATCH_MODELS=("${MODELS[@]:i:BATCH}")
    BATCH_NUM=$((i / BATCH + 1))
    TOTAL_BATCHES=$(( (TOTAL + BATCH - 1) / BATCH ))
    
    echo ""
    echo "─── Batch $BATCH_NUM/$TOTAL_BATCHES ───"
    
    for model in "${BATCH_MODELS[@]}"; do
        process_model "$model" &
    done
    
    wait
    
    # Count results
    PULLED=$((PULLED + ${#BATCH_MODELS[@]}))
    
    elapsed=$(( $(date +%s) - START ))
    echo "  ✓ Batch $BATCH_NUM done — ${elapsed}s elapsed"
    echo "  Free: $(df -h / | tail -1 | awk '{print $4}')"
done

echo ""
echo "═══════════════════════════════════════════"
echo " DONE"
echo " Models processed: $TOTAL"
echo " Duration:        $(( ($(date +%s) - START) / 60 ))m"
echo " Free:            $(df -h / | tail -1 | awk '{print $4}')"
echo "═══════════════════════════════════════════"
