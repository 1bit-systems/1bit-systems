#!/bin/bash
# continue_training.sh — Resume DSpark training from last checkpoint
set -euo pipefail

cd /home/bcloud/spec-decode

CACHE="/home/bcloud/spec-decode/target_cache_npu_per_pos_10k.pt"
OUTPUT_DIR="/home/bcloud/spec-decode/checkpoints/dspark_npu_10k"
VENV_PYTHON="/home/bcloud/spec-decode/train-venv/bin/python"
TRAIN_SCRIPT="/home/bcloud/spec-decode/train_dspark_from_npu.py"
LOG_FILE="/home/bcloud/spec-decode/training_dspark.log"

# Find the latest checkpoint
CHECKPOINT=""
LATEST=$(ls -t "$OUTPUT_DIR"/checkpoint_epoch*.pt 2>/dev/null | head -1)
if [[ -n "$LATEST" ]]; then
    CHECKPOINT="$LATEST"
    echo "$(date): Resuming from $CHECKPOINT" >> "$LOG_FILE"
fi

# Set up environment
export LD_LIBRARY_PATH="/home/bcloud/spec-decode/train-venv/lib/python3.14/site-packages/torch/lib:/opt/rocm-7.2.4/lib"
export OMP_NUM_THREADS=16

if [[ -n "$CHECKPOINT" ]]; then
    exec $VENV_PYTHON "$TRAIN_SCRIPT" \
        --cache "$CACHE" \
        --epochs 10 \
        --batch-size 1 \
        --output-dir "$OUTPUT_DIR" \
        --resume "$CHECKPOINT" \
        >> "$LOG_FILE" 2>&1
else
    exec $VENV_PYTHON "$TRAIN_SCRIPT" \
        --cache "$CACHE" \
        --epochs 10 \
        --batch-size 1 \
        --output-dir "$OUTPUT_DIR" \
        >> "$LOG_FILE" 2>&1
fi
