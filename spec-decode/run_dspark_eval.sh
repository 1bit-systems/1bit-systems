#!/usr/bin/env bash
# Run DSpark acceptance eval after Qwen3-4B is downloaded
# Compares DSpark (5 layers + Markov + confidence) vs Eagle3 (1 layer)
set -ueo pipefail

cd /home/bcloud/DeepSpec
source /home/bcloud/spec-decode/train-venv/bin/activate

echo "[dspark] $(date) Waiting for Qwen3-4B download..."

# Wait for download to complete
while true; do
  if [ -f "/home/bcloud/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/"*"/model.safetensors" ]; then
    echo "[dspark] $(date) Qwen3-4B downloaded"
    break
  fi
  # Check if download process is still running
  if ! ps aux | grep "AutoModelForCausalLM.from_pretrained" | grep -v grep > /dev/null 2>&1; then
    # Process might have crashed or exited - check if model exists
    if [ -f "/home/bcloud/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/"*"/model.safetensors" ]; then
      echo "[dspark] $(date) Qwen3-4B downloaded (process finished)"
      break
    fi
  fi
  tail -1 /tmp/qwen3_4b_download.log 2>/dev/null || true
  sleep 30
done

echo "[dspark] $(date) Running DSpark eval with 10 samples..."
echo "Target: Qwen/Qwen3-4B"
echo "Draft: /home/bcloud/spec-decode/checkpoints/dspark_qwen3_4b"

# Run eval with very limited scope (just gsm8k with 10 samples)
python3 eval.py \
  --target_name_or_path Qwen/Qwen3-4B \
  --draft_name_or_path /home/bcloud/spec-decode/checkpoints/dspark_qwen3_4b \
  --max-new-tokens 64 \
  --seed 42 2>&1 | tee /tmp/dspark_eval_output.log

echo "[dspark] $(date) === DSPARK EVAL COMPLETE ==="
echo ""
echo "Now compare with Eagle3 results from npu_spec_decode"
