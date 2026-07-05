#!/usr/bin/env bash
set -uo pipefail
# Chained pipeline: regen -> target cache -> train, for the ~10k-example scale-up.
# Each step logs to its own file; the script only proceeds if the prior step's
# output file exists and is non-empty (crude but sufficient success check for
# an unattended overnight run).

cd /home/bcloud/spec-decode
source train-venv/bin/activate

DATA_DIR=train_data_10k
REGEN_OUT=$DATA_DIR/perfectblend_train_regen.jsonl
CACHE_DIR=target_cache_10k
CKPT_NAME=eagle3_qwen3_0.6b_10k

echo "[pipeline] $(date) Step 1/3: data regeneration"
python3 scripts_local/regen_train_data.py \
  --input-file-path $DATA_DIR/perfectblend_train.jsonl \
  --output-file-path $REGEN_OUT \
  --batch-size 64 --max-new-tokens 200
if [ ! -s "$REGEN_OUT" ]; then
  echo "[pipeline] $(date) FAILED: $REGEN_OUT empty or missing"
  exit 1
fi
echo "[pipeline] $(date) Step 1/3 done: $(wc -l < $REGEN_OUT) rows"

echo "[pipeline] $(date) Step 2/3: target cache extraction"
cd /home/bcloud/DeepSpec
export PYTHONPATH=/home/bcloud/DeepSpec:${PYTHONPATH:-}
export RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
rm -rf /home/bcloud/spec-decode/$CACHE_DIR
python3 scripts/data/prepare_target_cache.py \
  --config /home/bcloud/spec-decode/configs/eagle3_qwen3_0.6b.py \
  --opts data.max_length=1024 \
  --train-data-path /home/bcloud/spec-decode/$REGEN_OUT \
  --output-dir /home/bcloud/spec-decode/$CACHE_DIR \
  --local-batch-size 8 \
  --num-workers 2
if [ ! -d "/home/bcloud/spec-decode/$CACHE_DIR" ]; then
  echo "[pipeline] $(date) FAILED: target cache dir missing"
  exit 1
fi
echo "[pipeline] $(date) Step 2/3 done"

echo "[pipeline] $(date) Step 3/3: training"
python3 train.py \
  --config /home/bcloud/spec-decode/configs/eagle3_qwen3_0.6b.py \
  --opts data.target_cache_path=/home/bcloud/spec-decode/$CACHE_DIR \
  --opts train.global_batch_size=32 \
  --opts train.local_batch_size=2 \
  --opts train.num_train_epochs=3 \
  --opts exp_name=$CKPT_NAME
echo "[pipeline] $(date) Step 3/3 done. Checkpoint at ~/checkpoints/1bit-spec/$CKPT_NAME/"
echo "[pipeline] $(date) ALL DONE"
