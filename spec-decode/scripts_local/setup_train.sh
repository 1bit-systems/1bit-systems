#!/usr/bin/env bash
# Setup + train Eagle3 draft model for 1bit speculative decoding
# Run on any GPU machine with PyTorch
#
# Usage:
#   ./setup_train.sh                    # Full pipeline: setup → train → export
#   ./setup_train.sh --gpu 0            # Use specific GPU
#   ./setup_train.sh --epochs 5         # Fewer epochs for faster testing
#   ./setup_train.sh --cloud            # Setup for cloud GPU (no local DeepSpec)

set -euo pipefail
cd "$(dirname "$0")/.."
SPEC_DIR="$PWD"

echo "═══ Eagle3 Draft Training Pipeline ═══"
echo ""

# Parse args
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-10}"
CLOUD="${CLOUD:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --cloud) CLOUD=1; shift ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

# Check GPU
if ! python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}')" 2>/dev/null; then
  echo "❌ PyTorch not found. Installing..."
  pip install torch --index-url https://download.pytorch.org/whl/cu124 2>/dev/null || pip install torch
fi

HAS_GPU=$(python3 -c "import torch; print(int(torch.cuda.is_available()))")
if [ "$HAS_GPU" = "0" ]; then
  echo "⚠️  No GPU detected — training on CPU (will be slow, ~1 hour)"
  echo "   Recommend: use --cloud mode or run on a GPU instance"
else
  echo "✅ GPU $(python3 -c "import torch; print(torch.cuda.get_device_name(0))")"
fi

# Step 1: Setup DeepSpec
if [ ! -d "/home/bcloud/DeepSpec" ] && [ "$CLOUD" = "0" ]; then
  echo "Step 1: Cloning DeepSpec..."
  git clone https://github.com/deepseek-ai/DeepSpec /home/bcloud/DeepSpec
  pip install -r /home/bcloud/DeepSpec/requirements.txt
elif [ "$CLOUD" = "1" ]; then
  echo "Step 1: Installing DeepSpec..."
  pip install git+https://github.com/deepseek-ai/DeepSpec
fi
echo "✅ DeepSpec ready"

# Step 2: Prepare target cache
echo ""
echo "Step 2: Preparing target cache..."
CACHE_DIR="$SPEC_DIR/target_cache_10k"
rm -rf "$CACHE_DIR"
export CUDA_VISIBLE_DEVICES=$GPU
export RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
python3 -c "
import sys
sys.path.insert(0, '/home/bcloud/DeepSpec')
from deepspec.scripts.data.prepare_target_cache import main as prep
import argparse
args = argparse.Namespace(
    config='$SPEC_DIR/configs/eagle3_qwen3_0.6b.py',
    train_data_path='$SPEC_DIR/train_data_10k/perfectblend_train.jsonl',
    output_dir='$CACHE_DIR',
    local_batch_size=8,
    num_workers=2,
    max_length=1024,
)
prep(args)
" 2>&1 | tail -5
echo "✅ Target cache ready ($(ls $CACHE_DIR/*.bin 2>/dev/null | wc -l) files)"

# Step 3: Train
echo ""
echo "Step 3: Training Eagle3 draft..."
CKPT_NAME="eagle3_qwen3_0.6b_10k"
python3 /home/bcloud/DeepSpec/train.py \
  --config "$SPEC_DIR/configs/eagle3_qwen3_0.6b.py" \
  --opts "data.target_cache_path=$CACHE_DIR" \
  --opts "train.global_batch_size=32" \
  --opts "train.local_batch_size=2" \
  --opts "train.num_train_epochs=$EPOCHS" \
  --opts "exp_name=$CKPT_NAME" 2>&1 | tail -10
echo "✅ Training complete"

# Step 4: Export to C++ format
echo ""
echo "Step 4: Exporting weights..."
CKPT_PATH="/home/bcloud/checkpoints/1bit-spec/$CKPT_NAME/checkpoint-*/model.safetensors"
python3 "$SPEC_DIR/scripts_local/export_draft_weights.py" \
  --checkpoint $(ls $CKPT_PATH | head -1) \
  --output "$SPEC_DIR/checkpoints/eagle3_draft.bin"
echo "✅ Exported to $SPEC_DIR/checkpoints/eagle3_draft.bin ($(du -h $SPEC_DIR/checkpoints/eagle3_draft.bin | cut -f1))"

# Step 5: Benchmark
echo ""
echo "Step 5: Running benchmark..."
cd "$SPEC_DIR/build" && cmake .. -DENABLE_NPU=OFF && make -j4 spec_decode_bench 2>/dev/null
./spec_decode_bench --checkpoint "$SPEC_DIR/checkpoints/eagle3_draft.bin" 2>&1 | tail -15

echo ""
echo "═══ DONE ═══"
echo "Draft checkpoint: $SPEC_DIR/checkpoints/eagle3_draft.bin"
echo "Test: cd $SPEC_DIR/build && ./npu_spec_decode"
