#!/usr/bin/env bash
# Extract ALL Zaya1-8B weights to /tmp/zaya_weights/ as float32 .bin files.
# Usage: bash tools/extract_all_zaya_weights.sh
set -e
cd ${HOME}
WEIGHTS_DIR="${ZAYA_WEIGHTS_DIR:-/tmp/zaya_weights}"
mkdir -p "$WEIGHTS_DIR"

# Check if already done
COUNT=$(ls "$WEIGHTS_DIR"/*.bin 2>/dev/null | wc -l)
if [ "$COUNT" -ge 1280 ]; then
    echo "Already have $COUNT weight files in $WEIGHTS_DIR — skipping extraction."
    exit 0
fi

echo "Extracting ALL Zaya weights to $WEIGHTS_DIR/ (1283 tensors)..."
echo "Using unsloth-env python (torch + safetensors)"

# Run the Python extraction
unsloth-env/bin/python3 -c "
import torch, os, sys, json, time
from safetensors import safe_open
import numpy as np

MODEL_DIR = os.environ.get('ZAYA_MODEL_DIR', '${HOME}/models/ZAYA1-8B')
WEIGHTS_DIR = os.environ.get('ZAYA_WEIGHTS_DIR', '/tmp/zaya_weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

shards = sorted([os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith('.safetensors')])
print(f'Loading {len(shards)} shard(s) from {MODEL_DIR}')

total = 0
t0 = time.time()
for spath in shards:
    with safe_open(spath, framework='pt') as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)  # bfloat16 tensor
            arr = t.to(torch.float32).numpy()
            name = k.replace('.', '_') + '.bin'
            out_path = os.path.join(WEIGHTS_DIR, name)
            arr.tofile(out_path)
            total += 1
            if total % 200 == 0:
                dt = time.time() - t0
                print(f'  [{total}/1283] {dt:.1f}s ({total/dt:.0f} tensors/s)')

total_size = sum(os.path.getsize(os.path.join(WEIGHTS_DIR, f)) 
                 for f in os.listdir(WEIGHTS_DIR) if f.endswith('.bin'))
dt = time.time() - t0
print(f'\\nDone! {total} tensors dumped ({total_size/1e9:.2f} GB in {dt:.0f}s)')
" 2>&1
