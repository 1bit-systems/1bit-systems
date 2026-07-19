#!/usr/bin/env python3
"""Extract ALL Zaya1-8B weights to /tmp/zaya_weights/ as float32 .bin files.

Dumps every model parameter — CCA attention (layer 0 already done elsewhere),
EDA router, MoE experts, embeddings, norms, residual scales, lm_head.

Usage: venv-hf/bin/python tools/extract_all_zaya_weights.py
"""
import os, sys, struct, json
from safetensors import safe_open

MODEL_DIR = os.environ.get("ZAYA_MODEL_DIR", os.path.expanduser("~/models/ZAYA1-8B"))
WEIGHTS_DIR = os.environ.get("ZAYA_WEIGHTS_DIR", "/tmp/zaya_weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# Load config for dimension info
with open(os.path.join(MODEL_DIR, "config.json")) as f:
    cfg = json.load(f)
H = cfg["hidden_size"]
V = cfg["vocab_size"]
N_LAYERS = cfg["num_hidden_layers"]
N_EXP = cfg["num_experts"]
N_FF = cfg["moe_intermediate_size"]       # 2048
N_EXP_PER = cfg["num_experts_per_tok"]     # 1
RTR_H = cfg.get("router_hidden_size", 256) # 256

# Collect all shard files
shards = sorted([os.path.join(MODEL_DIR, f)
                 for f in os.listdir(MODEL_DIR) if f.endswith(".safetensors")])
print(f"Loading {len(shards)} shard(s) from {MODEL_DIR}")

# Build the full weight map
to_dump = {}  # output_basename -> (numpy_array_or_none_if_skipped)
total_count = 0

def add(full_key, tensor):
    """Schedule a tensor for dumping. Returns the scheduled key name."""
    global total_count
    # Convert dots to underscores for C++ filename compatibility
    name = full_key.replace(".", "_")
    to_dump[name + ".bin"] = tensor
    total_count += 1

# Load all shards
for spath in shards:
    print(f"  Loading {os.path.basename(spath)}...")
    with safe_open(spath, framework="pt") as f:
        keys = f.keys()
        for k in keys:
            t = f.get_tensor(k)
            add(k, t)

print(f"\nDumping {total_count} tensors to {WEIGHTS_DIR}...")

dumped = 0
skipped = 0
for fname, tensor in sorted(to_dump.items()):
    arr = tensor.to(torch.float32).numpy() if hasattr(tensor, 'to') else tensor.float().numpy()
    out_path = os.path.join(WEIGHTS_DIR, fname)
    arr.tofile(out_path)
    dumped += 1
    if dumped % 100 == 0:
        print(f"  Dumped {dumped}/{total_count} ({dumped*100//total_count}%)")
    # Free memory
    del tensor, arr

print(f"\nDone! Dumped {dumped} tensors to {WEIGHTS_DIR}/")
print(f"Total size: {sum(os.path.getsize(os.path.join(WEIGHTS_DIR, f)) for f in os.listdir(WEIGHTS_DIR) if f.endswith('.bin'))/1e9:.2f} GB")
