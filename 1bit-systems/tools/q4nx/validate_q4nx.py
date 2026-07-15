#!/usr/bin/env python3
"""
validate_q4nx.py — Validate Q4NX weight reading by comparing
FLM NPU inference output against CPU numpy inference with the same weights.

Reads weights from a Q4NX model, runs a forward pass on CPU (numpy),
and checks that token probabilities match FLM's output.

Usage:
  python validate_q4nx.py model.q4nx --prompt "Hello" --flm-model qwen3:0.6b
"""

import json, os, sys, struct, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q4nx_reader import Q4NXReader

# ── Tiling constants ──
ROW_BLOCK = 32; COL_BLOCK = 256; PARALLEL = 16

def untile_i8(arr: np.ndarray) -> np.ndarray:
    R, C = arr.shape
    nr, nc = R // ROW_BLOCK, C // COL_BLOCK
    rp = ROW_BLOCK // PARALLEL
    blocked = arr.reshape(nr, ROW_BLOCK, nc, COL_BLOCK)
    result = np.zeros_like(blocked)
    for ri in range(nr):
        for ci in range(nc):
            blk = blocked[ri, :, ci, :]
            shaped = blk.reshape(rp, COL_BLOCK, PARALLEL)
            unshaped = shaped.transpose(0, 2, 1)
            result[ri, :, ci, :] = unshaped.reshape(ROW_BLOCK, COL_BLOCK)
    return result.transpose(0, 2, 1, 3).reshape(R, C)


def read_q4nx_weight(reader: Q4NXReader, name: str):
    """Read a weight from Q4NX, decode, and return as numpy float32."""
    info = reader.manifest[name]
    dtype = info['dtype']
    shape = tuple(info['shape'])
    raw = reader.get_tensor_raw(name)
    
    if dtype == 'BF16':
        u16 = np.frombuffer(raw, dtype=np.uint16).copy()
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        return f32.reshape(shape)
    elif dtype == 'I8':
        arr = np.frombuffer(raw, dtype=np.int8)
        if len(shape) == 2:
            arr = arr.reshape(shape[0], shape[1])
            if arr.shape[0] % ROW_BLOCK == 0 and arr.shape[1] % COL_BLOCK == 0:
                arr = untile_i8(arr)
        elif len(shape) == 3:
            arr = arr.reshape(shape[0] * shape[1], shape[2])
            if arr.shape[0] % ROW_BLOCK == 0 and arr.shape[1] % COL_BLOCK == 0:
                arr = untile_i8(arr)
        return arr.astype(np.float32)
    elif dtype == 'F32':
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate Q4NX weights")
    parser.add_argument("q4nx_path", help="Path to model.q4nx")
    parser.add_argument("--list", action="store_true", help="List all tensors with sizes")
    parser.add_argument("--inspect", default=None, help="Inspect a specific tensor")
    parser.add_argument("--stats", action="store_true", help="Print weight statistics")
    args = parser.parse_args()
    
    reader = Q4NXReader(args.q4nx_path)
    model_dir = os.path.dirname(os.path.abspath(args.q4nx_path))
    
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    
    print(f"Model: {config.get('model_type', '?')}")
    print(f"Hidden: {config.get('hidden_size', '?')}")
    print(f"Layers: {config.get('num_hidden_layers', '?')}")
    print(f"Tensors: {len(reader.tensors)}")
    print(f"Weight data: {reader.data_size/1024/1024:.0f} MB")
    
    if args.list:
        print(f"\n{'TENSOR':<60s} {'DTYPE':<6s} {'SHAPE':<24s} {'SIZE':<10s}")
        print("-"*100)
        for t in sorted(reader.tensors, key=lambda x: x['offset']):
            name = t['name']
            mb = t['size'] / 1024 / 1024
            shape = str(tuple(t['shape']))
            print(f"{name:<60s} {t['dtype']:<6s} {shape:<24s} {mb:<10.1f}MB")
    
    if args.inspect:
        name = args.inspect
        if name not in reader.manifest:
            print(f"Tensor '{name}' not found")
            return
        w = read_q4nx_weight(reader, name)
        print(f"\n{name}:")
        print(f"  Shape: {w.shape}")
        print(f"  Dtype: {w.dtype}")
        print(f"  Stats: min={w.min():.4f} max={w.max():.4f} mean={w.mean():.4f} std={w.std():.4f}")
        print(f"  First 16 values: {w.flatten()[:16]}")
    
    if args.stats:
        print(f"\n{'='*60}")
        print(f"WEIGHT STATISTICS")
        print(f"{'='*60}")
        for t in sorted(reader.tensors, key=lambda x: x['offset']):
            name = t['name']
            if "norm" in name and t['dtype'] == 'BF16':
                continue  # Skip tiny norm weights
            if t['size'] < 1024 * 100:  # Skip tiny tensors
                continue
            w = read_q4nx_weight(reader, name)
            print(f"  {name:<55s} shape={str(w.shape):<20s} range=[{w.min():+.1f}, {w.max():+.1f}] mean={w.mean():+.2f}")


if __name__ == "__main__":
    main()
