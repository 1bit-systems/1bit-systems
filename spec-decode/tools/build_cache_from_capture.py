#!/usr/bin/env python3
"""
build_cache_from_capture.py — Convert NPU-captured hidden states to training cache.

Reads the flat binary output from capture_npu_hidden_per_pos (new per-position format)
or capture_npu_hidden (old last-position format). Produces .pt for training.

New format (detected by presence of num_positions field):
  [num_examples: i32]
  For each example:
    [input_len: i32]
    [input_ids: input_len x i32]
    [num_layers: i32]
    [hidden_size: i32]
    [num_positions: i32]  (MUST be present — v2 always writes this)
    [features: num_positions x num_layers x hidden_size x f32]

Old format (no num_positions — last position only):
  Same but without num_positions, features = num_layers x hidden_size x f32

The output .pt is a list of dicts:
  {'input_ids': LongTensor[seq_len],
   'target_features': FloatTensor[num_positions, hidden_size, num_layers]}

Usage:
  python3 tools/build_cache_from_capture.py <input_bin> <output_pt> [max_examples]
"""
import os, sys, struct, torch
import numpy as np

def read_i32(f):
    data = f.read(4)
    if len(data) < 4:
        return None
    return struct.unpack('<i', data)[0]

def convert(input_path, output_path, max_examples=None):
    print(f"═══ Build Cache from Capture ═══")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    with open(input_path, 'rb') as f:
        num_examples = read_i32(f)
        if num_examples is None:
            print("ERROR: Empty file")
            return 1
        print(f"Examples in file: {num_examples}")

        if max_examples is not None and max_examples < num_examples:
            num_examples = max_examples
            print(f"  Limiting to: {num_examples}")

        cache = []

        for idx in range(num_examples):
            input_len = read_i32(f)
            if input_len is None:
                print(f"  WARNING: Unexpected EOF at example {idx}")
                break

            # Read input_ids
            input_ids_raw = f.read(input_len * 4)
            if len(input_ids_raw) < input_len * 4:
                print(f"  WARNING: Truncated input_ids at example {idx}")
                break
            input_ids = np.frombuffer(input_ids_raw, dtype=np.int32).copy()

            # Read num_layers, hidden_size
            nl = read_i32(f)
            hs = read_i32(f)
            if nl is None or hs is None:
                print(f"  WARNING: Truncated metadata at example {idx}")
                break

            # ALWAYS read num_positions (v2 format always writes it)
            num_positions = read_i32(f)
            if num_positions is None or num_positions <= 0 or num_positions > 100000:
                print(f"  WARNING: Invalid num_positions={num_positions} at example {idx}")
                break

            # Read features: num_positions * nl * hs floats
            n_features = num_positions * nl * hs
            features_raw = f.read(n_features * 4)
            if len(features_raw) < n_features * 4:
                print(f"  WARNING: Truncated features at example {idx} "
                      f"(expected {n_features*4}B, got {len(features_raw)}B)")
                break
            features = np.frombuffer(features_raw, dtype=np.float32).copy()

            # Reshape: position-major [num_positions, num_layers, hidden_size]
            features_3d = features.reshape(num_positions, nl, hs)

            # Transpose to [num_positions, hidden_size, num_layers]
            features_t = features_3d.transpose(0, 2, 1).copy()

            cache.append({
                'input_ids': torch.from_numpy(input_ids).long(),
                'target_features': torch.from_numpy(features_t).float(),
            })

            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{num_examples}")

        print(f"  Processed {len(cache)}/{num_examples} examples")
        print(f"  Positions: {sum(d['target_features'].size(0) for d in cache)} total")

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(cache, output_path)

    file_size = os.path.getsize(output_path)
    print(f"\n═══ Saved ═══")
    print(f"  {output_path} ({file_size / 1024 / 1024:.1f} MB)")
    print(f"  {len(cache)} examples")
    if cache:
        ex = cache[0]
        print(f"  input_ids: {ex['input_ids'].shape}")
        print(f"  target_features: {ex['target_features'].shape}")

    return 0

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_bin> <output_pt> [max_examples]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    max_examples = int(sys.argv[3]) if len(sys.argv) > 3 else None

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    sys.exit(convert(input_path, output_path, max_examples))
