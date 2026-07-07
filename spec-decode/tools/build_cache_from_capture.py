#!/usr/bin/env python3
"""
build_cache_from_capture.py — Convert NPU-captured hidden states to training cache.

Reads the flat binary output from capture_npu_hidden and produces a .pt file
in the format expected by train_from_cache.py.

The binary format is:
  [num_examples: i32]
  For each example:
    [input_len: i32]
    [input_ids: input_len × i32]
    [num_layers: i32]  (= 5)
    [hidden_size: i32] (= 1024)
    [features: num_layers × hidden_size × f32]

The output .pt is a list of dicts:
  {'input_ids': LongTensor[seq_len], 'target_features': FloatTensor[hidden_size, num_layers]}

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

def read_f32(f, n):
    data = f.read(n * 4)
    if len(data) < n * 4:
        return None
    return struct.unpack(f'<{n}f', data)

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
            # Read input_len
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
            
            # Read features
            n_features = nl * hs
            features_raw = f.read(n_features * 4)
            if len(features_raw) < n_features * 4:
                print(f"  WARNING: Truncated features at example {idx}")
                break
            features = np.frombuffer(features_raw, dtype=np.float32).copy()
            
            # Reshape: [num_layers, hidden_size] -> [hidden_size, num_layers]
            # (matching the train_from_cache.py expected format)
            features_2d = features.reshape(nl, hs).T.copy()  # [hs, nl]
            
            cache.append({
                'input_ids': torch.from_numpy(input_ids).long(),
                'target_features': torch.from_numpy(features_2d).float(),
            })
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{num_examples}")
        
        print(f"  Processed {len(cache)}/{num_examples} examples")
    
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
