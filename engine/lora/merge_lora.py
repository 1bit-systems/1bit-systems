#!/usr/bin/env python3
"""
merge_lora.py — Convert a PEFT LoRA adapter to a compact .lora delta file
for load-time merging into the NPU engine.

Usage:
  python3 merge_lora.py adapter_dir/ output.lora

The .lora file contains pre-computed B·A deltas in float32, one per layer/module.
The NPU engine reads this at init time and adds the deltas to dequantized weights
before re-quantizing to INT8 for the NPU GEMM.
"""

import os, sys, struct, json, numpy as np
from safetensors import safe_open

KNOWN_MODULES = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}

def parse_module_key(key):
    """Parse PEFT LoRA keys into (layer, module, part).

    Standard:  base_model.model.model.layers.23.self_attn.q_proj.lora_A.weight
    ZAYA:      base_model.model.model.layers.0.self_attn.qkv_proj.q_proj.lora_A.weight
               base_model.model.model.layers.0.mlp.gate.down_proj.lora_A.weight
    → (layer, module, part) e.g. (23, 'q_proj', 'A')
    """
    parts = key.split('.')
    layer = None
    module = None
    part = None
    for i, p in enumerate(parts):
        if p == 'layers' and i + 1 < len(parts):
            layer = int(parts[i + 1])
    # Module name is the token right before .lora_A or .lora_B
    for i, p in enumerate(parts):
        if p in ('lora_A', 'lora_B'):
            part = p.replace('lora_', '')
            if i > 0 and parts[i - 1] in KNOWN_MODULES:
                module = parts[i - 1]
    return layer, module, part

# Module name → group_id for the NPU engine
# group 0 = QKV (fused), group 1 = O, group 2 = GU (fused), group 3 = D
MODULE_GROUP = {
    'q_proj': 0, 'k_proj': 0, 'v_proj': 0,
    'o_proj': 1,
    'gate_proj': 2, 'up_proj': 2,
    'down_proj': 3,
}

# In the NPU engine, the fused groups have these offsets:
# QKV: Q at offset 0 (2048 rows), K at offset 2048 (1024 rows), V at offset 3072 (1024 rows)
# GU: gate at offset 0 (3072 rows), up at offset 3072 (3072 rows)
MODULE_OFFSET_IN_GROUP = {
    'q_proj': (0, 0),       # (group, row_offset)
    'k_proj': (0, 2048),
    'v_proj': (0, 3072),
    'o_proj': (1, 0),
    'gate_proj': (2, 0),
    'up_proj': (2, 3072),
    'down_proj': (3, 0),
}

def compute_deltas(lora_path):
    """Read a PEFT adapter. Return A/B matrices and config, NOT pre-computed deltas."""
    with open(os.path.join(lora_path, "adapter_config.json")) as cf:
        config = json.load(cf)
    
    lora_alpha = config.get("lora_alpha", 16)
    rank = config.get("r", 8)
    use_rslora = config.get("use_rslora", False)
    scale = lora_alpha / rank if not use_rslora else lora_alpha / (rank ** 0.5)
    
    # Load A/B matrices
    ab_mats = {}  # (layer, module) → {'A': np.array, 'B': np.array}
    with safe_open(os.path.join(lora_path, "adapter_model.safetensors"), framework="np") as f:
        keys = list(f.keys())
        for key in keys:
            if not key.endswith('.weight'):
                continue
            layer, module, part = parse_module_key(key)
            if module is None:
                continue
            key2 = (layer, module)
            if key2 not in ab_mats:
                ab_mats[key2] = {}
            t = f.get_tensor(key)
            ab_mats[key2][part] = np.array(t, dtype=np.float32)
    
    return ab_mats, rank, lora_alpha, scale

def write_lora_file(ab_mats, scale, output_path):
    """Write A/B matrices to a compact binary .lora file.

    Format:
      magic: b'LORA'
      num_layers: uint32
      scale: float32  (lora_alpha / rank)
      For each layer (0..num_layers-1):
        num_modules: uint32
        For each module:
          module_id: uint32
          rank: uint32
          in_dim: uint32
          out_dim: uint32
          A_data: rank * in_dim * float32
          B_data: out_dim * rank * float32

    Size for rank=8: ~5 MB per adapter (vs 20 MB for PEFT safetensors)
    """
    MODULE_IDS = {
        'q_proj': 0, 'k_proj': 1, 'v_proj': 2,
        'o_proj': 3,
        'gate_proj': 4, 'up_proj': 5, 'down_proj': 6,
    }
    
    with open(output_path, 'wb') as f:
        num_layers = max(l for (l, _) in ab_mats) + 1
        f.write(b'LORA')
        f.write(struct.pack('<If', num_layers, scale))

        for layer in range(num_layers):
            layer_modules = [(mod, mats) for (l, mod), mats in ab_mats.items() if l == layer]
            f.write(struct.pack('<I', len(layer_modules)))
            
            for mod, mats in sorted(layer_modules, key=lambda x: MODULE_IDS.get(x[0], 99)):
                mod_id = MODULE_IDS.get(mod, 99)
                rank = mats['A'].shape[0]
                in_dim = mats['A'].shape[1]
                out_dim = mats['B'].shape[0]
                A_data = np.ascontiguousarray(mats['A'].flatten())
                B_data = np.ascontiguousarray(mats['B'].flatten())
                
                f.write(struct.pack('<IIII', mod_id, rank, in_dim, out_dim))
                f.write(A_data.tobytes())
                f.write(B_data.tobytes())
    
    return output_path

def read_lora_file(path):
    """Read a .lora file and return AB matrices + scale.
    
    Returns: (ab_mats, scale) where ab_mats[(layer, mod)] = {'A': ..., 'B': ...}
    """
    ab_mats = {}
    with open(path, 'rb') as f:
        magic = f.read(4)
        assert magic == b'LORA', f"Bad magic: {magic}"
        num_layers, scale = struct.unpack('<If', f.read(8))
        
        MODULE_NAMES = {0: 'q_proj', 1: 'k_proj', 2: 'v_proj', 3: 'o_proj',
                        4: 'gate_proj', 5: 'up_proj', 6: 'down_proj'}
        
        for layer in range(num_layers):
            num_modules = struct.unpack('<I', f.read(4))[0]
            for _ in range(num_modules):
                mod_id, rank, in_dim, out_dim = struct.unpack('<IIII', f.read(16))
                mod_name = MODULE_NAMES.get(mod_id, f"unknown_{mod_id}")
                
                a_bytes = rank * in_dim * 4
                b_bytes = out_dim * rank * 4
                A = np.frombuffer(f.read(a_bytes), dtype=np.float32).reshape(rank, in_dim)
                B = np.frombuffer(f.read(b_bytes), dtype=np.float32).reshape(out_dim, rank)
                ab_mats[(layer, mod_name)] = {'A': A, 'B': B}
    
    return ab_mats, scale

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 merge_lora.py <adapter_dir> <output.lora>")
        print("       python3 merge_lora.py --read <output.lora>  (verify)")
        sys.exit(1)
    
    if sys.argv[1] == '--read':
        ab_mats, scale = read_lora_file(sys.argv[2])
        print(f"Read {len(ab_mats)} AB pairs from {sys.argv[2]} (scale={scale:.4f}):")
        total_a = 0
        total_b = 0
        for (layer, mod), mats in sorted(ab_mats.items()):
            A = mats['A']
            B = mats['B']
            total_a += A.nbytes
            total_b += B.nbytes
        print(f"  Total A: {total_a / 1024 / 1024:.1f} MB")
        print(f"  Total B: {total_b / 1024 / 1024:.1f} MB")
        print(f"  Combined: {(total_a + total_b) / 1024 / 1024:.1f} MB")
        return
    
    adapter_dir = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Reading LoRA from {adapter_dir}...")
    ab_mats, rank, alpha, scale = compute_deltas(adapter_dir)
    
    if not ab_mats:
        print("ERROR: No LoRA weights found")
        sys.exit(1)
    
    a_bytes = sum(m['A'].nbytes for m in ab_mats.values())
    b_bytes = sum(m['B'].nbytes for m in ab_mats.values())
    print(f"  Found {len(ab_mats)} AB pairs (rank={rank}, alpha={alpha}, scale={scale:.4f})")
    print(f"  A matrices: {a_bytes / 1024 / 1024:.1f} MB, B matrices: {b_bytes / 1024 / 1024:.1f} MB")
    
    print(f"Writing {output_path}...")
    write_lora_file(ab_mats, scale, output_path)
    
    # Verify
    verify, vscale = read_lora_file(output_path)
    vtotal = sum(m['A'].nbytes + m['B'].nbytes for m in verify.values())
    print(f"  Written {len(verify)} AB pairs, {vtotal / 1024 / 1024:.1f} MB")
    print(f"\nDone! Use: npu_engine --lora {output_path}")

if __name__ == '__main__':
    main()
