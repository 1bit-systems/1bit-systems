#!/usr/bin/env python3
"""
WAN 2.1 weight quantizer — convert F32 safetensors to INT8 NPU format.

Produces:
  - wan_weights.bin: flat binary of all INT8-quantized projections
  - wan_meta.json: metadata (shapes, scales, layer mapping)
"""
import json, os, struct, sys
import numpy as np

MODEL_DIR = "/home/bcloud/models/wan2.1"
OUT_DIR = "/home/bcloud/npu-sandbox/npu-infer/build/wan"

# ── helpers ────────────────────────────────────────────────────────────

def read_safetensors(path):
    """Read safetensors file, return dict of tensor_name → np.ndarray."""
    with open(path, 'rb') as f:
        hsz = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hsz).decode('utf-8'))
        meta = header.pop('__metadata__', {})
        tensors = {}
        for name, info in header.items():
            off_start, off_end = info['data_offsets']
            dtype = np.dtype({'F32': 'float32', 'F16': 'float16', 'BF16': 'bfloat16', 
                          'I8': 'int8', 'I16': 'int16', 'I32': 'int32', 'I64': 'int64',
                          'U8': 'uint8'}.get(info['dtype'], info['dtype']))
            shape = info['shape']
            count = int(np.prod(shape))
            f.seek(8 + hsz + off_start)
            data = np.frombuffer(f.read(off_end - off_start), dtype=dtype).reshape(shape)
            tensors[name] = data
    return tensors, meta


def quantize_i8(w_f32):
    """Symmetric INT8 quantization: returns (i8_data, scale)."""
    amax = np.max(np.abs(w_f32))
    if amax < 1e-12:
        return np.zeros(w_f32.shape, dtype=np.int8), 1.0
    scale = float(amax / 127.0)
    iscale = 127.0 / amax
    i8 = np.clip(np.round(w_f32 * iscale), -127, 127).astype(np.int8)
    return i8, scale


# ── main ───────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("Loading WAN 2.1 DiT weights...")
    dit_path = os.path.join(MODEL_DIR, "diffusion_pytorch_model.safetensors")
    weights, _ = read_safetensors(dit_path)
    print(f"  Loaded {len(weights)} tensors")
    
    # Group weights by block
    blocks = {}
    global_weights = {}
    for name, w in weights.items():
        if name.startswith('blocks.'):
            parts = name.split('.')
            blk = int(parts[1])
            subname = '.'.join(parts[2:])
            if blk not in blocks:
                blocks[blk] = {}
            blocks[blk][subname] = w
        else:
            global_weights[name] = w
    
    n_blocks = max(blocks.keys()) + 1
    print(f"  Blocks: {n_blocks} (0-{n_blocks-1})")
    print(f"  Global tensors: {len(global_weights)}")
    
    # Build quantized binary
    binary = bytearray()
    meta = {
        "n_blocks": n_blocks,
        "hidden_dim": 1536,
        "ffn_dim": 8960,
        "tensors": [],
        "xclbins": {
            "attn": "wan_attn_1536.xclbin",
            "ffn_gate": "wan_ffn_gate.xclbin",
            "ffn_down": "wan_ffn_down.xclbin",
        }
    }
    
    def append_i8(name, f32, out_features, in_features):
        """Quantize and append weight with correct NPU B buffer layout."""
        K_actual = in_features
        N = out_features
        kt = 64
        nt = 192 if N <= 1536 else 160
        
        # Pad K to next multiple of 512 (mtk alignment)
        K = ((K_actual + 511) // 512) * 512
        
        if f32.ndim != 2:
            print(f"  SKIP {name}: ndim={f32.ndim} shape={f32.shape}")
            return
        if f32.shape[0] != out_features or f32.shape[1] != K_actual:
            print(f"  SKIP {name}: shape mismatch {f32.shape} vs expected ({out_features},{K_actual})")
            return
        
        total = K * N
        flat = np.zeros(total, dtype=np.float32)  # zero-initialized (padding is 0)
        
        # Validate that N fits our tile scheme
        if N % (8 * nt) != 0:
            print(f"  SKIP {name}: N={N} not divisible by 8*nt={8*nt}")
            return
        
        print(f"  {name}: N={N} K_pad={K} K_actual={K_actual} nt={nt} total={total}")
        
        # Use vectorized numpy to rearrange the weight data
        # Build index arrays for the flat buffer
        out_idx = np.arange(N)[:, None]  # [N, 1]
        inn_idx = np.arange(K_actual)[None, :]  # [1, K_actual]
        
        g = out_idx // (8 * nt)
        c = (out_idx % (8 * nt)) // nt
        j = out_idx % nt
        t = inn_idx // kt
        i = inn_idx % kt
        
        p = (g * 8 + c) * nt * K + t * kt * nt + i * nt + j
        
        flat[p] = f32[:N, :K_actual]
        
        i8_data, scale = quantize_i8(flat.reshape(-1))
        
        offset = len(binary)
        binary.extend(i8_data.tobytes())
        
        # Store scale as float32
        scale_bytes = struct.pack('<f', scale)
        binary.extend(scale_bytes)
        
        entry = {
            "name": name,
            "offset": offset,
            "shape": [K, N],  # [in_f_padded, out_f]
            "out_f": N,
            "in_f": K_actual,
            "in_f_padded": K,
            "scale": float(scale),
        }
        meta["tensors"].append(entry)
        sz_mb = i8_data.nbytes / 1e6
        print(f"  {name}: [{out_features}x{in_features}] → {sz_mb:.1f}MB scale={scale:.6f}")
    
    offset = 0
    
    # Global tensors (time embedding, head, etc.)
    for name, w in sorted(global_weights.items()):
        if w.ndim == 1:
            # Bias/1D params — keep as F32 floats, not for NPU GEMM
            print(f"  {name}: shape={w.shape} (1D bias, skipping NPU quant)")
            continue
        if 'head.' in name:
            print(f"  {name}: shape={w.shape} (head/conv, skipping NPU quant)")
            continue
        if 'modulation' in name:
            print(f"  {name}: shape={w.shape} (modulation, skipping NPU quant)")
            continue
        append_i8(name, w, w.shape[0], w.shape[1] if w.ndim >= 2 else 1)
    
    # Block weights
    for blk in range(n_blocks):
        bw = blocks.get(blk, {})
        
        # Self-attention QKV — share the same xclbin shape
        for proj in ['q', 'k', 'v']:
            key = f"self_attn.{proj}.weight"
            if key in bw and bw[key].ndim >= 2:
                if 'modulation' not in key and 'bias' not in key:
                    append_i8(f"blocks.{blk}.{key}", bw[key], 1536, 1536)
        # Self-attention O
        if 'self_attn.o.weight' in bw and bw['self_attn.o.weight'].ndim >= 2:
            append_i8(f"blocks.{blk}.self_attn.o.weight", bw['self_attn.o.weight'], 1536, 1536)
        
        # Cross-attention QKV
        for proj in ['q', 'k', 'v']:
            key = f"cross_attn.{proj}.weight"
            if key in bw and bw[key].ndim >= 2:
                append_i8(f"blocks.{blk}.{key}", bw[key], 1536, 1536)
        # Cross-attention O
        if 'cross_attn.o.weight' in bw and bw['cross_attn.o.weight'].ndim >= 2:
            append_i8(f"blocks.{blk}.cross_attn.o.weight", bw['cross_attn.o.weight'], 1536, 1536)
        
        # FFN Gate
        if 'ffn.0.weight' in bw and bw['ffn.0.weight'].ndim >= 2:
            append_i8(f"blocks.{blk}.ffn.0.weight", bw['ffn.0.weight'], 8960, 1536)
        # FFN Down
        if 'ffn.2.weight' in bw and bw['ffn.2.weight'].ndim >= 2:
            append_i8(f"blocks.{blk}.ffn.2.weight", bw['ffn.2.weight'], 1536, 8960)
        
        # Bias/modulation — note but skip
        for key in bw:
            if 'bias' in key or 'modulation' in key or 'norm' in key:
                pass  # skip NPU quant for now
    
    # Write binary
    bin_path = os.path.join(OUT_DIR, "wan_weights.bin")
    with open(bin_path, 'wb') as f:
        f.write(binary)
    print(f"\n  Binary: {bin_path} ({len(binary)/1e6:.1f} MB)")
    
    # Write meta
    meta_path = os.path.join(OUT_DIR, "wan_meta.json")
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta: {meta_path}")
    
    total_gb = sum(t['shape'][0]*t['shape'][1] for t in meta['tensors']) * 1e-9
    print(f"  Total parameters (quantized): {total_gb:.1f}B")
    print("Done.")


if __name__ == "__main__":
    main()
