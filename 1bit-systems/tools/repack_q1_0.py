#!/usr/bin/env python3
"""Repack Q1_0 GGUF to 2-bit packed ternary for NPU kernel."""
import json, os, struct, sys, math, re
import numpy as np

QK = 256
BLOCK_BYTES_Q1_0 = 36

NAME_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}

def map_name(gguf_name):
    if gguf_name in NAME_MAP:
        return NAME_MAP[gguf_name]
    m = re.match(r"blk\.(\d+)\.attn_q\.weight", gguf_name)
    if m: return f"model.layers.{m.group(1)}.self_attn.q_proj.weight"
    m = re.match(r"blk\.(\d+)\.attn_k\.weight", gguf_name)
    if m: return f"model.layers.{m.group(1)}.self_attn.k_proj.weight"
    m = re.match(r"blk\.(\d+)\.attn_v\.weight", gguf_name)
    if m: return f"model.layers.{m.group(1)}.self_attn.v_proj.weight"
    m = re.match(r"blk\.(\d+)\.attn_output\.weight", gguf_name)
    if m: return f"model.layers.{m.group(1)}.self_attn.o_proj.weight"
    m = re.match(r"blk\.(\d+)\.ffn_gate\.weight", gguf_name)
    if m: return f"model.layers.{m.group(1)}.mlp.gate_proj.weight"
    m = re.match(r"blk\.(\d+)\.ffn_up\.weight", gguf_name)
    if m: return f"model.layers.{m.group(1)}.mlp.up_proj.weight"
    m = re.match(r"blk\.(\d+)\.ffn_down\.weight", gguf_name)
    if m: return f"model.layers.{m.group(1)}.mlp.down_proj.weight"
    m = re.match(r"blk\.(\d+)\.attn_norm\.weight", gguf_name)
    if m: return f"model.layers.{m.group(1)}.input_layernorm.weight"
    m = re.match(r"blk\.(\d+)\.ffn_norm\.weight", gguf_name)
    if m: return f"model.layers.{m.group(1)}.post_attention_layernorm.weight"
    m = re.match(r"blk\.(\d+)\.attn_q_norm\.weight", gguf_name)
    if m: return f"model.layers.{m.group(1)}.self_attn.q_norm.weight"
    m = re.match(r"blk\.(\d+)\.attn_k_norm\.weight", gguf_name)
    if m: return f"model.layers.{m.group(1)}.self_attn.k_norm.weight"
    return gguf_name

def skip_kv(f, n_kv):
    for i in range(n_kv):
        kl = struct.unpack('<Q', f.read(8))[0]; f.read(kl)
        vt = struct.unpack('<I', f.read(4))[0]
        if vt == 8: f.read(struct.unpack('<Q', f.read(8))[0])
        elif vt == 9:
            et = struct.unpack('<I', f.read(4))[0]; cnt = struct.unpack('<Q', f.read(8))[0]
            if et == 8:
                for _ in range(cnt): f.read(struct.unpack('<Q', f.read(8))[0])
            elif et == 5: f.read(cnt*4)
            elif et in (4,6): f.read(cnt*4)
            else: f.read(cnt*8)
        elif vt in (0,1,7): f.read(1)
        elif vt in (2,3): f.read(2)
        elif vt in (4,5,6): f.read(4)
        elif vt in (10,11,12): f.read(8)
        else: f.read(4)

def pack_q1_0(raw_data, M, K):
    """Convert Q1_0 36-byte blocks to 2-bit packed ternary."""
    K_packed = K // 4
    n_blocks_per_row = K // QK
    total_blocks = M * n_blocks_per_row
    
    data = np.frombuffer(raw_data[:total_blocks * BLOCK_BYTES_Q1_0], dtype=np.uint8)
    blocks = data.reshape(total_blocks, BLOCK_BYTES_Q1_0)
    
    # Extract per-block scales d from bytes 0:2
    d_bytes = blocks[:, :2].copy()
    d = np.frombuffer(d_bytes.tobytes(), dtype=np.float16).astype(np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).reshape(M, n_blocks_per_row)
    # Per-block scales (NOT row-averaged — each block keeps its own scale)
    block_scales = d.astype(np.float32)  # [M, n_blocks_per_row]
    
    # Extract 1-bit qs from bytes 2:34 (32 bytes = 256 bits)
    qs_raw = blocks[:, 2:34].copy()
    qs_expanded = np.unpackbits(qs_raw, axis=1, bitorder='little')  # [blocks, 256]
    # Map Q1_0: bit 0 → -1 (ternary code 0b00), bit 1 → +1 (ternary code 0b10)
    ternary = np.where(qs_expanded == 0, np.uint8(0), np.uint8(2)).astype(np.uint8)
    
    # Pack 4 × 2-bit values per byte (per-block: 256 values → 64 bytes)
    block_k_packed = QK // 4  # 64
    packed = np.zeros((total_blocks, block_k_packed), dtype=np.uint8)
    for i in range(4):
        packed |= (ternary[:, i::4] & np.uint8(3)) << np.uint8(i * 2)
    
    # Reshape to [M, K_packed] by interleaving blocks across rows
    weights = packed.reshape(M, n_blocks_per_row, block_k_packed)
    weights = weights.reshape(M, K_packed)
    return weights, block_scales

def main():
    src, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    
    with open(src, 'rb') as gf:
        gf.read(4); gf.read(4)
        n_tensors = struct.unpack('<Q', gf.read(8))[0]
        n_kv = struct.unpack('<Q', gf.read(8))[0]
        skip_kv(gf, n_kv)
        
        # First read ALL tensor metadata
        tensors = []
        for _ in range(n_tensors):
            nl = struct.unpack('<Q', gf.read(8))[0]
            name = gf.read(nl).decode()
            nd = struct.unpack('<I', gf.read(4))[0]
            dims = [struct.unpack('<Q', gf.read(8))[0] for _ in range(nd)]
            dt = struct.unpack('<I', gf.read(4))[0]
            off = struct.unpack('<Q', gf.read(8))[0]
            tensors.append((name, dims, dt, off))
        
        # data_start is aligned position after ALL tensor metadata
        data_start = (gf.tell() + 31) & ~31
        
        manifest = {}
        chunks = []
        total_bytes = 0
        
        for name, dims, dt, off in tensors:
            hf_name = map_name(name)
            n_elems = int(math.prod(dims))
            
            if dt == 41:  # Q1_0 / IQ1_S
                # GGUF stores [in_features, out_features] = [K, M]
                K = dims[0]; M = dims[1] if len(dims) > 1 else 1
                if K == 1 or M == 1: continue
                
                # Store embeddings/lm_head as raw F16 (decode Q1_0→F16)
                if 'token_embd' in name or name == 'output.weight':
                    raw_size = (n_elems // QK) * BLOCK_BYTES_Q1_0
                    gf.seek(data_start + off)
                    raw = gf.read(raw_size)
                    # Decode Q1_0 to F16
                    data = np.frombuffer(raw[:raw_size], dtype=np.uint8)
                    blocks = data.reshape(-1, BLOCK_BYTES_Q1_0)
                    d_bytes = blocks[:, :2].copy()
                    d = np.frombuffer(d_bytes.tobytes(), dtype=np.float16).astype(np.float32)
                    qs_raw = blocks[:, 2:34]
                    qs = np.unpackbits(qs_raw, axis=1, bitorder='little')  # [blocks, 256]
                    # Decode: bit 0→-d, bit 1→+d
                    vals = np.where(qs == 0, -d[:, None], d[:, None]).astype(np.float32).reshape(-1)
                    vals_f16 = vals.astype(np.float16).tobytes()
                    manifest[hf_name] = {"shape": [K, M], "dtype": "F16",
                                         "offset_bytes": total_bytes, "size_bytes": len(vals_f16)}
                    chunks.append(vals_f16)
                    total_bytes += len(vals_f16)
                    print(f"  {hf_name}: F16 [{K}, {M}] (decoded from Q1_0)")
                    continue
                
                raw_size = (n_elems // QK) * BLOCK_BYTES_Q1_0
                gf.seek(data_start + off)
                raw = gf.read(raw_size)
                
                weights, block_scales = pack_q1_0(raw, M, K)
                K_packed = K // 4
                n_blocks = K // 256
                wb = weights.tobytes()
                # Store per-block scales as BF16 (one per 256-weight block)
                # block_scales shape: [M, n_blocks]
                sb_flat = (block_scales.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).ravel().tobytes()
                
                off_w = total_bytes; off_s = off_w + len(wb)
                chunks.append(wb + sb_flat)
                total_bytes += len(wb) + len(sb_flat)
                
                manifest[hf_name] = {"M": M, "K": K, "K_packed": K_packed, "n_blocks": n_blocks,
                                     "offset_weights": off_w, "offset_scales": off_s,
                                     "size_bytes": len(wb) + len(sb_flat)}
                print(f"  {hf_name}: M={M} K={K} n_blk={n_blocks} → {len(wb)+len(sb_flat)} bytes")
            
            elif dt in (0, 1):
                el_sz = 4 if dt == 0 else 2
                gf.seek(data_start + off)
                raw = gf.read(n_elems * el_sz)
                manifest[hf_name] = {"shape": list(dims), "dtype": "F32" if dt == 0 else "F16",
                                     "offset_bytes": total_bytes, "size_bytes": len(raw)}
                chunks.append(raw)
                total_bytes += len(raw)
        
        with open(os.path.join(out_dir, 'weights.bin'), 'wb') as wf:
            for c in chunks: wf.write(c)
        with open(os.path.join(out_dir, 'manifest.json'), 'w') as mf:
            json.dump(manifest, mf, indent=2)
        
        print(f"\nWrote {len(manifest)} tensors, {total_bytes/1e6:.1f} MB")

if __name__ == '__main__':
    main()
