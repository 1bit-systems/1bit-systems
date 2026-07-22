#!/usr/bin/env python3
"""safetensors → 1BP converter. Reads raw bytes, handles BF16."""
import struct, sys, os, json, numpy as np

def f32b(v): return np.float32(v).view(np.uint32) >> 16

def quant_tile(data, tr=32, tc=256, gs=32):
    r, c = data.shape
    pr, pc = tr, tc
    grps = pc // gs
    padded = np.zeros((pr, pc), dtype=np.float32)
    padded[:r, :c] = data
    grouped = padded.reshape(pr, grps, gs)
    mn = grouped.min(axis=2)
    mx = grouped.max(axis=2)
    rng = mx - mn
    scale = np.where(rng < 1e-10, 0.0, rng / 15.0)
    zp = np.where(rng < 1e-10, 0.0, mn)
    scale = np.where(scale < 1e-10, 1.0, scale).astype(np.float32)
    zp = np.where(scale == 1.0, 0.0, zp).astype(np.float32)
    sc = f32b(scale).astype(np.uint16)
    zpk = f32b(zp).astype(np.uint16)
    inv = 1.0 / scale
    qi = np.clip(np.round((grouped - zp[:, :, None]) * inv[:, :, None]), 0, 15).astype(np.uint8)
    qif = qi.reshape(pr, pc)
    pk = (qif[:, 1::2] << 4) | qif[:, 0::2]
    return sc.tobytes() + zpk.tobytes() + pk.tobytes()

def tiled_size(rows, cols, tr=32, tc=256, gs=32):
    ntr = (rows + tr - 1) // tr
    ntc = (cols + tc - 1) // tc
    return ntr * ntc * (tr * (tc // gs) * 4 + tr * tc // 2)

def read_safetensors(path):
    """Read safetensors file, return list of (name, float32_array)."""
    with open(path, 'rb') as f:
        header_len = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(header_len).decode('utf-8'))
        data_start = 8 + header_len
        
        result = []
        for name, meta in header.items():
            if name == '__metadata__':
                continue
            dtype_str = meta['dtype']
            shape = meta['shape']
            offsets = meta['data_offsets']
            byte_len = offsets[1] - offsets[0]
            
            f.seek(data_start + offsets[0])
            raw = f.read(byte_len)
            
            # Convert to float32
            if dtype_str == 'BF16':
                # BF16: each 2 bytes -> upper 16 bits of float32
                n = len(raw) // 2
                arr = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                arr = arr.view(np.float32).reshape(shape)
            elif dtype_str == 'F32':
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            elif dtype_str == 'F16':
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
            else:
                print(f"  WARN: unknown dtype {dtype_str} for {name}, skipping")
                continue
            
            result.append((name, arr))
    
    return result

def map_name(hf_name):
    import re
    m = re.match(r'model\.layers\.(\d+)\.self_attn\.(\w+)\.weight', hf_name)
    if m:
        pm = {'q_proj':'attn_q','k_proj':'attn_k','v_proj':'attn_v','o_proj':'attn_output'}
        return f'blk.{m.group(1)}.{pm.get(m.group(2),m.group(2))}.weight'
    m = re.match(r'model\.layers\.(\d+)\.mlp\.(\w+)\.weight', hf_name)
    if m:
        pm = {'gate_proj':'ffn_gate','up_proj':'ffn_up','down_proj':'ffn_down'}
        return f'blk.{m.group(1)}.{pm.get(m.group(2),m.group(2))}.weight'
    m = re.match(r'model\.layers\.(\d+)\.input_layernorm\.weight', hf_name)
    if m: return f'blk.{m.group(1)}.attn_norm.weight'
    m = re.match(r'model\.layers\.(\d+)\.post_attention_layernorm\.weight', hf_name)
    if m: return f'blk.{m.group(1)}.ffn_norm.weight'
    m = re.match(r'aux_hidden_norms\.(\d+)\.weight', hf_name)
    if m: return f'blk.{m.group(1)}.hidden_norm.weight'
    mm = {'model.embed_tokens.weight':'token_embd.weight', 'model.norm.weight':'output_norm.weight',
          'lm_head.weight':'output.weight'}
    return mm.get(hf_name, hf_name)

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> output.1bp")
        sys.exit(1)
    
    model_dir, out_path = sys.argv[1], sys.argv[2]
    
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)
    
    arch = config.get("architectures", ["Unknown"])[0]
    H = config.get("hidden_size", 2048)
    L = config.get("num_hidden_layers", 6)
    NH = config.get("num_attention_heads", 72)
    NKV = config.get("num_key_value_heads", NH)
    HD = config.get("head_dim", H // NH)
    FF = config.get("intermediate_size", H * 4)
    V = config.get("vocab_size", 100352)
    max_seq_len = config.get("max_position_embeddings", 2048)
    rope_theta = config.get("rope_theta", 10000.0)
    is_dflash = "DFlash" in arch
    
    print(f"Model: {arch} H={H} L={L} NH={NH} NKV={NKV} HD={HD} FF={FF} V={V}")
    
    # Read safetensors
    st_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.safetensors')])
    tensors = []
    for sf in st_files:
        print(f"  Reading {sf}...")
        tensors.extend(read_safetensors(os.path.join(model_dir, sf)))
    
    print(f"  {len(tensors)} tensors loaded")
    
    # Build 1BP tensor list
    tr, tc, gs = 32, 256, 32
    tlist = []
    total = 0
    for hf_name, data in tensors:
        name = map_name(hf_name)
        shape = data.shape
        if len(shape) == 1:
            sz = shape[0] * 4
            tlist.append((name, 1, [shape[0]], total, sz))
            total += sz
        elif len(shape) == 2:
            rows, cols = shape[1], shape[0]
            sz = tiled_size(rows, cols, tr, tc, gs)
            tlist.append((name, 2, [rows, cols], total, sz))
            total += sz
        else:
            print(f"  SKIP {name}: {shape}")
    
    print(f"  {len(tlist)} tensors, data: {total/1e6:.1f} MB")
    
    # Write 1BP
    with open(out_path, 'wb') as f:
        hdr = bytearray(struct.pack('<23I', 
            0x00504231, 1, 0, 0, 0,
            H, L, NH, NKV, HD, FF, V, max_seq_len,
            tr, tc, gs, 0, 0, 0,
            int(rope_theta * 1000), 0, 2, len(tlist)))
        hdr.extend(b'\x00' * (256 - len(hdr)))
        assert len(hdr) == 256
        f.write(hdr)
        
        for name, ndim, dims, off, sz in tlist:
            nb = len(name)
            f.write(struct.pack('<I', nb))
            f.write(name.encode())
            f.write(b'\x00')
            f.write(struct.pack('<I', ndim))
            f.write(struct.pack(f'<{ndim}I', *dims))
            f.write(struct.pack('<QQ', off, sz))
        
        print("  Writing weights...")
        for idx, (hf_name, data) in enumerate(tensors):
            name = map_name(hf_name)
            w = data if len(data.shape) == 1 else data.T.reshape(data.shape[1], data.shape[0])
            
            if len(data.shape) == 1:
                f.write(w.astype(np.float32).tobytes())
            else:
                rows, cols = data.shape[1], data.shape[0]
                ntr = (rows + tr - 1) // tr
                ntc = (cols + tc - 1) // tc
                for rr in range(ntr):
                    for cc in range(ntc):
                        td = w[rr*tr:rr*tr+tr, cc*tc:cc*tc+tc]
                        f.write(quant_tile(td, tr, tc, gs))
            
            if (idx + 1) % 10 == 0:
                print(f"    [{idx+1}/{len(tensors)}] {name} ({f.tell()/1e6:.0f} MB)")
        
        print(f"\n  Done: {out_path} ({f.tell()/1e6:.1f} MB)")

if __name__ == '__main__':
    main()
