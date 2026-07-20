#!/usr/bin/env python3
"""GGUF → 1BP converter with actual Q4NX quantization.
Uses the gguf Python package for tensor reading + numpy for quant."""
import struct, sys, os, numpy as np
from gguf import GGUFReader, dequantize

def f32b(v): return np.float32(v).view(np.uint32) >> 16

def quant_tile(data, tr=32, tc=256, gs=32):
    r, c = data.shape
    pr, pc = tr, tc
    grps = pc // gs
    padded = np.zeros((pr, pc), dtype=np.float32)
    padded[:r, :c] = data
    sc = np.zeros((pr, grps), dtype=np.uint16)
    zp = np.zeros((pr, grps), dtype=np.uint16)
    pk = np.zeros((pr, pc // 2), dtype=np.uint8)
    for rr in range(pr):
        for g in range(grps):
            c0 = g * gs
            ch = padded[rr, c0:c0+gs]
            mn, mx = ch.min(), ch.max()
            if mx - mn < 1e-10:
                scale, mn = 0.0, 0.0
            else:
                scale = (mx - mn) / 15.0
            if scale < 1e-10:
                scale = 1.0; mn = 0.0
            inv = 1.0 / scale
            sc[rr, g] = f32b(scale)
            zp[rr, g] = f32b(mn)
            qi = np.clip(np.round((ch - mn) * inv), 0, 15).astype(np.uint8)
            for i in range(0, gs, 2):
                bi = (rr * pc + c0 + i) // 2
                v0 = qi[i] if i < gs else 0
                v1 = qi[i+1] if i+1 < gs else 0
                pk.flat[bi] = (v1 << 4) | v0
    return sc.tobytes() + zp.tobytes() + pk.tobytes()

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.gguf output.1bp [max_tensors]"); sys.exit(1)
    
    print(f"Reading {sys.argv[1]}...")
    rd = GGUFReader(sys.argv[1])
    max_t = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    def gf(field, alt=None):
        for fn in [field, alt] if alt else [field]:
            if not fn: continue
            v = rd.fields.get(fn)
            if v is None or len(v.parts) < 4: continue
            val = v.parts[3]
            if hasattr(val, '__len__') and len(val) > 0:
                try: return int(val[0])
                except: pass
            try: return int(val)
            except: pass
        return 0
    
    def gs_str(field):
        v = rd.fields.get(field)
        if v is None or len(v.parts) < 5: return ''
        raw = v.parts[4]
        if hasattr(raw, 'tobytes'):
            try: return bytes(raw.tobytes()).decode('utf-8')
            except: return ''
        return ''
    
    arch = gs_str("general.architecture") or "unknown"
    H = gf("hidden_size") or gf(f"{arch}.embedding_length")
    L = gf("num_hidden_layers") or gf(f"{arch}.block_count")
    NH = gf("num_attention_heads") or gf(f"{arch}.attention.head_count")
    NKV = gf("num_key_value_heads") or gf(f"{arch}.attention.head_count_kv")
    HD = gf("head_dim") or gf(f"{arch}.attention.key_length")
    IM = gf("intermediate_size") or gf(f"{arch}.feed_forward_length")
    V = gf("vocab_size") or gf(f"{arch}.vocab_size")
    if not NKV: NKV = NH
    if not HD and NH: HD = H // NH
    
    print(f"Model: {arch}  H={H} L={L} NH={NH} NKV={NKV} HD={HD} IM={IM} V={V}")
    if not H or not L:
        print("ERROR: could not read model config"); sys.exit(1)
    
    # Build header
    tr, tc, gs = 32, 256, 32
    hdr = struct.pack('<5I8i10I',
        0x00504231, 1, 0, 0, 0,
        H, L, NH, NKV, HD, IM, V, 4096,
        tr, tc, gs, 0, 0, 0, 1000000, 1, 2, 0)
    hdr = bytearray(hdr.ljust(256, b'\x00'))
    
    # Collect tensors
    tlist = []
    total = 0
    for tn in rd.tensors:
        if len(tn.shape) != 2: continue
        rows = int(tn.shape[1]) if len(tn.shape) >= 2 else 1
        cols = int(tn.shape[0]) if len(tn.shape) >= 2 else int(tn.shape[0])
        if len(tn.shape) == 1:
            rows, cols = 1, int(tn.shape[0])
        elif rows * cols > 600_000_000:  # allow up to 600M (covers token_embd)
            print(f"  SKIP oversized: {tn.name} {rows}x{cols}")
            continue
        ntr = (rows + tr - 1) // tr; ntc = (cols + tc - 1) // tc
        tsz = ntr * ntc * (tr * (tc // gs) * 4 + tr * tc // 2)
        tlist.append((tn.name, rows, cols, total, tsz))
        total += tsz
    
    print(f"Tensors: {len(tlist)}, data: {total/1e6:.1f} MB")
    struct.pack_into('<I', hdr, 88, len(tlist))
    
    # Write output
    fout = open(sys.argv[2], 'wb')
    fout.write(bytes(hdr))
    for name, nr, nc, off, sz in tlist:
        nb = len(name)
        fout.write(struct.pack('<I', nb))
        fout.write(name.encode())
        fout.write(b'\0')
        fout.write(struct.pack('<III', 2, nr, nc))  # ndim=2 + dims
        fout.write(struct.pack('<QQ', off, sz))
    
    # Quantize tensors
    print(f"Quantizing {len(tlist)} tensors to Q4NX...")
    done = 0
    for name, nr, nc, off, sz in tlist:
        done += 1
        if max_t and done > max_t: break
        
        # Find tensor
        ten = None
        for t in rd.tensors:
            if t.name == name: ten = t; break
        if ten is None: continue
        
        # Dequantize to float32
        if ten.tensor_type <= 1:  # F32 or F16
            dt = np.float32 if ten.tensor_type == 0 else np.float16
            w = np.frombuffer(ten.data, dtype=dt).reshape(nr, nc).astype(np.float32)
        else:
            # Use gguf dequantize
            w = dequantize(ten.data, ten.tensor_type).reshape(nr, nc).astype(np.float32)
        
        # Tile and quantize
        ntr = (nr + tr - 1) // tr; ntc = (nc + tc - 1) // tc
        for rr in range(ntr):
            for cc in range(ntc):
                td = w[rr*tr:rr*tr+tr, cc*tc:cc*tc+tc]
                fout.write(quant_tile(td, tr, tc, gs))
        
        if done <= 5 or done % 100 == 0:
            print(f"  [{done}/{len(tlist)}] {name}: {nr}x{nc}")
    
    fout.close()
    mb = os.path.getsize(sys.argv[2]) / 1e6
    print(f"\nDone: {sys.argv[2]} ({mb:.1f} MB)")

if __name__ == '__main__':
    main()
