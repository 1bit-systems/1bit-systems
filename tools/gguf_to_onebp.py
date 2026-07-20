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

def tiled_size(rows, cols, tr=32, tc=256, gs=32):
    ntr = (rows + tr - 1) // tr
    ntc = (cols + tc - 1) // tc
    return ntr * ntc * (tr * (tc // gs) * 4 + tr * tc // 2)

def to_f32(ten):
    """Dequantize a GGUF tensor to a flat float32 array (any dtype, any shape)."""
    if ten.tensor_type <= 1:  # F32 or F16
        dt = np.float32 if ten.tensor_type == 0 else np.float16
        return np.frombuffer(ten.data, dtype=dt).astype(np.float32)
    return dequantize(ten.data, ten.tensor_type).astype(np.float32).reshape(-1)

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.gguf output.1bp [max_tensors]"); sys.exit(1)

    print(f"Reading {sys.argv[1]}...")
    rd = GGUFReader(sys.argv[1])
    max_t = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    by_name = {t.name: t for t in rd.tensors}

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

    # Collect tensors — ndim 1 (norms/biases, stored raw), 2 (weight
    # matrices, tile-quantized), and 3 (MoE expert stacks: [num_experts,
    # rows, cols], each expert slice tile-quantized independently).
    tlist = []  # (name, ndim, dims:list[int], file_off, byte_size)
    total = 0
    n_skipped_other = 0
    for tn in rd.tensors:
        shape = tn.shape
        if len(shape) == 1:
            length = int(shape[0])
            sz = length * 4  # raw f32, no tiling — tiny, precision-sensitive
            tlist.append((tn.name, 1, [length], total, sz))
            total += sz
        elif len(shape) == 2:
            rows, cols = int(shape[1]), int(shape[0])
            sz = tiled_size(rows, cols, tr, tc, gs)
            tlist.append((tn.name, 2, [rows, cols], total, sz))
            total += sz
        elif len(shape) == 3:
            cols, rows, n_experts = int(shape[0]), int(shape[1]), int(shape[2])
            per_expert = tiled_size(rows, cols, tr, tc, gs)
            sz = per_expert * n_experts
            tlist.append((tn.name, 3, [n_experts, rows, cols], total, sz))
            total += sz
        else:
            print(f"  SKIP unsupported ndim={len(shape)}: {tn.name} {shape}")
            n_skipped_other += 1

    print(f"Tensors: {len(tlist)} ({n_skipped_other} skipped, unsupported rank), data: {total/1e6:.1f} MB")
    struct.pack_into('<I', hdr, 88, len(tlist))

    # Write output
    fout = open(sys.argv[2], 'wb')
    fout.write(bytes(hdr))
    for name, ndim, dims, off, sz in tlist:
        nb = len(name)
        fout.write(struct.pack('<I', nb))
        fout.write(name.encode())
        fout.write(b'\0')
        fout.write(struct.pack('<I', ndim))
        fout.write(struct.pack(f'<{ndim}I', *dims))
        fout.write(struct.pack('<QQ', off, sz))

    # Quantize/write tensor data
    print(f"Writing {len(tlist)} tensors...")
    done = 0
    for name, ndim, dims, off, sz in tlist:
        done += 1
        if max_t and done > max_t: break

        ten = by_name.get(name)
        if ten is None: continue
        flat = to_f32(ten)

        if ndim == 1:
            fout.write(flat.tobytes())
        elif ndim == 2:
            nr, nc = dims
            w = flat.reshape(nr, nc)
            ntr = (nr + tr - 1) // tr; ntc = (nc + tc - 1) // tc
            for rr in range(ntr):
                for cc in range(ntc):
                    td = w[rr*tr:rr*tr+tr, cc*tc:cc*tc+tc]
                    fout.write(quant_tile(td, tr, tc, gs))
        elif ndim == 3:
            ne, nr, nc = dims
            # GGUF stores expert-stacked tensors as (cols, rows, n_experts)
            # i.e. flat is ordered [cols fastest, then rows, then experts]
            w = flat.reshape(ne, nr, nc)
            ntr = (nr + tr - 1) // tr; ntc = (nc + tc - 1) // tc
            for e in range(ne):
                we = w[e]
                for rr in range(ntr):
                    for cc in range(ntc):
                        td = we[rr*tr:rr*tr+tr, cc*tc:cc*tc+tc]
                        fout.write(quant_tile(td, tr, tc, gs))

        if done <= 5 or done % 100 == 0:
            print(f"  [{done}/{len(tlist)}] {name}: ndim={ndim} dims={dims}")

    fout.close()
    mb = os.path.getsize(sys.argv[2]) / 1e6
    print(f"\nDone: {sys.argv[2]} ({mb:.1f} MB)")

if __name__ == '__main__':
    main()
