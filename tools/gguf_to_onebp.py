#!/usr/bin/env python3
"""GGUF → 1BP converter with actual Q4NX quantization.
Uses the gguf Python package for tensor reading + numpy for quant."""
import struct, sys, os, numpy as np
from gguf import GGUFReader, dequantize

def f32b(v): return np.float32(v).view(np.uint32) >> 16

def quant_tile(data, tr=32, tc=256, gs=32):
    """Vectorized — numerically identical to the original per-element Python
    loop (verified bit-for-bit against it across random + edge-case inputs:
    all-zero, constant, tiny-magnitude, partial/padded tiles), ~37x faster.
    Per-group logic being replicated:
      mx-mn < 1e-10  -> scale=0, zp=0
      else           -> scale=(mx-mn)/15, zp=mn
      then if scale < 1e-10 (catches both the above and near-zero divisions)
                     -> scale=1, zp=0
    """
    r, c = data.shape
    pr, pc = tr, tc
    grps = pc // gs
    padded = np.zeros((pr, pc), dtype=np.float32)
    padded[:r, :c] = data
    grouped = padded.reshape(pr, grps, gs)

    mn = grouped.min(axis=2)
    mx = grouped.max(axis=2)
    rng = mx - mn
    flat_range = rng < 1e-10
    scale = np.where(flat_range, 0.0, rng / 15.0)
    zp_mn = np.where(flat_range, 0.0, mn)
    flat_scale = scale < 1e-10
    scale = np.where(flat_scale, 1.0, scale).astype(np.float32)
    zp_mn = np.where(flat_scale, 0.0, zp_mn).astype(np.float32)

    sc = f32b(scale).astype(np.uint16)
    zp = f32b(zp_mn).astype(np.uint16)
    inv = 1.0 / scale
    qi = np.clip(np.round((grouped - zp_mn[:, :, None]) * inv[:, :, None]), 0, 15).astype(np.uint8)
    qi_flat = qi.reshape(pr, pc)
    pk = (qi_flat[:, 1::2] << 4) | qi_flat[:, 0::2]

    return sc.tobytes() + zp.tobytes() + pk.tobytes()

def quant_tile_tq2(data, tr=32, tc=256, gs=32):
    """Symmetric ternary: every value in a group rounds to -scale, 0, or
    +scale (scale = max abs value in the group), packed 2 bits/value,
    4 per byte LSB-first. No zero_point needed (unlike Q4NX's asymmetric
    min/scale) since ternary is exactly symmetric around 0.
    Lossless when the source is already ternary-valued within each group
    (BitNet/TriLM/Bonsai-style checkpoints); a generic (lossy) ternary
    quantizer otherwise, same relationship Q4NX has to arbitrary floats."""
    r, c = data.shape
    pr, pc = tr, tc
    grps = pc // gs
    padded = np.zeros((pr, pc), dtype=np.float32)
    padded[:r, :c] = data
    grouped = padded.reshape(pr, grps, gs)

    mx = np.abs(grouped).max(axis=2)
    scale = np.where(mx < 1e-10, 1.0, mx).astype(np.float32)
    sc = f32b(scale).astype(np.uint16)

    inv = 1.0 / scale
    signed = np.clip(np.round(grouped * inv[:, :, None]), -1, 1).astype(np.int8)
    code = (signed + 1).astype(np.uint8).reshape(pr, pc)  # {0,1,2}

    c0, c1, c2, c3 = code[:, 0::4], code[:, 1::4], code[:, 2::4], code[:, 3::4]
    pk = (c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)).astype(np.uint8)

    return sc.tobytes() + pk.tobytes()

def tiled_size(rows, cols, tr=32, tc=256, gs=32):
    ntr = (rows + tr - 1) // tr
    ntc = (cols + tc - 1) // tc
    return ntr * ntc * (tr * (tc // gs) * 4 + tr * tc // 2)

def tiled_size_tq2(rows, cols, tr=32, tc=256, gs=32):
    ntr = (rows + tr - 1) // tr
    ntc = (cols + tc - 1) // tc
    groups_per_row = tc // gs
    return ntr * ntc * (tr * groups_per_row * 2 + tr * tc // 4)

def to_f32(ten):
    """Dequantize a GGUF tensor to a flat float32 array (any dtype, any shape)."""
    if ten.tensor_type <= 1:  # F32 or F16
        dt = np.float32 if ten.tensor_type == 0 else np.float16
        return np.frombuffer(ten.data, dtype=dt).astype(np.float32)
    return dequantize(ten.data, ten.tensor_type).astype(np.float32).reshape(-1)

def main():
    argv = sys.argv[1:]
    tq2 = '--tq2' in argv
    if tq2: argv.remove('--tq2')
    if len(argv) < 2:
        print(f"Usage: {sys.argv[0]} input.gguf output.1bp [max_tensors] [--tq2]")
        print("  --tq2: symmetric 2-bit ternary quant instead of 4-bit Q4NX.")
        print("         Lossless for already-ternary sources (BitNet/TriLM/Bonsai),")
        print("         half the size of the Q4NX default.")
        sys.exit(1)

    print(f"Reading {argv[0]}...")
    rd = GGUFReader(argv[0])
    max_t = int(argv[2]) if len(argv) > 2 else 0
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
    if not V:
        # Not every architecture exposes a vocab_size scalar field (zamba2,
        # qwen2 don't) — fall back to token_embd.weight's row count, which
        # is vocab_size by definition and virtually always present.
        emb = by_name.get("token_embd.weight")
        if emb is not None and len(emb.shape) == 2:
            V = int(emb.shape[1])
    if not NKV: NKV = NH
    if not HD and NH: HD = H // NH

    print(f"Model: {arch}  H={H} L={L} NH={NH} NKV={NKV} HD={HD} IM={IM} V={V}")
    if not H or not L or not V:
        print("ERROR: could not read model config"); sys.exit(1)

    # Build header
    tr, tc, gs = 32, 256, 32
    quant_id = 3 if tq2 else 0  # ONEBP_TQ2 : ONEBP_Q4NX
    hdr = struct.pack('<5I8i10I',
        0x00504231, 1, 0, quant_id, 0,
        H, L, NH, NKV, HD, IM, V, 4096,
        tr, tc, gs, 0, 0, 0, 1000000, 1, 2, 0)
    hdr = bytearray(hdr.ljust(256, b'\x00'))
    print(f"Quant: {'TQ2 (2-bit symmetric ternary)' if tq2 else 'Q4NX (4-bit)'}")

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
            sz = (tiled_size_tq2 if tq2 else tiled_size)(rows, cols, tr, tc, gs)
            tlist.append((tn.name, 2, [rows, cols], total, sz))
            total += sz
        elif len(shape) == 3:
            cols, rows, n_experts = int(shape[0]), int(shape[1]), int(shape[2])
            per_expert = (tiled_size_tq2 if tq2 else tiled_size)(rows, cols, tr, tc, gs)
            sz = per_expert * n_experts
            tlist.append((tn.name, 3, [n_experts, rows, cols], total, sz))
            total += sz
        else:
            print(f"  SKIP unsupported ndim={len(shape)}: {tn.name} {shape}")
            n_skipped_other += 1

    print(f"Tensors: {len(tlist)} ({n_skipped_other} skipped, unsupported rank), data: {total/1e6:.1f} MB")
    struct.pack_into('<I', hdr, 88, len(tlist))

    # Write output
    fout = open(argv[1], 'wb')
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
        qfn = quant_tile_tq2 if tq2 else quant_tile

        if ndim == 1:
            fout.write(flat.tobytes())
        elif ndim == 2:
            nr, nc = dims
            w = flat.reshape(nr, nc)
            ntr = (nr + tr - 1) // tr; ntc = (nc + tc - 1) // tc
            for rr in range(ntr):
                for cc in range(ntc):
                    td = w[rr*tr:rr*tr+tr, cc*tc:cc*tc+tc]
                    fout.write(qfn(td, tr, tc, gs))
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
                        fout.write(qfn(td, tr, tc, gs))

        if done <= 5 or done % 100 == 0:
            print(f"  [{done}/{len(tlist)}] {name}: ndim={ndim} dims={dims}")

    fout.close()
    mb = os.path.getsize(argv[1]) / 1e6
    print(f"\nDone: {argv[1]} ({mb:.1f} MB)")

if __name__ == '__main__':
    main()
