#!/usr/bin/env python3
"""
Q1_0 (1-bit binary) GGUF → packed ternary flat buffers for NPU kernel.
Converts Q1_0 to mm_ternary_32x64x128 format: [weights (4×2bit/byte)] [BF16 scales]

Usage: python3 q1_0_to_packed.py model.gguf output_dir/ [tensor_name]
"""
import struct, json, os, sys, math

QK = 128
BLOCK_BYTES = 18

def parse_gguf(path):
    """Reuse from q2_0_decode.py."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'tools'))
    from q2_0_decode import parse_gguf as pg
    return pg(path)

def dequant_q1_0(raw, n_elems):
    """Q1_0 → f32 list (pure Python)."""
    nb = n_elems // QK
    raw = raw[:nb * BLOCK_BYTES]
    result = []
    for b in range(nb):
        off = b * BLOCK_BYTES
        scale_bits = struct.unpack('<H', raw[off:off+2])[0]
        bits32 = scale_bits << 16
        scale = struct.unpack('<f', struct.pack('<I', bits32))[0]
        if math.isnan(scale) or math.isinf(scale): scale = 0.0
        for byte_idx in range(16):
            byte = raw[off + 2 + byte_idx]
            for bit in range(8):
                result.append(scale if (byte >> bit) & 1 else -scale)
                if len(result) >= n_elems: return result
    return result[:n_elems]

def pack_ternary(f32_row):
    """f32 list → (packed_bytes, bf16_scale)."""
    max_val = max((abs(v) for v in f32_row), default=1.0)
    if max_val < 1e-10: max_val = 1.0
    threshold = max_val * 0.3
    codes = []
    for v in f32_row:
        if v > threshold: codes.append(2)        # +1
        elif v < -threshold: codes.append(0)      # -1
        else: codes.append(1)                      # 0
    while len(codes) % 4: codes.append(1)
    packed = bytes(
        (codes[i]&3) | ((codes[i+1]&3)<<2) | ((codes[i+2]&3)<<4) | ((codes[i+3]&3)<<6)
        for i in range(0, len(codes), 4)
    )
    # f32 → bf16
    f32_bits = struct.unpack('<I', struct.pack('<f', max_val))[0]
    rounding = ((f32_bits >> 16) & 1) + 0x7FFF
    bf16_val = ((f32_bits + rounding) >> 16) & 0xFFFF
    return packed, bf16_val

def convert_tensor(gguf_path, tensor_name, output_dir):
    ti, ds, f = parse_gguf(gguf_path)
    if tensor_name not in ti:
        print(f"'{tensor_name}' not found. Q1_0 tensors:")
        for n,(d,qt,off) in sorted(ti.items()):
            if qt==41: print(f"  {n}: {d}")
        return None
    dims, qt, offset = ti[tensor_name]
    if qt != 41: print(f"ERROR: type {qt}, not Q1_0"); return None
    M, K = dims[0], dims[1] if len(dims)>1 else dims[0]
    n_elems = M * K
    f.seek(ds + offset)
    raw = f.read((n_elems // QK) * BLOCK_BYTES)
    print(f"{tensor_name}: M={M} K={K} elements={n_elems}")
    f32_all = dequant_q1_0(raw, n_elems)
    all_packed, all_scales = [], []
    for r in range(M):
        p, s = pack_ternary(f32_all[r*K:(r+1)*K])
        all_packed.append(p); all_scales.append(s)
    K_packed = len(all_packed[0])
    os.makedirs(output_dir, exist_ok=True)
    name = tensor_name.replace('/','_').replace('.','_')
    path = os.path.join(output_dir, f"{name}.bin")
    with open(path,'wb') as out:
        for p in all_packed: out.write(p)
        for s in all_scales: out.write(struct.pack('<H', s))
    sz = os.path.getsize(path)
    print(f"  → {path} ({sz/1024:.1f} KB, K_packed={K_packed}B/row, scale=0x{all_scales[0]:04x})")
    return {'name':tensor_name,'M':M,'K':K,'K_packed':K_packed,
            'K_ternary':K_packed*4,'weight_bytes':M*K_packed,
            'scale_bytes':M*2,'file':os.path.basename(path)}

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: q1_0_to_packed.py <model.gguf> <output_dir> [tensor]")
        sys.exit(1)
    gguf_path, output_dir = sys.argv[1], sys.argv[2]
    tensor = sys.argv[3] if len(sys.argv) > 3 else None
    if tensor:
        convert_tensor(gguf_path, tensor, output_dir)
    else:
        ti, ds, f = parse_gguf(gguf_path)
        for n,(d,qt,off) in sorted(ti.items()):
            if qt == 41:
                convert_tensor(gguf_path, n, output_dir)
