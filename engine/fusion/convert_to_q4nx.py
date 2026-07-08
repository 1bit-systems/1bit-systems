#!/usr/bin/env python3
"""Convert any GGUF model to Q4NX for engine_peak benchmarking.

Usage: python3 convert_to_q4nx.py model.gguf output.q4nx
"""
import sys, json, struct, numpy as np

# Import gguf from the venv
import gguf

TILE_R = 32
TILE_C = 256
I8_ROW_B = 5120  # 1024 scale/zp + 4096 packed 4-bit data

def bf16(f):
    return struct.unpack('<I', struct.pack('<f', float(f)))[0] >> 16

def pack_i8_tile(w_2d):
    """Pack a 2D weight matrix into I8 tiled format."""
    rows, cols = w_2d.shape
    ntr = (rows + TILE_R - 1) // TILE_R
    ntc = (cols + TILE_C - 1) // TILE_C
    data = bytearray()
    
    for tr in range(ntr):
        for tc in range(ntc):
            tile = bytearray(I8_ROW_B)
            for lr in range(TILE_R):
                ri = tr * TILE_R + lr
                if ri >= rows: continue
                lane = lr // 16; lr2 = lr % 16; bi = lr2 // 2; ns = lr % 2
                
                for g in range(TILE_C // 32):
                    cs = tc * TILE_C + g * 32
                    vals = [w_2d[ri, c] if ri < rows and c < cols else 0.0 for c in range(cs, min(cs+32, cols))]
                    while len(vals) < 32: vals.append(0.0)
                    va = np.array(vals)
                    vmin, vmax = float(va.min()), float(va.max())
                    
                    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin < 1e-10:
                        sc, zr = 1.0, 0.0
                    else:
                        sc = (vmax - vmin) / 15.0; zr = vmin
                    
                    si = g * 32 + lr
                    if np.isfinite(sc) and np.isfinite(zr):
                        struct.pack_into('<H', tile, si * 2, bf16(sc))
                        struct.pack_into('<H', tile, 512 + si * 2, bf16(zr))
                    
                    for ci in range(min(32, cols - cs)):
                        c = cs + ci
                        v = float(w_2d[ri, c]) if ri < rows and c < cols else 0.0
                        if not np.isfinite(v): v = 0.0
                        if sc > 1e-10:
                            qv = (v - zr) / sc
                            q = max(0, min(15, int(round(qv))))
                        else:
                            q = 0
                        po = 1024 + lane * (TILE_C * 8) + ci * 8 + bi
                        if po < I8_ROW_B:
                            tile[po] = (tile[po] & (0xF0 if ns else 0x0F)) | (q << 4 if ns else q)
                        po = 1024 + lane * (TILE_C * 8) + ci * 8 + bi
                        if po < I8_ROW_B:
                            tile[po] = (tile[po] & (0xF0 if ns else 0x0F)) | (q << 4 if ns else q)
            data.extend(tile)
    return bytes(data), ntr


def dequant_tensor(t):
    """Dequantize a gguf tensor to FP32."""
    dt = int(t.tensor_type)
    shape = [int(s) for s in t.shape]
    n_elems = int(np.prod(shape))
    
    # F32
    if dt == 0:
        return t.data.view(np.float32).copy().reshape(shape)
    # F16
    if dt == 1:
        return t.data.view(np.float16).astype(np.float32).reshape(shape)
    
    # For quantized types, use gguf's block info
    raw = t.data.reshape(-1).tobytes()
    
    # Q1_0 (dtype 41): try 5 or 6 bytes/block
    if dt == 41:
        bs = 32
        for bpb in [6, 5]:
            if len(raw) % bpb == 0:
                nb = len(raw) // bpb
                result = np.zeros(n_elems, dtype=np.float32)
                for b in range(nb):
                    off = b * bpb
                    d = np.frombuffer(raw[off:off+2], dtype=np.float16)[0]
                    for i in range(min(bs, n_elems - b*bs)):
                        if len(raw) > off + 2 + i//8:
                            bit = (raw[off+2+i//8] >> (i % 8)) & 1
                            result[b*bs + i] = (bit * 2 - 1) * d
                if np.any(np.isfinite(result)):
                    return result.reshape(shape)
    
    # Q4_0 (dtype 2): 32 elems/block, 18 bytes/block
    if dt == 2:
        bs, bpb = 32, 18
        nb = len(raw) // bpb
        result = np.zeros(n_elems, dtype=np.float32)
        for b in range(nb):
            off = b * bpb
            d = np.frombuffer(raw[off:off+2], dtype=np.float16)[0]
            for i in range(bs):
                code = (raw[off+2+i//2] >> (4*(i%2))) & 0xF
                if b*bs + i < n_elems:
                    result[b*bs + i] = (code - 8) * d
        return result.reshape(shape)
    
    # Q8_0 (dtype 8): 32 elems/block, 34 bytes/block
    if dt == 8:
        bs, bpb = 32, 34
        nb = len(raw) // bpb
        result = np.zeros(n_elems, dtype=np.float32)
        for b in range(nb):
            off = b * bpb
            d = np.frombuffer(raw[off:off+2], dtype=np.float16)[0]
            for i in range(bs):
                if b*bs + i < n_elems:
                    result[b*bs + i] = (float(np.int8(raw[off+2+i])) - 128) * d
        return result.reshape(shape)
    
    # Generic fallback: read as raw
    print(f"  WARN: unknown dtype {dt}, reading {len(raw)} bytes")
    return np.frombuffer(raw[:n_elems*4], dtype=np.float32).reshape(shape)


def main():
    if len(sys.argv) < 3:
        print("Usage: convert_to_q4nx.py model.gguf output.q4nx")
        sys.exit(1)
    
    reader = gguf.GGUFReader(sys.argv[1])
    
    # Get model dims from metadata
    n_layer = 0
    for k, v in reader.fields.items():
        kn = k.split('.')[-1]
        if kn == 'block_count': n_layer = int(v.parts[-1][0])
    
    manifest = {}
    tiledata = bytearray()
    offset = 0
    
    for t in reader.tensors:
        name = str(t.name)
        shape = [int(s) for s in t.shape]
        print(f"  {name}: {shape} dtype={t.tensor_type}...", end=' ', flush=True)
        
        try:
            w = dequant_tensor(t)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        
        if len(shape) == 1:
            # Norm/bias: store as FP32
            data = w.astype(np.float32).tobytes()
            manifest[name] = {"shape": shape, "data_offsets": [offset, offset+len(data)], "dtype": "f32"}
            tiledata.extend(data)
            offset += len(data)
            print(f"{len(data)}B FP32")
        else:
            # Weight matrix: pack as I8 tiles
            w_2d = w.reshape(shape[0], -1)
            tiles, i8r = pack_i8_tile(w_2d)
            manifest[name] = {
                "shape": shape, "data_offsets": [offset, offset+len(tiles)],
                "dtype": "i8_tiled", "i8_rows": i8r, "i8_inner": shape[1]
            }
            tiledata.extend(tiles)
            offset += len(tiles)
            print(f"{len(tiles)//1024}KB I8 tiles")
    
    # Write
    mjson = json.dumps(manifest)
    with open(sys.argv[2], 'wb') as f:
        f.write(struct.pack('<Q', len(mjson)))
        f.write(mjson.encode('utf-8'))
        f.write(bytes(tiledata))
    
    print(f"\nWrote {sys.argv[2]} ({offset//1024//1024} MB)")


if __name__ == '__main__':
    main()
