#!/usr/bin/env python3
"""Convert GGUF models to Q4NX format for engine_peak benchmarking.

Usage: ./convert_gguf_to_q4nx.py model.gguf output.q4nx
"""

import json, struct, sys, numpy as np
gguf_available = False
try:
    import gguf
    gguf_available = True
except:
    pass

TILE_R = 32
TILE_C = 256
I8_ROW_B = 5120  # 1024 scale/zp + 4096 packed 4-bit data


def dequantize_tensor(tensor):
    """Dequantize a GGUF tensor to FP32 numpy array."""
    data = tensor.data
    dtype = tensor.tensor_type
    shape = tensor.shape
    n_elems = int(np.prod(shape))
    
    # F32
    if dtype == 0:
        return data.view(np.float32).copy().reshape(shape)
    # F16
    if dtype == 1:
        return data.view(np.float16).astype(np.float32).reshape(shape)
    # Q4_0
    if dtype == 10:
        bs = 32; bpb = 18
        result = np.zeros(n_elems, dtype=np.float32)
        n_blocks = len(data) // bpb
        for b in range(n_blocks):
            off = b * bpb
            d = data[off:off+2].view(np.float16)[0]
            qs = data[off+2:off+bpb]
            for i in range(bs):
                code = (qs[i//2] >> (4*(i%2))) & 0xF
                val = (code - 8) * d
                if b*bs + i < n_elems:
                    result[b*bs + i] = val
        return result.reshape(shape)
    # Q4_1
    if dtype == 11:
        bs = 32; bpb = 20
        result = np.zeros(n_elems, dtype=np.float32)
        n_blocks = len(data) // bpb
        for b in range(n_blocks):
            off = b * bpb
            d = data[off:off+2].view(np.float16)[0]
            m = data[off+2:off+4].view(np.float16)[0]
            qs = data[off+4:off+bpb]
            for i in range(bs):
                code = (qs[i//2] >> (4*(i%2))) & 0xF
                val = code * d + m
                if b*bs + i < n_elems:
                    result[b*bs + i] = val
        return result.reshape(shape)
    # Q1_0 (type 41) — 32 elems, 5 bytes/block
    if dtype == 41:
        bs = 32; bpb = 5
        result = np.zeros(n_elems, dtype=np.float32)
        n_blocks = len(data) // bpb
        for b in range(n_blocks):
            off = b * bpb
            d = data[off:off+2].view(np.float16)[0]
            qs = data[off+2:off+bpb]
            for i in range(bs):
                code = (qs[i//2] >> (4*(i%2))) & 0xF
                val = (code - 8) * d
                if b*bs + i < n_elems:
                    result[b*bs + i] = val
        return result.reshape(shape)
    # Q2_0 / stq1_0 (type 42)
    if dtype == 42:
        # Try Q2_0 format: 128 elems, 34 bytes/block
        bs_q2 = 128; bpb_q2 = 34
        n_blocks_q2 = n_elems // bs_q2
        result = np.zeros(n_elems, dtype=np.float32)
        for b in range(min(n_blocks_q2, len(data)//bpb_q2)):
            off = b * bpb_q2
            d = data[off:off+2].view(np.float16)[0]
            qs = data[off+2:off+bpb_q2]
            for i in range(bs_q2):
                code = (qs[i//4] >> (2*(i%4))) & 0x3
                val = (code - 1) * d
                if b*bs_q2 + i < n_elems:
                    result[b*bs_q2 + i] = val
        return result.reshape(shape)
    # Q8_0
    if dtype == 8:
        bs = 32; bpb = 34
        result = np.zeros(n_elems, dtype=np.float32)
        n_blocks = len(data) // bpb
        for b in range(n_blocks):
            off = b * bpb
            d = data[off:off+2].view(np.float16)[0]
            qs = data[off+2:off+bpb]
            for i in range(bs):
                val = (int(qs[i]) - 128) * d
                if b*bs + i < n_elems:
                    result[b*bs + i] = val
        return result.reshape(shape)
    # Fallback: raw bytes as FP16
    print(f"  WARN: unknown dtype {dtype}, reading as FP16")
    return data[:n_elems*2].view(np.float16).astype(np.float32).reshape(shape)


def float_to_bf16(f):
    return struct.unpack('<I', struct.pack('<f', float(f)))[0] >> 16


def pack_to_i8_tiles(w_2d):
    """Pack FP32 weight matrix into I8 tiled format."""
    rows, cols = w_2d.shape
    ntr = (rows + TILE_R - 1) // TILE_R
    ntc = (cols + TILE_C - 1) // TILE_C
    tile_data = bytearray()
    
    for tr in range(ntr):
        for tc in range(ntc):
            tile = bytearray(I8_ROW_B)
            for lr in range(TILE_R):
                row_idx = tr * TILE_R + lr
                if row_idx >= rows:
                    continue
                lane = lr // 16
                lr2 = lr % 16
                bi = lr2 // 2
                ns = lr % 2
                for g in range(TILE_C // 32):
                    col_start = tc * TILE_C + g * 32
                    vals = []
                    for c in range(col_start, min(col_start + 32, cols)):
                        vals.append(w_2d[row_idx, c] if c < cols else 0.0)
                    while len(vals) < 32:
                        vals.append(0.0)
                    vals = np.array(vals)
                    vmin, vmax = vals.min(), vals.max()
                    if vmax - vmin < 1e-10:
                        scale = 1.0; zero = 0.0
                    else:
                        scale = (vmax - vmin) / 15.0; zero = vmin
                    sc_idx = g * 32 + lr
                    struct.pack_into('<H', tile, sc_idx * 2, float_to_bf16(scale))
                    struct.pack_into('<H', tile, 512 + sc_idx * 2, float_to_bf16(zero))
                    for ci, c in enumerate(range(col_start, min(col_start + 32, cols))):
                        v = w_2d[row_idx, c] if c < cols and row_idx < rows else 0.0
                        q = int(round((v - zero) / scale)) if scale > 1e-10 else 0
                        q = max(0, min(15, q))
                        pk_off = 1024 + lane * (TILE_C * 8) + ci * 8 + bi
                        if pk_off < I8_ROW_B:
                            tile[pk_off] = (tile[pk_off] & (0xF0 if ns else 0x0F)) | (q << 4 if ns else q)
            tile_data.extend(tile)
    return bytes(tile_data), ntr, ntc


def main():
    if len(sys.argv) < 3:
        print("Usage: convert_gguf_to_q4nx.py model.gguf output.q4nx")
        sys.exit(1)
    
    gguf_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Loading {gguf_path}...")
    if not gguf_available:
        print("Install gguf: pip install gguf numpy")
        sys.exit(1)
    
    reader = gguf.GGUFReader(gguf_path)
    
    # Read metadata
    n_embd = 0; n_head = 0; n_layer = 0; n_ff = 0; n_kv_head = 0
    for k, v in reader.fields.items():
        if k == 'general.architecture':
            arch = str(v.parts[-1].tobytes().decode('utf-8').strip('\x00'))
        elif k.endswith('embedding_length'):
            n_embd = int(v.parts[-1][0])
        elif k.endswith('attention.head_count'):
            n_head = int(v.parts[-1][0])
        elif k.endswith('attention.head_count_kv'):
            n_kv_head = int(v.parts[-1][0]) if v.parts[-1][0] > 0 else n_head
        elif k.endswith('block_count'):
            n_layer = int(v.parts[-1][0])
        elif k.endswith('feed_forward_length'):
            n_ff = int(v.parts[-1][0])
    
    n_kv_head = n_kv_head or n_head
    print(f"  Model: {arch}, {n_layer} layers, {n_embd} dim, {n_head} heads, {n_ff} FFN")
    
    # Process tensors
    tile_data = bytearray()
    manifest = {}
    current_offset = 0
    hd = n_embd // n_head if n_head > 0 else 128
    im = n_ff or n_embd * 4
    
    for t in reader.tensors:
        name = t.name
        shape = list(t.shape)
        if not shape:
            continue
        
        print(f"  {name}: {shape} dtype={t.tensor_type}...", end=' ')
        
        try:
            w = dequantize_tensor(t)
        except Exception as e:
            print(f"ERROR: {e}")
            continue
        
        if len(shape) < 2:
            # 1D: norms, etc — store as FP32
            data = w.astype(np.float32).tobytes()
            manifest[name] = {"shape": shape, "data_offsets": [current_offset, current_offset + len(data)], "dtype": "f32"}
            tile_data.extend(data)
            current_offset += len(data)
            print(f"FP32 {len(data)} bytes")
        else:
            # 2D: weight matrices — pack as I8 tiles
            w_2d = w.reshape(shape[0], shape[1]) if len(shape) >= 2 else w.reshape(1, -1)
            tiles, i8_r, i8_c = pack_to_i8_tiles(w_2d)
            manifest[name] = {
                "shape": shape, "data_offsets": [current_offset, current_offset + len(tiles)],
                "dtype": "i8_tiled", "i8_rows": i8_r, "i8_inner": shape[1]
            }
            tile_data.extend(tiles)
            current_offset += len(tiles)
            print(f"I8 tiles {len(tiles)//1024}KB ({i8_r}×{i8_c})")
    
    # Write Q4NX
    manifest_str = json.dumps(manifest)
    header = struct.pack('<Q', len(manifest_str))
    
    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(manifest_str.encode('utf-8'))
        f.write(bytes(tile_data))
    
    total = len(header) + len(manifest_str) + len(tile_data)
    print(f"\nWritten {output_path} ({total} bytes, {total//1024//1024} MB)")
    print(f"  Tensors: {len(manifest)}")


if __name__ == '__main__':
    main()
