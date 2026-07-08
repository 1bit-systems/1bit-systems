#!/usr/bin/env python3
"""
Q4NX chunk format writer — produces torch2aie-compatible weight data.

Each weight tensor is stored as 5120-byte rows (32×256 tile):
  [0..511]:   256 BF16 per-row scales (for 8 groups × 32 rows)
  [512..1023]: 256 BF16 zero_points (all zeros for us)
  [1024..5119]: 4096 bytes INT4 packed data

INT4 packing: 2 values per byte (lo nibble = row within lane pair, hi = other)
Lane 0: rows 0-15, Lane 1: rows 16-31
For col c in 0..255, byte offset within lane: c*8 + (row%16)//2
Nibble: row%2==0 → lo nibble, row%2==1 → hi nibble

Input: plain INT8 weights (n_elems values), output: chunked format.
"""
import json, math, struct, sys, os
import numpy as np

TILE_ROWS = 32
TILE_COLS = 256
ROW_BYTES = 5120  # 512 scales + 512 zps + 4096 data


def int8_to_chunk(i8_data: np.ndarray, shape: tuple) -> bytes:
    """Convert INT8 weight matrix → Q4NX chunk format.
    
    Weight is stored as [out_features, in_features].
    Tile grid: n_tile_rows × n_tile_cols, each tile = 32×256.
    """
    out_f, in_f = shape
    n_tile_rows = max(1, math.ceil(out_f / TILE_ROWS))
    n_tile_cols = max(1, math.ceil(in_f / TILE_COLS))
    
    # Pad to tile boundaries
    padded = np.zeros((n_tile_rows * TILE_ROWS, n_tile_cols * TILE_COLS), dtype=np.float32)
    padded[:out_f, :in_f] = i8_data.astype(np.float32)
    
    rows = []
    for tr in range(n_tile_rows):
        for tc in range(n_tile_cols):
            tile = padded[tr*TILE_ROWS:(tr+1)*TILE_ROWS, tc*TILE_COLS:(tc+1)*TILE_COLS]
            row = bytearray(ROW_BYTES)
            
            # BF16 scales: one per row, 8 groups × 32 rows = 256 values
            # For INT8→INT4, group rows by 32, compute max abs per group
            scale_vals = np.zeros(256, dtype=np.float32)
            for g in range(8):
                g_start = g * 32
                g_end = min(g_start + 32, TILE_ROWS)
                if g_start >= TILE_ROWS:
                    break
                group_data = tile[g_start:g_end, :]
                amax = np.max(np.abs(group_data))
                if amax < 1e-10:
                    amax = 1.0
                # Scale for INT4: want values in range [0,15]
                scale = amax / 7.0
                for r in range(g_end - g_start):
                    scale_vals[g * 32 + r] = scale
            
            # Write BF16 scales
            for i, s in enumerate(scale_vals):
                struct.pack_into('<H', row, i * 2, np.float16(s).view(np.uint16))
            
            # Zero points — all zero for symmetric quantization
            # (bytes 512-1023 already zero from bytearray init)
            
            # INT4 packed data (bytes 1024-5119)
            # Quantize tile values to INT4 [0,15]
            for r in range(TILE_ROWS):
                scale = scale_vals[(r // 32) * 32 + r % 32]
                if scale < 1e-10:
                    scale = 1.0
                lane = r // 16
                lane_row = r % 16
                byte_idx = lane_row // 2
                nibble_sel = r % 2  # 0=lo, 1=hi
                
                lane_base = 1024 + lane * (TILE_COLS * 8)
                
                for c in range(TILE_COLS):
                    val = tile[r, c]
                    # Quantize to INT4 [0, 15]
                    code = int(np.clip(np.round(val / scale) + 8, 0, 15))
                    
                    addr = lane_base + c * 8 + byte_idx
                    if nibble_sel == 0:
                        row[addr] = (row[addr] & 0xF0) | (code & 0x0F)
                    else:
                        row[addr] = (row[addr] & 0x0F) | ((code & 0x0F) << 4)
            
            rows.append(bytes(row))
    
    return b''.join(rows)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} model.q4nx output_chunked.q4nx")
        print("Converts plain INT8 Q4NX to torch2aie chunk format")
        sys.exit(1)
    
    src, dst = sys.argv[1], sys.argv[2]
    
    with open(src, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len
    
    new_header = {}
    data_blocks = []
    data_offset = 0
    
    for name, info in hdr.items():
        if info.get('dtype') == 'I8':
            shape = info['shape']
            n_elems = math.prod(shape)
            
            off = data_start + info['data_offsets'][0]
            size = info['data_offsets'][1] - info['data_offsets'][0]
            
            with open(src, 'rb') as f:
                f.seek(off)
                raw = f.read(size)
            
            i8_vals = np.frombuffer(raw, dtype=np.int8)
            
            # Convert to chunk format
            chunk_data = int8_to_chunk(i8_vals.reshape(shape), shape)
            
            new_header[name] = {
                'dtype': 'I8',
                'shape': list(shape),
                'data_offsets': [data_offset, data_offset + len(chunk_data)],
            }
            data_blocks.append(chunk_data)
            data_offset += len(chunk_data)
            
            print(f"  {name}: {n_elems} elems → {len(chunk_data)} bytes chunked")
        
        elif info.get('dtype') in ('F32', 'F16'):
            # Pass through F32/F16 as-is
            off = data_start + info['data_offsets'][0]
            size = info['data_offsets'][1] - info['data_offsets'][0]
            with open(src, 'rb') as f:
                f.seek(off)
                raw = f.read(size)
            
            new_header[name] = {
                'dtype': info['dtype'],
                'shape': info['shape'],
                'data_offsets': [data_offset, data_offset + len(raw)],
            }
            data_blocks.append(raw)
            data_offset += len(raw)
    
    # Write output
    hdr_json = json.dumps(new_header, separators=(',', ':'))
    hdr_bytes = hdr_json.encode()
    
    with open(dst, 'wb') as out:
        out.write(struct.pack('<Q', len(hdr_bytes)))
        out.write(hdr_bytes)
        for block in data_blocks:
            out.write(block)
    
    print(f"\nWrote {len(new_header)} tensors → {dst}")
    print(f"Size: {os.path.getsize(dst)/1e6:.1f} MB")


if __name__ == '__main__':
    main()
