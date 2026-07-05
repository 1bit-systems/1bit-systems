#!/usr/bin/env python3
"""Q4NX I8 dequantization — matches dequant_q4nx.c byte-for-byte."""
import struct, math
import numpy as np

def bf16_to_f32(v: int) -> float:
    bits = v << 16
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def dequant_i8_chunk(data: bytes, i8_rows: int, in_features: int = 1024) -> np.ndarray:
    """
    Dequantize I8 Q4NX weight tensor, matching dequant_q4nx.c exactly.
    
    Args:
        data: Raw I8 bytes (i8_rows * 5120 bytes)
        i8_rows: Number of I8 rows in the tensor
        in_features: Model hidden dimension (controls tile grid)
    
    Returns:
        float64 array [out_rows, out_cols] = [n_tile_rows*32, n_tile_cols*256]
    """
    TILE_ROWS, TILE_COLS = 32, 256
    n_tile_cols = in_features // TILE_COLS
    n_tile_rows = i8_rows // n_tile_cols
    out_rows = n_tile_rows * TILE_ROWS
    out_cols = n_tile_cols * TILE_COLS
    
    raw = np.frombuffer(data, dtype=np.uint8).reshape(i8_rows, 5120)
    out = np.zeros((out_rows, out_cols), dtype=np.float64)
    
    for ir in range(i8_rows):
        row = raw[ir]
        tile_r = ir // n_tile_cols
        tile_c = ir % n_tile_cols
        
        # Parse scales/zeros as BF16
        scales_u16 = row[0:512].view(np.uint16)
        zeros_u16 = row[512:1024].view(np.uint16)
        scales_f32 = (scales_u16.astype(np.uint32) << 16).view(np.float32).reshape(256)
        zeros_f32 = (zeros_u16.astype(np.uint32) << 16).view(np.float32).reshape(256)
        
        # Match C dequant: if (!isfinite(s) || fabsf(s) > 100.0f) s = 0.0f
        bad_s = ~(np.isfinite(scales_f32)) | (np.abs(scales_f32) > 100.0)
        bad_z = ~(np.isfinite(zeros_f32)) | (np.abs(zeros_f32) > 100.0)
        scales_f32[bad_s] = 0.0
        zeros_f32[bad_z] = 0.0
        
        packed = row[1024:].reshape(2, 256, 8)
        
        for lr in range(TILE_ROWS):
            lane = lr // 16
            lane_row = lr % 16
            byte_idx = lane_row // 2
            nibble_sel = lr % 2
            
            lane_bytes = packed[lane, :, byte_idx]
            if nibble_sel == 0:
                vals = (lane_bytes & 0x0F).astype(np.int8)
            else:
                vals = ((lane_bytes >> 4) & 0x0F).astype(np.int8)
            vals = np.where(vals >= 8, vals - 16, vals).astype(np.float64)
            
            group = np.arange(TILE_COLS) // 32
            s = scales_f32[group * 32 + lr]
            z = zeros_f32[group * 32 + lr]
            
            out[tile_r * TILE_ROWS + lr, tile_c * TILE_COLS:(tile_c + 1) * TILE_COLS] = vals * s + z
    
    return out
