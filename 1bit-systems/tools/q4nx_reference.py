"""
Q4NX model loader — dequantize Qwen3-0.6B weights from NPU2 Q4NX format.

Usage:
    from q4nx_reference import *
    list_tensors()
    w = dequantize_weight(offset, i8_rows=384, in_features=1024)
    v = read_bf16(offset, n=1024)
"""

import json
import mmap
import os
import struct

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.expanduser("~/.config/flm/models/Qwen3-0.6B-NPU2")
MODEL_PATH = os.path.join(MODEL_DIR, "model.q4nx")

# ── globals (lazy-loaded) ──────────────────────────────────────────────
_header = None          # parsed JSON dict
_header_bytes = 0       # length of the JSON header (raw bytes)
_mm = None              # mmap object
_file = None            # backing file handle

TILE_BYTES = 5120       # bytes per Q4NX tile
LANE_BYTES = 2048       # bytes per lane (half-tile, 16 rows)
COLS_PER_TILE = 256     # input columns per tile
ROWS_PER_TILE = 32      # output rows per tile
GROUP_SIZE = 32         # columns per quantization group


# ── helpers ────────────────────────────────────────────────────────────

def _ensure_loaded():
    """Lazy-open the model file and parse the JSON header."""
    global _header, _header_bytes, _mm, _file
    if _header is not None:
        return
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    _file = open(MODEL_PATH, "rb")
    _mm = mmap.mmap(_file.fileno(), 0, access=mmap.ACCESS_READ)
    hsz = struct.unpack_from("<Q", _mm, 0)[0]
    header_bytes = _mm[8:8 + hsz]
    _header_bytes = 8 + hsz  # full header: size field + json bytes
    _header = json.loads(header_bytes.decode("utf-8"))


def bf16_to_f32(bits):
    """Convert BF16 uint16(s) to float32.  NaN/Inf → 0.0.

    Accepts a scalar *uint16* or a numpy array of dtype uint16.
    """
    if isinstance(bits, np.ndarray):
        i32 = bits.astype(np.uint32) << np.uint32(16)
        nan_mask = (bits & 0x7F80) == 0x7F80
        out = i32.view(np.float32)
        out[nan_mask] = 0.0
        return out
    # scalar
    if bits & 0x7F80 == 0x7F80:
        return 0.0
    u32 = np.uint32(bits) << np.uint32(16)
    return struct.unpack("<f", struct.pack("<I", int(u32)))[0]


# ── public API ─────────────────────────────────────────────────────────

def get_header():
    """Return the parsed JSON header dict (tensor metadata)."""
    _ensure_loaded()
    return _header


def list_tensors():
    """Print every tensor name, shape, dtype, and byte offset range."""
    _ensure_loaded()
    print(f"{'Tensor name':<55} {'Shape':<20} {'Dtype':<6} {'Offset':<12} {'Size':<12}")
    print("-" * 115)
    for name, info in sorted(_header.items()):
        off_start, off_end = info["data_offsets"]
        sz = off_end - off_start
        shape_str = str(info["shape"])
        print(f"{name:<55} {shape_str:<20} {info['dtype']:<6} {off_start:<12} {sz:<12}")


def read_bf16(offset, n):
    """Read *n* BF16 values starting at *offset* (into the data section),
    return them as a float32 numpy array of shape ``(n,)``."""
    _ensure_loaded()
    raw = np.frombuffer(
        _mm, dtype=np.uint16, count=n, offset=_header_bytes + offset
    )
    return bf16_to_f32(raw)


def _raw_file_offset(offset):
    """Convert a data-section offset (from header data_offsets) to a raw file offset."""
    _ensure_loaded()
    return _header_bytes + offset


def dequantize_weight(offset, i8_rows, in_features):
    """Dequantize a Q4NX I8 tensor into a float32 weight matrix.

    Parameters
    ----------
    offset : int
        Byte offset into the data section (from header ``data_offsets[0]``).
    i8_rows : int
        Number of Q4NX tiles (first dimension of the I8 tensor in the header).
    in_features : int
        Number of input features (columns of the logical weight matrix).

    Returns
    -------
    ndarray of shape ``(out_features, in_features)`` with dtype float32.
    """
    _ensure_loaded()
    n_tile_cols = in_features // COLS_PER_TILE
    out_features = (i8_rows // n_tile_cols) * ROWS_PER_TILE

    result = np.zeros((out_features, in_features), dtype=np.float32)

    raw_off = _raw_file_offset(offset)

    for tile_idx in range(i8_rows):
        tile_off = raw_off + tile_idx * TILE_BYTES
        tile_row = tile_idx // n_tile_cols
        tile_col = tile_idx % n_tile_cols

        # ── scales (256 BF16) ──────────────────────────────────────────
        scales_raw = np.frombuffer(
            _mm, dtype=np.uint16, count=256, offset=tile_off
        )
        scales = bf16_to_f32(scales_raw)  # shape (256,)  idx = group*32 + row

        # ── zero_points (256 BF16) ─────────────────────────────────────
        zps_raw = np.frombuffer(
            _mm, dtype=np.uint16, count=256, offset=tile_off + 512
        )
        zps = bf16_to_f32(zps_raw)  # shape (256,)

        # ── INT4 data ──────────────────────────────────────────────────
        int4_raw = np.frombuffer(
            _mm, dtype=np.uint8, count=4096, offset=tile_off + 1024
        )

        # Decode both lanes
        for lane in range(2):
            lane_base = lane * LANE_BYTES  # 0 or 2048
            for local_row_in_lane in range(16):
                local_row = lane * 16 + local_row_in_lane
                global_row = tile_row * ROWS_PER_TILE + local_row
                byte_idx = local_row_in_lane // 2
                is_lo = (local_row_in_lane % 2) == 0

                for local_col in range(COLS_PER_TILE):
                    addr = lane_base + local_col * 8 + byte_idx
                    byte_val = int(int4_raw[addr])

                    if is_lo:
                        raw_nibble = byte_val & 0x0F
                    else:
                        raw_nibble = (byte_val >> 4) & 0x0F

                    # signed INT4: sign-extend nibble (matches dequant_q4nx.c)
                    int4_val = raw_nibble if raw_nibble < 8 else raw_nibble - 16

                    global_col = tile_col * COLS_PER_TILE + local_col
                    group = local_col // GROUP_SIZE  # 0..7

                    s = scales[group * ROWS_PER_TILE + local_row]
                    zp = zps[group * ROWS_PER_TILE + local_row]

                    result[global_row, global_col] = float(int4_val) * s + zp

    return result
