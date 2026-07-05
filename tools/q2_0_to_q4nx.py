#!/usr/bin/env python3
"""
Q2_0 (ternary 1.58-bit) → INT8 Q4NX converter for NPU inference.

Reads a GGUF file with Q2_0 (type 42) ternary blocks, dequantizes to f32,
re-quantizes to INT8 symmetric-per-tensor, and writes a .q4nx file the
NPU engine can load.

Usage:
    python3 tools/q2_0_to_q4nx.py Ternary-Bonsai-1.7B-q2_0.gguf output.q4nx
"""

import struct
import sys
import numpy as np


# ── Q2_0 decoder (from q2_0_decode.py, extracted) ──────────────────────

def parse_gguf_tensors(path):
    """Return list of (name, shape, dtype_code, data_bytes) for all tensors."""
    f = open(path, "rb")

    def rd(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

    def rstr():
        return f.read(rd("Q")).decode("utf-8", "replace")

    assert f.read(4) == b"GGUF"
    rd("I")
    n_tensors = rd("Q")
    n_kv = rd("Q")

    for _ in range(n_kv):
        k = rstr()
        t = rd("I")
        if t == 6: rd("Q")  # array
        elif t == 7: rstr()
        elif t in (11, 12): rd("Q" if t == 11 else "I")
        elif t == 8: rstr()
        # skip others

    tensors = []
    alignment = rd("I") if n_tensors > 0 else 32
    stored_dtypes = {0: np.float32, 1: np.float16, 2: np.float64, 3: np.int8, 4: np.int16,
                     5: np.int32, 6: np.int64, 7: np.uint8, 8: np.uint16, 9: np.uint32, 10: np.uint64}
    ggml_types = {
        0: ("f32", 4), 1: ("f16", 2), 2: ("f64", 8),
        10: ("q8_0", 34), 11: ("q4_0", 18), 12: ("q4_1", 22),
        42: ("q2_0", 34),  # Q2_0 ternary (128-block, 34 bytes)
    }

    for _ in range(n_tensors):
        name = rstr()
        ndim = rd("I")
        shape = tuple(reversed([rd("Q") for _ in range(ndim)]))
        dtype = rd("I")
        offset = rd("Q")
        blk = ggml_types.get(dtype, ("?", 0))
        tensors.append((name, shape, dtype, offset, blk))

    return tensors, f


QK_Q2_0 = 128
BLOCK_BYTES_Q2_0 = 2 + QK_Q2_0 // 4  # scale f16 + 32 bytes of 2-bit codes


def decode_q2_0_block(data, offset):
    """Decode one Q2_0 block (34 bytes) → 128 f32 values."""
    d = struct.unpack_from("<e", data, offset)[0]  # f16 scale
    codes = np.frombuffer(data, dtype=np.uint8, count=32, offset=offset + 2)
    # 4 codes per byte, LSB-first, codes 0→{-1}, 1→{0}, 2→{+1}, 3→{+2}
    vals = np.zeros(128, dtype=np.float32)
    for j in range(128):
        c = (codes[j // 4] >> ((j % 4) * 2)) & 0x3
        vals[j] = (int(c) - 1) * d
    return vals


# ── Q4NX writer ───────────────────────────────────────────────────────

def write_q4nx(tensors, src_path, dst_path):
    """Convert Q2_0 GGUF tensors to INT8 Q4NX format."""
    tensors_meta, f = parse_gguf_tensors(src_path)
    f.close()
    f = open(src_path, "rb")

    out = open(dst_path, "wb")
    n_written = 0

    for name, shape, dtype, offset, (blk_name, blk_bytes) in tensors_meta:
        if dtype != 42:
            continue  # skip non-ternary tensors (embeddings, norms, etc.)

        print(f"  {name:50s} {str(shape):20s} {blk_name}")

        # Read raw blocks
        n_blocks = int(np.prod(shape) / QK_Q2_0)
        assert np.prod(shape) % QK_Q2_0 == 0, f"{name}: size not multiple of 128"

        f.seek(offset)
        raw = f.read(n_blocks * BLOCK_BYTES_Q2_0)

        # Dequantize all blocks → flat f32
        total_elems = n_blocks * QK_Q2_0
        f32_vals = np.zeros(total_elems, dtype=np.float32)
        for b in range(n_blocks):
            boff = b * BLOCK_BYTES_Q2_0
            f32_vals[b * QK_Q2_0:(b + 1) * QK_Q2_0] = decode_q2_0_block(raw, boff)

        # Reshape to original dims
        f32_vals = f32_vals.reshape(shape)

        # Symmetric INT8 quantization: scale = max(abs) / 127
        abs_max = np.max(np.abs(f32_vals))
        if abs_max < 1e-10:
            abs_max = 1.0
        scale = abs_max / 127.0
        i8_vals = np.clip(np.round(f32_vals / scale), -128, 127).astype(np.int8)

        # Write Q4NX tensor header + data
        # Simple format: 4 bytes size, 4 bytes scale (f32), then int8 data
        out.write(struct.pack("<I", i8_vals.nbytes))
        out.write(struct.pack("<f", scale))
        out.write(i8_vals.tobytes())
        n_written += 1

    f.close()
    out.close()
    print(f"\nWrote {n_written} tensors → {dst_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 tools/q2_0_to_q4nx.py model.gguf output.q4nx")
        sys.exit(1)

    tensors_meta, _ = parse_gguf_tensors(sys.argv[1])
    print(f"Found {len(tensors_meta)} tensors in {sys.argv[1]}")
    print("Converting Q2_0 ternary weights to INT8...")
    write_q4nx(tensors_meta, sys.argv[1], sys.argv[2])
