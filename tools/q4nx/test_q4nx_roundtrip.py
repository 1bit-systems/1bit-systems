#!/usr/bin/env python3
"""Q4NX quantize->dequantize round-trip regression test (issue #153).

Guards the packed-nibble layout: the packed I4 index must use the GLOBAL
column within a tile (g*32 + c), not the within-group column c. A regression
to `c` aliases all 8 column-groups onto the first 256 bytes of each lane and
corrupts 7/8 of every tile's weights, which silently produces degenerate
generation. This test quantizes a known block and dequantizes it with an
independent reader replica; any layout drift blows the error tolerance.

Run: python3 tools/q4nx/test_q4nx_roundtrip.py
"""
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load_writer():
    spec = importlib.util.spec_from_file_location(
        "q4nx_writer", ROOT / "tools" / "q4nx" / "hf_to_q4nx.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["q4nx_writer"] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def _bf16f(u16):
    return np.frombuffer(struct.pack("<I", int(u16) << 16), dtype="<f4")[0]


def _dequantize_block(enc):
    """Independent replica of engine/fusion/model_data.zig dequantizeI8Block."""
    scales = np.frombuffer(enc[0:512], dtype="<u2")
    zps = np.frombuffer(enc[512:1024], dtype="<u2")
    packed = enc[1024:5120]
    out = np.zeros((32, 256), np.float32)
    for lr in range(32):
        lane, bi, ns = lr // 16, (lr % 16) // 2, lr % 2
        for g in range(8):
            s = _bf16f(scales[g * 32 + lr])
            z = _bf16f(zps[g * 32 + lr])
            for c in range(32):
                idx = lane * 2048 + (g * 32 + c) * 8 + bi  # GLOBAL column
                bv = packed[idx]
                nib = (bv & 0x0F) if ns == 0 else ((bv >> 4) & 0x0F)
                out[lr, g * 32 + c] = nib * s + z
    return out


def main():
    writer = _load_writer()
    rng = np.random.default_rng(1234)
    max_err = 0.0
    for _ in range(8):
        block = (rng.standard_normal((32, 256)).astype(np.float32)) * 0.05
        enc = writer.quantize_block(block)
        assert len(enc) == 5120, f"expected 5120-byte tile, got {len(enc)}"
        rt = _dequantize_block(enc)
        max_err = max(max_err, float(np.abs(rt - block).max()))

    # 4-bit min-max quant of a ~0.3-wide row => step ~0.02 => max err ~0.01.
    # A layout regression (group aliasing) pushes this well past 0.1.
    tol = 0.02
    print(f"Q4NX round-trip max abs error: {max_err:.5f} (tol {tol})")
    if max_err > tol:
        print("FAIL: packed-nibble layout regression (see issue #153)")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
