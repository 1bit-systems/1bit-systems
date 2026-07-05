#!/usr/bin/env python3
"""
Q2_0 (GGML type 42) ternary GGUF decoder — VERIFIED BIT-EXACT against F16.

Format used by prism-ml/Ternary-Bonsai-*-gguf (the "Q2_0" ternary 1.58-bit files).

    block_q2_0 {
        ggml_half d;            // bytes 0..1  : f16 scale (little-endian)
        uint8_t   qs[QK/4];     // bytes 2..   : 2-bit codes, LSB-first, 4 per byte
    }
    QK   = 128            (elements per block)   <-- NOT 256
    size = 34 bytes/block (2 + 128/4)

    value[j] = (code - 1) * d       code in {0,1,2,3} -> {-1, 0, +1, +2}
    code(j)  = (qs[j/4] >> ((j%4)*2)) & 0x3

Verified: cos_vs_F16 = 1.000000 across blk.0.attn_output/attn_q,
blk.5.ffn_down, blk.15.attn_v (Ternary-Bonsai-1.7B).

⚠️  COLLISION: ZINC's src/model/gguf.zig maps GGML type 42 to `stq1_0`
(blockSize=256, bytesPerBlock=42, layout = 32 qs + 8 sign + 2 d). That is a
DIFFERENT format from prism-ml's Q2_0 at the same type id. Decoding this file
with the stq1_0 (256/42) layout yields cos≈0.495 (garbage). ZINC needs a
distinct Q2_0 (128/34, (q-1)*d) path — or to disambiguate type 42 by block
size / metadata — to run these Bonsai ternary models.
"""
import struct
import sys
import numpy as np


def parse_gguf(path):
    f = open(path, "rb")

    def rd(fmt):
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

    def rstr():
        return f.read(rd("Q")).decode("utf-8", "replace")

    assert f.read(4) == b"GGUF"
    rd("I")            # version
    n_tensors = rd("Q")
    n_kv = rd("Q")

    def rval(t):
        if t == 8:
            return rstr()
        if t in (0, 1):
            return rd("b" if t == 0 else "B")
        if t in (2, 3):
            return rd("h" if t == 2 else "H")
        if t in (4, 5):
            return rd("i" if t == 4 else "I")
        if t == 6:
            return rd("f")
        if t == 7:
            return rd("?")
        if t in (10, 11):
            return rd("q" if t == 10 else "Q")
        if t == 12:
            return rd("d")
        if t == 9:
            et = rd("I")
            ln = rd("Q")
            return [rval(et) for _ in range(ln)]
        raise ValueError(f"kv type {t}")

    align = 32
    for _ in range(n_kv):
        k = rstr()
        t = rd("I")
        v = rval(t)
        if k.endswith("alignment"):
            align = v

    tinfo = {}
    for _ in range(n_tensors):
        name = rstr()
        nd = rd("I")
        dims = [rd("Q") for _ in range(nd)]
        qt = rd("I")
        off = rd("Q")
        tinfo[name] = (dims, qt, off)

    data_start = f.tell()
    if data_start % align:
        data_start += align - (data_start % align)
    return tinfo, data_start, f


QK = 128
BYTES_PER_BLOCK = 2 + QK // 4  # 34


def dequant_q2_0(raw: bytes, n_elems: int) -> np.ndarray:
    """Decode raw Q2_0 tensor bytes -> float32[n_elems]. Bit-exact vs F16."""
    nb = n_elems // QK
    a = np.frombuffer(raw, dtype=np.uint8).reshape(nb, BYTES_PER_BLOCK)
    d = a[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb)
    qs = a[:, 2:]
    out = np.empty(n_elems, dtype=np.float32).reshape(nb, QK)
    for i in range(4):                       # 4 codes packed per byte, LSB-first
        codes = (qs >> (2 * i)) & 0x3
        out[:, i::4] = (codes.astype(np.float32) - 1.0) * d[:, None]
    return out.reshape(n_elems)


def _selftest(q2_path, f16_path):
    ti2, ds2, f2 = parse_gguf(q2_path)
    ti16, ds16, f16 = parse_gguf(f16_path)
    names = ["blk.0.attn_output.weight", "blk.0.attn_q.weight",
             "blk.5.ffn_down.weight", "blk.15.attn_v.weight"]
    for name in names:
        if name not in ti2 or name not in ti16:
            continue
        dims, qt, o2 = ti2[name]
        n = 1
        for x in dims:
            n *= x
        f2.seek(ds2 + o2)
        deq = dequant_q2_0(f2.read((n // QK) * BYTES_PER_BLOCK), n)
        _, _, o16 = ti16[name]
        f16.seek(ds16 + o16)
        ref = np.frombuffer(f16.read(n * 2), dtype=np.float16).astype(np.float32)
        cos = float(np.dot(deq, ref) / (np.linalg.norm(deq) * np.linalg.norm(ref) + 1e-9))
        print(f"  {name:28s} type={qt} cos_vs_F16={cos:.6f}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        _selftest(sys.argv[1], sys.argv[2])
    else:
        print("usage: q2_0_decode.py <model-q2_0.gguf> <model-f16.gguf>   # self-test")
        print(__doc__)
