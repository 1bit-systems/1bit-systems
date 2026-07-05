#!/usr/bin/env python3
"""
Q2_0 (ternary 1.58-bit, GGUFv3) → Q4NX converter for NPU inference.

Produces a Q4NX file with JSON header + INT8 weight data that
npu_engine_universal.cpp can load directly.

Usage:
    python3 tools/q2_0_to_q4nx.py model.gguf output.q4nx
    ./npu_engine_universal output.q4nx
"""

import json
import re
import struct
import sys
import numpy as np

QK_Q2_0 = 128
BLOCK_BYTES_Q2_0 = 34

# GGUF → HuggingFace name mapping (Qwen3/Bonsai)
NAME_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
BLK_MAP = {
    "attn_norm": "input_layernorm",
    "ffn_norm": "post_attention_layernorm",
    "attn_k_norm": "self_attn.k_norm",
    "attn_q_norm": "self_attn.q_norm",
    "attn_k": "self_attn.k_proj",
    "attn_q": "self_attn.q_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def map_name(gguf_name):
    if gguf_name in NAME_MAP:
        return NAME_MAP[gguf_name]
    m = re.match(r"blk\.(\d+)\.(.+?)(\.weight)?$", gguf_name)
    if m:
        n, rest = int(m.group(1)), m.group(2)
        if rest in BLK_MAP:
            return f"model.layers.{n}.{BLK_MAP[rest]}.weight"
    return gguf_name


def gguf_tensors(path):
    """Yield (name, shape, dtype_code, offset) for all GGUF tensors."""

    def r32(f):
        return struct.unpack("<I", f.read(4))[0]

    def r64(f):
        return struct.unpack("<Q", f.read(8))[0]

    def rstr(f):
        return f.read(r64(f)).decode("utf-8", "replace")

    def skip_val(f, t):
        if t in (0, 1, 7):
            f.read(1)
        elif t in (2, 3):
            f.read(2)
        elif t in (4, 5, 6, 10):
            f.read(4)
        elif t in (11, 12):
            f.read(8)
        elif t == 8:
            f.read(r64(f))
        elif t == 9:
            at = r32(f)
            n = r64(f)
            for _ in range(n):
                skip_val(f, at)

    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        ver = r32(f)
        n_tensors = r64(f)
        n_kv = r64(f)

        for _ in range(n_kv):
            rstr(f)
            skip_val(f, r32(f))

        # GGUFv3: no alignment between KV and tensor infos

        for _ in range(n_tensors):
            name = rstr(f)
            ndim = r32(f)
            shape = tuple(reversed([r64(f) for _ in range(ndim)]))
            dtype = r32(f)
            offset = r64(f)
            yield name, shape, dtype, offset


def decode_q2_0(data):
    """Decode Q2_0 ternary data → f32 array."""
    n_blocks = len(data) // BLOCK_BYTES_Q2_0
    scales = np.frombuffer(data, dtype=np.float16, count=n_blocks, offset=0).astype(np.float32)
    codes = np.frombuffer(data, dtype=np.uint8, count=n_blocks * 32, offset=2).reshape(n_blocks, 32)
    decoded = np.zeros((n_blocks, 128), dtype=np.float32)
    for bit in range(4):
        val = ((codes >> (bit * 2)) & 0x3).astype(np.float32)
        decoded[:, bit::4] = (val - 1.0)
    decoded *= scales[:, np.newaxis]
    return decoded.ravel()


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} model.gguf output.q4nx")
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]

    tensors = list(gguf_tensors(src))
    print(f"GGUF: {len(tensors)} tensors")

    header = {}
    data_blocks = []
    data_offset = 0

    with open(src, "rb") as gf:
        for name, shape, dtype, offset in tensors:
            n_elems = int(np.prod(shape))
            hf_name = map_name(name)

            if dtype == 42:
                # Q2_0 ternary → INT8
                gf.seek(offset)
                raw = gf.read(n_elems // QK_Q2_0 * BLOCK_BYTES_Q2_0)
                f32_vals = decode_q2_0(raw).reshape(shape)

                abs_max = np.max(np.abs(f32_vals))
                scale_f = abs_max / 127.0 if abs_max > 1e-10 else 1.0
                i8_vals = np.clip(np.round(f32_vals / scale_f), -128, 127).astype(np.int8)
                raw_data = i8_vals.tobytes()

                header[hf_name] = {
                    "dtype": "I8",
                    "shape": list(shape),
                    "data_offsets": [data_offset, data_offset + len(raw_data)],
                    "scale": round(float(scale_f), 6),
                }
                data_blocks.append(raw_data)
                data_offset += len(raw_data)

            elif dtype in (0, 1):
                # f32 or f16 (norms, lm_head, embed)
                el_sz = 4 if dtype == 0 else 2
                dtype_name = "F32" if dtype == 0 else "F16"
                gf.seek(offset)
                raw_data = gf.read(n_elems * el_sz)

                header[hf_name] = {
                    "dtype": dtype_name,
                    "shape": list(shape),
                    "data_offsets": [data_offset, data_offset + len(raw_data)],
                }
                data_blocks.append(raw_data)
                data_offset += len(raw_data)

    # Add lm_head.weight alias if missing (tied embeddings)
    if "lm_head.weight" not in header and "model.embed_tokens.weight" in header:
        emb = header["model.embed_tokens.weight"]
        header["lm_head.weight"] = dict(emb)
        # Copy the data block reference
        emb_off = emb["data_offsets"]
        header["lm_head.weight"]["data_offsets"] = list(emb_off)

    hdr_json = json.dumps(header, separators=(",", ":"))
    hdr_bytes = hdr_json.encode()

    with open(dst, "wb") as out:
        out.write(struct.pack("<Q", len(hdr_bytes)))
        out.write(hdr_bytes)
        for block in data_blocks:
            out.write(block)

    n_w = sum(1 for v in header.values() if v.get("dtype") == "I8")
    print(f"\nWrote {len(header)} tensors ({n_w} INT8 weights) → {dst}")
    print(f"Size: {data_offset / 1e6:.1f} MB")

    import os
    actual = os.path.getsize(dst)
    expected = 8 + len(hdr_bytes) + data_offset
    assert actual == expected, "Size mismatch!"
    print("✅ Q4NX format valid (engine-ready)")


if __name__ == "__main__":
    main()
