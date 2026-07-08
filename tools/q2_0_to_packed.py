#!/usr/bin/env python3
"""
q2_0_to_packed.py — Convert Q2_0 GGUF ternary weights to packed 2-bit format
for native ternary NPU kernels.

The native ternary kernel (mm_ternary_32x64x128) expects:
  [M * K_packed bytes uint8 (2-bit packed ternary, 4 values/byte)]
  [M * 2 bytes bf16 (per-row scales)]

where:
  - M = output features (rows)
  - K = input features (columns)
  - K_packed = K / 4 (4 ternary values per byte)
  - Each 2-bit code: 00→-1.0, 01→0.0, 10→+1.0, 11→-1.0

GGUF Q2_0 format:
  - Blocks of 128 values: 2(fp16 scale) + 32(packed 2-bit codes)
  - Same 2-bit encoding as the kernel
  - Per-block scales need to be combined into per-row scales

Output: model.ternary directory containing:
  - manifest.json: {tensor_name: {path, M, K, offset_weights, offset_scales}}
  - weights.bin: concatenated [M*K_packed uint8...] [M*2 bf16...] per tensor

Usage:
  python tools/q2_0_to_packed.py model.gguf output_dir/
"""

import json
import math
import os
import struct
import sys
import numpy as np

# ── GGUF parsing ──────────────────────────────────────────────

QK_Q2_0 = 128
BLOCK_BYTES_Q2_0 = 34  # 2(fp16 scale) + 32(packed 2-bit)

# GGUF → HuggingFace name mapping (Qwen3/Bonsai)
NAME_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}


def map_name(gguf_name):
    if gguf_name in NAME_MAP:
        return NAME_MAP[gguf_name]
    # blk.{N}.attn_{q,k,v,output}.weight → model.layers.{N}.self_attn.{q,k,v,o}_proj.weight
    m = re.match(r"blk\.(\d+)\.attn_q\.weight", gguf_name)
    if m:
        return f"model.layers.{m.group(1)}.self_attn.q_proj.weight"
    m = re.match(r"blk\.(\d+)\.attn_k\.weight", gguf_name)
    if m:
        return f"model.layers.{m.group(1)}.self_attn.k_proj.weight"
    m = re.match(r"blk\.(\d+)\.attn_v\.weight", gguf_name)
    if m:
        return f"model.layers.{m.group(1)}.self_attn.v_proj.weight"
    m = re.match(r"blk\.(\d+)\.attn_output\.weight", gguf_name)
    if m:
        return f"model.layers.{m.group(1)}.self_attn.o_proj.weight"
    m = re.match(r"blk\.(\d+)\.ffn_gate\.weight", gguf_name)
    if m:
        return f"model.layers.{m.group(1)}.mlp.gate_proj.weight"
    m = re.match(r"blk\.(\d+)\.ffn_up\.weight", gguf_name)
    if m:
        return f"model.layers.{m.group(1)}.mlp.up_proj.weight"
    m = re.match(r"blk\.(\d+)\.ffn_down\.weight", gguf_name)
    if m:
        return f"model.layers.{m.group(1)}.mlp.down_proj.weight"
    m = re.match(r"blk\.(\d+)\.attn_norm\.weight", gguf_name)
    if m:
        return f"model.layers.{m.group(1)}.input_layernorm.weight"
    m = re.match(r"blk\.(\d+)\.ffn_norm\.weight", gguf_name)
    if m:
        return f"model.layers.{m.group(1)}.post_attention_layernorm.weight"
    m = re.match(r"blk\.(\d+)\.attn_q_norm\.weight", gguf_name)
    if m:
        return f"model.layers.{m.group(1)}.self_attn.q_norm.weight"
    m = re.match(r"blk\.(\d+)\.attn_k_norm\.weight", gguf_name)
    if m:
        return f"model.layers.{m.group(1)}.self_attn.k_norm.weight"
    # kwarg Qwen names
    m = re.match(r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight", gguf_name)
    if m:
        return gguf_name
    m = re.match(r"model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight", gguf_name)
    if m:
        return gguf_name
    return gguf_name


def gguf_tensors(path):
    """Yield (name, shape, dtype_code, absolute_offset) for all GGUF tensors."""

    def r32(f):
        return struct.unpack("<I", f.read(4))[0]

    def r64(f):
        return struct.unpack("<Q", f.read(8))[0]

    def rstr(f):
        return f.read(r64(f)).decode("utf-8", errors="replace")

    def skip_val(f, typ):
        if typ == 0:
            f.read(1)
        elif typ == 1:
            f.read(1)
        elif typ == 2:
            f.read(2)
        elif typ == 3:
            f.read(4)
        elif typ == 4:
            f.read(4)
        elif typ == 5:
            f.read(8)
        elif typ == 6:
            f.read(8)
        elif typ == 7:
            f.read(8)
        elif typ == 8:
            n = r64(f)
            f.read(n)
        elif typ == 9:
            rstr(f)
        elif typ == 10:
            n = r32(f)
            for _ in range(n):
                rstr(f)
                skip_val(f, r32(f))

    with open(path, "rb") as f:
        magic = r32(f)
        if magic != 0x46554747:
            raise ValueError(f"Not a GGUF file: magic={magic:#x}")
        version = r32(f)
        n_tensors = r64(f)
        n_kv = r64(f)

        # Skip metadata KV pairs
        for _ in range(n_kv):
            rstr(f)
            skip_val(f, r32(f))

        # Read tensor infos
        tensor_infos = []
        for _ in range(n_tensors):
            name = rstr(f)
            ndim = r32(f)
            shape = tuple(reversed([r64(f) for _ in range(ndim)]))
            dtype = r32(f)
            offset = r64(f)
            tensor_infos.append((name, shape, dtype, offset))

        data_start = f.tell()
        data_start = (data_start + 31) & ~31

        for name, shape, dtype, offset in tensor_infos:
            yield name, shape, dtype, data_start + offset


# ── Q2_0 decoding to packed + per-row scales ──────────────────

def pack_q2_0_ternary(raw_data, M, K):
    """
    Convert Q2_0 raw data to packed ternary format.

    Q2_0 block (34 bytes, 128 values):
      [0:2]   fp16 super-block scale d
      [2:34]  32 bytes packed 2-bit codes (4 values/byte)

    Returns:
      weights: uint8 array [M * K_packed] — 2-bit packed (4 values/byte)
      scales:  float32 array [M] — per-row scale (product of block scales)
    """
    K_packed = K // 4  # 4 ternary values per byte
    assert K % 128 == 0, f"K={K} must be a multiple of 128"

    n_blocks_per_row = K // 128
    total_blocks = M * n_blocks_per_row
    expected_bytes = total_blocks * BLOCK_BYTES_Q2_0

    if len(raw_data) < expected_bytes:
        raise ValueError(f"Data too short: {len(raw_data)} < {expected_bytes}")

    raw = raw_data[:expected_bytes]
    block_data = np.frombuffer(raw, dtype=np.uint8).reshape(total_blocks, BLOCK_BYTES_Q2_0)

    # Extract super-block fp16 scales
    d_bytes = block_data[:, :2].copy()
    d = np.frombuffer(d_bytes.tobytes(), dtype=np.float16).astype(np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d = d.reshape(M, n_blocks_per_row)  # [M, blocks_per_row]

    # Per-row scale: mean of block scales (bf16 range)
    row_scales = d.mean(axis=1).astype(np.float32)

    # Extract packed 2-bit codes (32 bytes = 128 values per block)
    # The Q2_0 block stores codes in bytes [2:34], little-endian bit order
    packed = block_data[:, 2:34].copy()  # [total_blocks, 32]

    # Reorganize: interleave blocks back into rows
    # Each row has n_blocks_per_row blocks of 32 packed bytes each
    # Output: [M, K_packed] where K_packed = n_blocks_per_row * 32
    weights = packed.reshape(M, n_blocks_per_row * 32)

    return weights, row_scales


# ── Main ──────────────────────────────────────────────────────

def main():
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} model.gguf [output_dir]")
        print(f"  Converts Q2_0 ternary weights to packed format for native ternary NPU kernels.")
        print(f"  Output dir defaults to model.ternary/")
        sys.exit(1)

    src = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else src.replace(".gguf", ".ternary")
    os.makedirs(out_dir, exist_ok=True)

    tensors = list(gguf_tensors(src))
    print(f"GGUF: {len(tensors)} tensors from {src}")

    manifest = {}
    weights_bin_path = os.path.join(out_dir, "weights.bin")
    weight_chunks = []

    total_weight_bytes = 0

    with open(src, "rb") as gf:
        for name, shape, dtype, offset in tensors:
            n_elems = int(math.prod(shape))
            hf_name = map_name(name)

            if dtype == 42:  # Q2_0 ternary
                M = shape[0]  # out_features
                K = shape[1] if len(shape) > 1 else 1  # in_features

                if K == 1 or M == 1:
                    # 1D tensor (norm weights, etc.) — skip for now
                    print(f"  SKIP {hf_name} shape={shape} (1D)")
                    continue

                gf.seek(offset)
                raw_size = (n_elems // QK_Q2_0) * BLOCK_BYTES_Q2_0
                raw = gf.read(raw_size)

                weights, row_scales = pack_q2_0_ternary(raw, M, K)

                K_packed = K // 4
                weights_bytes = weights.tobytes()       # [M * K_packed] uint8
                scales_bf16 = row_scales.astype(np.float16).tobytes()  # [M * 2] bf16

                offset_weights = total_weight_bytes
                offset_scales = offset_weights + len(weights_bytes)

                chunk = weights_bytes + scales_bf16
                weight_chunks.append(chunk)
                total_weight_bytes += len(chunk)

                manifest[hf_name] = {
                    "M": M,
                    "K": K,
                    "K_packed": K_packed,
                    "offset_weights": offset_weights,
                    "offset_scales": offset_scales,
                    "size_bytes": len(chunk),
                }

                print(f"  {hf_name}: M={M} K={K} → {len(chunk)} bytes "
                      f"({len(weights_bytes)} weights + {len(scales_bf16)} scales)")

            elif dtype in (0, 1):  # F32 or F16 (norms, embed, lm_head)
                el_sz = 4 if dtype == 0 else 2
                dtype_name = "F32" if dtype == 0 else "F16"
                gf.seek(offset)
                raw = gf.read(n_elems * el_sz)

                manifest[hf_name] = {
                    "shape": list(shape),
                    "dtype": dtype_name,
                    "offset_bytes": total_weight_bytes,
                    "size_bytes": len(raw),
                }
                weight_chunks.append(raw)
                total_weight_bytes += len(raw)

                print(f"  {hf_name}: {dtype_name} {list(shape)}")

    # Write weights.bin
    weights_path = os.path.join(out_dir, "weights.bin")
    with open(weights_path, "wb") as f:
        for chunk in weight_chunks:
            f.write(chunk)

    # Write manifest.json
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(manifest)} tensors → {out_dir}/")
    print(f"  weights.bin:    {total_weight_bytes / 1e6:.2f} MB")
    print(f"  manifest.json:  {os.path.getsize(manifest_path)} bytes")
    print(f"\nReady for npu_ternary_serve.")


if __name__ == "__main__":
    import re
    main()
