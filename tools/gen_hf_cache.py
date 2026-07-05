#!/usr/bin/env python3
"""
Regenerate /tmp/hf_weights_cache with correct INT8 weights from Q4NX-dequant floats.

Reads Qwen3-0.6B model from Q4NX format (model.q4nx), dequantizes each
projection weight to float32, transposes to [in, out] layout, fuses Q/K/V
and Gate/Up as the engine expects, quantizes with a single global scale
(amax/127), and writes int8 + scale files.

Usage:
    python3 tools/gen_hf_cache.py

Output (/tmp/hf_weights_cache/):
    qkv_<l>.bin    int8 [1024, 4096]    Q[0:2048] K[2048:3072] V[3072:4096]
    o_<l>.bin      int8 [2048, 1024]    O projection
    gu_<l>.bin     int8 [1024, 6144]    Gate[0:3072] Up[3072:6144]
    d_<l>.bin      int8 [3072, 1024]    Down projection
    scales_<l>.bin float32 x 4          qk, o_, g_, d_ (one amax/127 per fused matrix)
    lm_head.bin    float32 [151936, 1024]   LM head (float, NOT int8)
    embeddings.bin float32 [151936, 1024]   Embedding table (float, NOT int8)
"""
import os, struct, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from q4nx_reference import dequantize_weight, read_bf16, get_header

# ── model dims ──────────────────────────────────────────────────────────
H = 1024
NH, NKV, HD = 16, 8, 128
IM = 3072
NV = 151936
NC = 28              # num layers
Q_OUT = NH * HD      # 2048
KV_OUT = NKV * HD    # 1024

CACHE = "/tmp/hf_weights_cache"
os.makedirs(CACHE, exist_ok=True)

# ── helpers ─────────────────────────────────────────────────────────────

def quantize_int8(weight):
    """Return (int8_bytes, scale) where weight is [in, out] float32.
    Uses a single global scale = amax/127 over the whole matrix.
    Matches packB() in npu_engine_cb.cpp.
    """
    amax = float(np.max(np.abs(weight)))
    if amax < 1e-12:
        amax = 1.0
    scale = amax / 127.0
    inv = 127.0 / amax
    q = np.clip(np.round(weight * inv), -127, 127).astype(np.int8)
    return q, scale


def transpose_fused(weights_dict, layers, keys_out_in):
    """
    For each layer, read each [out, in] weight by name, transpose to [in, out],
    fuse by concatenating columns, quantize, write.

    weights_dict: layer -> [(name, [out, in] float32)]
    keys_out_in: output filename keys per layer
    """
    scales = []
    for l in range(NC):
        fused_list = weights_dict[l]
        cols = []
        for _, w in fused_list:
            cols.append(w.T.copy())   # [out, in] → [in, out]
        fused = np.concatenate(cols, axis=1)  # [in, sum(out)]
        q_bytes, scale = quantize_int8(fused)
        scales.append(scale)
        for key, filename in keys_out_in:
            if filename is not None:
                pass  # handled by caller
    return scales


def write_bin(filename, arr):
    """Write numpy array as raw binary."""
    arr.astype(np.float32 if arr.dtype == np.float64 else arr.dtype).tofile(filename)


def main():
    print("=== Generating HF INT8 weight cache from Q4NX model ===\n")
    h = get_header()

    q_w, k_w, v_w, o_w, g_w, u_w, d_w = {}, {}, {}, {}, {}, {}, {}
    for l in range(NC):
        def dq(prefix, name, out, inp):
            key = f"model.layers.{l}.{prefix}.{name}.weight"
            off = h[key]["data_offsets"][0]
            rows = h[key]["shape"][0]
            return dequantize_weight(off, rows, inp)
        q_w[l] = dq("self_attn", "q_proj", Q_OUT, H)    # [2048, 1024]
        k_w[l] = dq("self_attn", "k_proj", KV_OUT, H)    # [1024, 1024]
        v_w[l] = dq("self_attn", "v_proj", KV_OUT, H)    # [1024, 1024]
        o_w[l] = dq("self_attn", "o_proj", H, Q_OUT)     # [1024, 2048]
        g_w[l] = dq("mlp", "gate_proj", IM, H)            # [3072, 1024]
        u_w[l] = dq("mlp", "up_proj", IM, H)              # [3072, 1024]
        d_w[l] = dq("mlp", "down_proj", H, IM)            # [1024, 3072]

        print(f"  layer {l:2d}: Q {q_w[l].shape} K {k_w[l].shape} V {v_w[l].shape} O {o_w[l].shape} G {g_w[l].shape} U {u_w[l].shape} D {d_w[l].shape}")

    print("\n--- Quantizing and writing ---\n")

    for l in range(NC):
        # ── QKV fused [1024, 4096] — quantize Q/K/V blocks independently ──
        q_t = q_w[l].T.copy()       # [1024, 2048]
        k_t = k_w[l].T.copy()       # [1024, 1024]
        v_t = v_w[l].T.copy()       # [1024, 1024]
        q_i8, q_scale = quantize_int8(q_t)   # Q block
        k_i8, k_scale = quantize_int8(k_t)   # K block
        v_i8, v_scale = quantize_int8(v_t)   # V block
        qkv_i8 = np.concatenate([q_i8, k_i8, v_i8], axis=1)  # [1024, 4096]
        write_bin(f"{CACHE}/qkv_{l}.bin", qkv_i8)

        # ── O [2048, 1024] ──
        o_t = o_w[l].T.copy()    # [2048, 1024]
        o_i8, o_scale = quantize_int8(o_t)
        write_bin(f"{CACHE}/o_{l}.bin", o_i8)

        # ── GU fused [1024, 6144] — quantize G/U blocks independently ──
        g_t = g_w[l].T.copy()    # [1024, 3072]
        u_t = u_w[l].T.copy()    # [1024, 3072]
        g_i8, g_scale = quantize_int8(g_t)   # G block
        u_i8, u_scale = quantize_int8(u_t)   # U block
        gu_i8 = np.concatenate([g_i8, u_i8], axis=1)  # [1024, 6144]
        write_bin(f"{CACHE}/gu_{l}.bin", gu_i8)

        # ── D [3072, 1024] ──
        d_t = d_w[l].T.copy()    # [3072, 1024]
        d_i8, d_scale = quantize_int8(d_t)
        write_bin(f"{CACHE}/d_{l}.bin", d_i8)

        # ── Scales: q, k, v, o, g, u, d (7 floats, matches struct ScaleSet)
        scales = np.array([q_scale, k_scale, v_scale, o_scale, g_scale, u_scale, d_scale], dtype=np.float32)
        write_bin(f"{CACHE}/scales_{l}.bin", scales)

        if l < 3 or l == NC - 1:
            print(f"  layer {l:2d}: q={q_scale:.6f} k={k_scale:.6f} v={v_scale:.6f} o={o_scale:.6f} g={g_scale:.6f} u={u_scale:.6f} d={d_scale:.6f}")

    # ── LM head [151936, 1024] float32 ──
    print("\n--- LM head (float) ---")
    lm_info = h["lm_head.weight"]
    lm_off = lm_info["data_offsets"][0]
    lm_i8_rows = lm_info["shape"][0]  # 18992 tile-rows
    # LM head is dtype=I8 (INT4 Q4NX tile format, same as projections), NOT BF16
    lm = dequantize_weight(lm_off, lm_i8_rows, H).astype(np.float32)  # [151936, 1024]
    write_bin(f"{CACHE}/lm_head.bin", lm)
    print(f"  range: [{lm.min():.4f}, {lm.max():.4f}] shape: {lm.shape}")

    # ── Embeddings [151936, 1024] float32 ──
    print("\n--- Embeddings (float) ---")
    emb_off = h["model.embed_tokens.weight"]["data_offsets"][0]
    emb = read_bf16(emb_off, NV * H).reshape(NV, H).astype(np.float32)
    write_bin(f"{CACHE}/embeddings.bin", emb)
    print(f"  range: [{emb.min():.4f}, {emb.max():.4f}] shape: {emb.shape}")

    print(f"\n=== Done → {CACHE} ===")
    # size summary
    for l in range(NC):
        for name in ["qkv", "o", "gu", "d"]:
            p = f"{CACHE}/{name}_{l}.bin"
            size_mb = os.path.getsize(p) / 1e6
            if l == 0:
                print(f"  {name}_{l}.bin: {os.path.getsize(p)} bytes ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()