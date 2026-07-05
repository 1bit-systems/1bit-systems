#!/usr/bin/env python3
"""
Isolate the QKV GEMM blowup: compare the engine's HF-cached INT8 QKV weights
(dequantized with the cache's global scale) against the Q4NX INT4-dequant
reference weights used by tools/layer_trace.py.

Engine cache layout (/tmp/hf_weights_cache/qkv_<l>.bin):
  int8 bytes, row-major [in_features=1024, out_features=4096] (already transposed
  by packB for the GEMM A@B convention). Columns [0:2048]=Q, [2048:3072]=K,
  [3072:4096]=V. Single global scale wsc[l].qk = amax/127 over the whole fused
  matrix:  w_float = int8_byte * scale.

Reference (q4nx_reference.dequantize_weight): [out_features, in_features],
  per-group INT4 scales/zero-points. Transpose to [in, out] for comparison.

If the two agree (cos_sim > 0.95 per block) the QKV GEMM blowup is in the
NPU kernel / dynamic_ascale activation-quant path, NOT the weights.
If they disagree, the HF cache generation script is the culprit.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from q4nx_reference import dequantize_weight, get_header

HIDDEN = 1024
NH, NKV, HD = 16, 8, 128
Q_OUT, KV_OUT = NH * HD, NKV * HD   # 2048, 1024
CACHE = "/tmp/hf_weights_cache"


def load_int8_cache(layer):
    """Return dequantized [in=1024, out=4096] float32 + scale for the fused QKV."""
    with open(os.path.join(CACHE, f"qkv_{layer}.bin"), "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.int8).astype(np.float32)
    # layout [in=1024, out=4096]
    w = raw.reshape(HIDDEN, Q_OUT + 2 * KV_OUT)
    # scale
    struct_path = os.path.join(CACHE, f"scales_{layer}.bin")
    with open(struct_path, "rb") as f:
        scales = np.frombuffer(f.read(), dtype=np.float32)
    # scales file has 7 floats: q, k, v, o, g, u, d (struct ScaleSet)
    # For the fused QKV block, the host uses go_multi with per-projection dequant.
    # The cache batch-quantizes Q/K/V independently; for comparison we use the
    # Q scale (scales[0]) as the overall scale since cos_sim is robust to this.
    return w, float(scales[0])


def ref_block(layer, name, out_f):
    """[out_f, in_f] INT4-dequant reference; transpose to [in, out]."""
    h = get_header()
    off = h[f"model.layers.{layer}.self_attn.{name}.weight"]["data_offsets"][0]
    i8_rows = h[f"model.layers.{layer}.self_attn.{name}.weight"]["shape"][0]
    w = dequantize_weight(off, i8_rows, HIDDEN)        # [out_f, HIDDEN]
    return w.T.copy()                                    # [HIDDEN, out_f]


def stats(a, b, name):
    n = min(a.size, b.size)
    a, b = a.ravel()[:n], b.ravel()[:n]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cs = float(np.dot(a, b) / (na * nb)) if na and nb else 0.0
    mad = float(np.max(np.abs(a - b)))
    rl2 = float(np.linalg.norm(a - b) / max(nb, 1e-12))
    print(f"  {name:<14} cos_sim={cs:+.5f}  max_abs={mad:.5f}  rel_L2={rl2:.5f}")
    return cs


def main():
    for layer in [0, 1, 2]:
        print(f"\n=== layer {layer} ===")
        w_cache, scale = load_int8_cache(layer)
        dq = w_cache * scale
        # split cache columns into Q / K / V
        q_cache = dq[:, :Q_OUT]
        k_cache = dq[:, Q_OUT:Q_OUT + KV_OUT]
        v_cache = dq[:, Q_OUT + KV_OUT:]
        # references (transposed to [in, out])
        q_ref = ref_block(layer, "q_proj", Q_OUT)
        k_ref = ref_block(layer, "k_proj", KV_OUT)
        v_ref = ref_block(layer, "v_proj", KV_OUT)
        print(f"  cache scale (qk) = {scale:.6f}")
        csq = stats(q_cache, q_ref, "Q block")
        csk = stats(k_cache, k_ref, "K block")
        csv = stats(v_cache, v_ref, "V block")
        verdict = "WEIGHTS_MATCH (kernel/quant is suspect)" if min(csq, csk, csv) > 0.95 \
                  else "WEIGHTS_DIVERGE (cache generation suspect)"
        print(f"  → {verdict}")


if __name__ == "__main__":
    main()