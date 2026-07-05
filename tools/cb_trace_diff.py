#!/usr/bin/env python3
"""
Diff C++ engine substage dumps (--trace) against HF reference (layer_trace.py).

Reads:
  /tmp/cb_trace/<key>.bin        (float32, from npu_engine_cb --trace, layer 0)
  /tmp/layer_trace_outputs.npz   (from tools/layer_trace.py)

For each shared key, reports cos_sim, max-abs-diff, and relative L2.
The first key where cos_sim drops is the offending substage.
"""
import os
import sys
import numpy as np

CB_DIR = "/tmp/cb_trace"
NPZ = "/tmp/layer_trace_outputs.npz"

# key -> (size, optional reshape)
# layer_trace.py stores 2D for heads (q_heads [16,128] etc); we compare raveled.
KEYS = [
    "input_embedding",   # engine-only (no npz key) — skipped if absent
    "h_ln1",
    "q_flat", "k_flat", "v_flat",
    "q_heads", "k_heads", "v_heads",
    "q_normed", "k_normed",
    "q_rope", "k_rope",
    "attn_out_flat",
    "attn_proj",
    "h_after_attn",
    "h_ln2",
    "ffn_gate", "ffn_up",
    "ffn_hidden",
    "ffn_out",
    "h_out",
]


def cos_sim(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a.ravel(), b.ravel()) / (na * nb))


def main():
    ref = np.load(NPZ) if os.path.exists(NPZ) else {}
    print(f"{'key':<18}{'n':>7}{'cos_sim':>12}{'max_abs':>14}{'rel_L2':>10}  status")
    print("-" * 65)
    for k in KEYS:
        cb_path = os.path.join(CB_DIR, f"{k}.bin")
        if not os.path.exists(cb_path):
            print(f"{k:<18}{'— (no cb)':>43}")
            continue
        cb = np.fromfile(cb_path, dtype=np.float32)
        if k in ref:
            r = ref[k].astype(np.float32).ravel()
            n = min(len(cb), len(r))
            cb_n, r_n = cb[:n], r[:n]
            cs = cos_sim(cb_n, r_n)
            mad = float(np.max(np.abs(cb_n - r_n))) if n else 0.0
            rl2 = float(np.linalg.norm(cb_n - r_n) / max(np.linalg.norm(r_n), 1e-12))
            status = "OK" if cs > 0.999 else ("DIVERGE" if cs > 0.5 else "BLOWUP")
            print(f"{k:<18}{n:>7}{cs:>12.5f}{mad:>14.5f}{rl2:>10.5f}  {status}")
        else:
            amax = float(np.max(np.abs(cb))) if len(cb) else 0.0
            print(f"{k:<18}{len(cb):>7}{'(no ref)':>12}{amax:>14.5f}{'—':>10}  engine-only")


if __name__ == "__main__":
    main()