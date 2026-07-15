#!/usr/bin/env python3
"""Export a trained DeepSpec Eagle3 checkpoint (model.safetensors) into the flat
binary format MTPDraftWeights::load() expects — a fixed-order sequential
concatenation of float32 arrays, one per field, in exactly the order declared
in draft/mtp_draft.h."""
import argparse
import struct
import sys

import numpy as np
from safetensors import safe_open


FIELDS = [
    ("embed_tokens.weight", "embed_tokens"),
    ("fc.weight", "fc"),
    ("layers.0.hidden_norm.weight", "hidden_norm"),
    ("layers.0.input_layernorm.weight", "input_layernorm"),
    ("layers.0.self_attn.q_proj.weight", "q_proj"),
    ("layers.0.self_attn.k_proj.weight", "k_proj"),
    ("layers.0.self_attn.v_proj.weight", "v_proj"),
    ("layers.0.self_attn.o_proj.weight", "o_proj"),
    ("layers.0.self_attn.q_norm.weight", "q_norm"),
    ("layers.0.self_attn.k_norm.weight", "k_norm"),
    ("layers.0.post_attention_layernorm.weight", "post_attention_layernorm"),
    ("layers.0.mlp.gate_proj.weight", "gate_proj"),
    ("layers.0.mlp.up_proj.weight", "up_proj"),
    ("layers.0.mlp.down_proj.weight", "down_proj"),
    ("norm.weight", "norm"),
    ("lm_head.weight", "lm_head"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to model.safetensors")
    ap.add_argument("--output", required=True, help="Output .bin path")
    args = ap.parse_args()

    with safe_open(args.checkpoint, framework="pt") as f:
        keys = set(f.keys())
        missing = [k for k, _ in FIELDS if k not in keys]
        if missing:
            print(f"ERROR: checkpoint missing tensors: {missing}", file=sys.stderr)
            sys.exit(1)

        total_bytes = 0
        with open(args.output, "wb") as out:
            for tensor_name, field_name in FIELDS:
                t = f.get_tensor(tensor_name).to(dtype=__import__("torch").float32)
                arr = t.numpy().astype(np.float32)
                out.write(arr.tobytes())
                total_bytes += arr.nbytes
                print(f"  {field_name:28s} {str(list(arr.shape)):20s} {arr.nbytes/1e6:8.1f} MB")

    print(f"\nWrote {args.output} ({total_bytes/1e6:.1f} MB total)")


if __name__ == "__main__":
    main()
