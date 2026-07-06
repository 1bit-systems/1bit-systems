#!/usr/bin/env python3
"""Export a DSpark checkpoint (model.safetensors) to the flat binary format
that DSparkDraftWeights::load() expects.

Binary layout (sequential float32 arrays, in this exact order):
  1. embed_tokens           [vocab, hidden]
  2. fc                     [hidden, num_target_layers * hidden]
  3. hidden_norm            [hidden]
  4. input_layernorm        [num_layers, hidden]
  5. q_proj                 [num_layers, num_heads*head_dim, hidden]
  6. k_proj                 [num_layers, num_kv_heads*head_dim, hidden]
  7. v_proj                 [num_layers, num_kv_heads*head_dim, hidden]
  8. o_proj                 [num_layers, hidden, num_heads*head_dim]
  9. q_norm                 [num_layers, head_dim]
  10. k_norm                [num_layers, head_dim]
  11. post_attention_layernorm [num_layers, hidden]
  12. gate_proj             [num_layers, inter_dim, hidden]
  13. up_proj               [num_layers, inter_dim, hidden]
  14. down_proj             [num_layers, hidden, inter_dim]
  15. norm                  [hidden]
  16. lm_head               [vocab, hidden]
  17. markov_w1             [vocab, markov_rank]
  18. markov_w2             [vocab, markov_rank]
  19. confidence_weight     [hidden + markov_rank]
  20. confidence_bias       [1]

Each per-layer field stores all N layers' data contiguously (layer 0, then
layer 1, ..., layer N-1) so the C++ loader can read one field at a time.

See: draft/dspark_draft.h DSparkDraftWeights::load() / validate()
"""

import argparse
import struct
import sys

import numpy as np
from safetensors import safe_open


# Per-layer fields (within each field, all layers are written contiguously).
# Each entry: (safetensors_template, cpp_field_name)
# The {l} placeholder is substituted with the layer index.
PER_LAYER_FIELDS = [
    ("layers.{l}.input_layernorm.weight",      "input_layernorm"),
    ("layers.{l}.self_attn.q_proj.weight",      "q_proj"),
    ("layers.{l}.self_attn.k_proj.weight",      "k_proj"),
    ("layers.{l}.self_attn.v_proj.weight",      "v_proj"),
    ("layers.{l}.self_attn.o_proj.weight",      "o_proj"),
    ("layers.{l}.self_attn.q_norm.weight",      "q_norm"),
    ("layers.{l}.self_attn.k_norm.weight",      "k_norm"),
    ("layers.{l}.post_attention_layernorm.weight", "post_attention_layernorm"),
    ("layers.{l}.mlp.gate_proj.weight",         "gate_proj"),
    ("layers.{l}.mlp.up_proj.weight",           "up_proj"),
    ("layers.{l}.mlp.down_proj.weight",         "down_proj"),
]

# Top-level (non-per-layer) tensors in binary order.
TOP_LEVEL = [
    ("embed_tokens.weight",          "embed_tokens"),
    ("fc.weight",                    "fc"),
    ("hidden_norm.weight",           "hidden_norm"),
    ("norm.weight",                  "norm"),
    ("lm_head.weight",               "lm_head"),
    ("markov_head.markov_w1.weight",  "markov_w1"),
    ("markov_head.markov_w2.weight",  "markov_w2"),
]


def write_tensor(f_out, tensor, name):
    """Write a bf16 or float32 tensor as float32 to the binary file."""
    t = tensor.to(dtype=__import__("torch").float32)
    arr = t.numpy().astype(np.float32)
    f_out.write(arr.tobytes())
    print(f"  {name:32s} {str(list(t.shape)):24s} {arr.nbytes/1e6:8.1f} MB")
    return arr.nbytes


def main():
    ap = argparse.ArgumentParser(
        description="Export DSpark checkpoint to flat C++ binary format"
    )
    ap.add_argument("--checkpoint", required=True,
                    help="Path to model.safetensors")
    ap.add_argument("--output", required=True,
                    help="Output .bin path for C++ loader")
    ap.add_argument("--num-layers", type=int, default=5,
                    help="Number of draft layers")
    args = ap.parse_args()

    NL = args.num_layers

    with safe_open(args.checkpoint, framework="pt") as f:
        keys = set(f.keys())
        total_bytes = 0

        with open(args.output, "wb") as out:
            # ---- Shared tensors (first 3 in binary layout) ----
            for tensor_name, field_name in TOP_LEVEL[:3]:
                if tensor_name not in keys:
                    print(f"ERROR: required tensor '{tensor_name}' not found",
                          file=sys.stderr)
                    sys.exit(1)
                total_bytes += write_tensor(
                    out, f.get_tensor(tensor_name), field_name)

            # ---- Per-layer fields (all layers of each field together) ----
            for safetensors_tpl, field_name in PER_LAYER_FIELDS:
                for l in range(NL):
                    tensor_name = safetensors_tpl.format(l=l)
                    if tensor_name not in keys:
                        print(f"ERROR: required tensor '{tensor_name}' not found",
                              file=sys.stderr)
                        sys.exit(1)
                    # Show shape on the first layer only to reduce verbosity
                    label = field_name if l == 0 else ""
                    total_bytes += write_tensor(
                        out, f.get_tensor(tensor_name), label)

            # ---- Final norm + output head (remaining top-level) ----
            for tensor_name, field_name in TOP_LEVEL[3:]:
                if tensor_name not in keys:
                    print(f"ERROR: required tensor '{tensor_name}' not found",
                          file=sys.stderr)
                    sys.exit(1)
                total_bytes += write_tensor(
                    out, f.get_tensor(tensor_name), field_name)

            # ---- Markov head ----
            # Already handled in TOP_LEVEL: markov_w1, markov_w2

            # ---- Confidence head ----
            # weight: [1, hidden + markov_rank] → squeeze to 1D [hidden + markov_rank]
            conf_weight_name = "confidence_head.proj.weight"
            if conf_weight_name not in keys:
                print(f"ERROR: required tensor '{conf_weight_name}' not found",
                      file=sys.stderr)
                sys.exit(1)
            cw_t = f.get_tensor(conf_weight_name)
            cw_t = cw_t.to(dtype=__import__("torch").float32)
            cw_arr = cw_t.numpy().astype(np.float32).flatten()
            out.write(cw_arr.tobytes())
            print(f"  confidence_weight          {str(list(cw_t.shape)):24s} {cw_arr.nbytes/1e6:8.1f} MB")
            total_bytes += cw_arr.nbytes

            # bias: [1]
            conf_bias_name = "confidence_head.proj.bias"
            if conf_bias_name not in keys:
                print(f"ERROR: required tensor '{conf_bias_name}' not found",
                      file=sys.stderr)
                sys.exit(1)
            cb_t = f.get_tensor(conf_bias_name)
            cb_t = cb_t.to(dtype=__import__("torch").float32)
            cb_arr = cb_t.numpy().astype(np.float32).flatten()
            out.write(cb_arr.tobytes())
            print(f"  confidence_bias            {str(list(cb_t.shape)):24s} {cb_arr.nbytes/1e6:8.1f} MB")
            total_bytes += cb_arr.nbytes

    print(f"\nWrote {args.output} ({total_bytes/1e6:.1f} MB total, "
          f"{total_bytes/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
