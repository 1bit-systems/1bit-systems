#!/usr/bin/env python3
"""
medusa_safetensors_to_h1bm.py — convert MedusaBitNet heads to .h1b-medusa
(v1 OR v2) sidecar files. See ../../docs/h1b-medusa-format.md.

Offline tooling only — Rule A bans Python in the runtime path, but `tools/`
scripts that produce on-disk artifacts are fair game.

Two writer modes:

  --variant vocab (default, v1)
      Reads HuggingFace per-head safetensors `medusa_*/lm_head.weight`
      [vocab, hidden] (or a torch .pt fallback) and emits a v1 sidecar
      with absmean-ternarized halo-1bit packing. Mismatch with the
      parrishcorcoran upstream (kept for legacy + synthetic-zero tests).

  --variant residual_mlp (v2)
      Reads parrishcorcoran/MedusaBitNet-2B-4T's `medusa_heads_step2000.pt`
      torch checkpoint:
          heads.w_in  : [1, num_heads, hidden, hidden]  (bf16)
          heads.w_out : [1, num_heads, hidden, hidden]  (bf16)
      and emits a v2 sidecar with `weight_dtype = bf16`. The runtime
      casts to fp16 at load time and reuses base.lm_head as the shared
      vocab projection.

  --synthetic-zero
      Emits a header-correct, all-zero file. Works with both variants;
      the v1 form is what tests/test_medusa_loader.cpp consumes for
      smoke-testing without real weights.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import List, Tuple


WEIGHT_FORMATS = {
    "halo_v2": 0,
    "sherry_i8": 1,
    "tq1": 2,
    "sherry_fp16": 3,
    "bonsai_q1": 4,
    "bonsai_tq2": 5,
}

# v2 weight_dtype enum (matches docs/h1b-medusa-format.md and
# include/rocm_cpp/medusa.h:rcpp_medusa_dtype_t).
V2_DTYPES = {
    "bf16": 0,
    "fp16": 1,
    "halo_v2_ternary": 2,
    "sherry_i8": 3,
}

# Variants (matches rcpp_medusa_variant_t).
V1_VARIANT_VOCAB        = 0
V2_VARIANT_RESIDUAL_MLP = 1


def _absmean_ternarize(w_row, eps=1e-5):
    """Quantize a row to {-1, 0, +1} via BitNet b1.58 absmean rule.
    Returns (codes_uint8 in {0,1,2}, scale_fp32)."""
    import numpy as np
    abs_row = np.abs(w_row)
    scale = float(abs_row.mean()) + eps
    quant = np.round(w_row / scale).clip(-1, 1).astype(np.int8)
    # codes: 0->-1, 1->0, 2->+1.
    codes = (quant + 1).astype(np.uint8)
    return codes, scale


def _pack_halo_byte(codes, hidden):
    """Pack a [vocab, hidden] uint8 codes array into halo-1bit bytes.
    Layout: uint8 [vocab, (hidden+3)//4], 4 codes per byte, K-contiguous.
    Code 3 (reserved → 0) is never emitted; we map {0,1,2} as documented."""
    import numpy as np
    vocab = codes.shape[0]
    packed_cols = (hidden + 3) // 4
    out = np.zeros((vocab, packed_cols), dtype=np.uint8)
    for j in range(4):
        idxs = slice(j, hidden, 4)
        sub = codes[:, idxs]            # [vocab, ceil((hidden-j)/4)]
        out[:, : sub.shape[1]] |= ((sub & 0x3) << (j * 2)).astype(np.uint8)
    return out


def _ternarize_head(weight, hidden, vocab):
    """Convert a [vocab, hidden] fp16/bf16/fp32 head to (packed bytes,
    fp32 row_scales). Pure-Python fallback if numpy isn't available
    runs via the absmean rule per row."""
    import numpy as np
    w = weight.astype(np.float32)
    assert w.shape == (vocab, hidden), \
        f"head shape {w.shape} != ({vocab}, {hidden})"
    codes = np.zeros_like(w, dtype=np.uint8)
    scales = np.zeros(vocab, dtype=np.float32)
    for r in range(vocab):
        c, s = _absmean_ternarize(w[r])
        codes[r] = c
        scales[r] = s
    packed = _pack_halo_byte(codes, hidden)
    return packed, scales


def _write_header(f, *, num_heads, hidden, vocab, weight_format):
    """v1 header (32 bytes, version=1)."""
    f.write(b"H1BM")
    f.write(struct.pack("<I", 1))                # version
    f.write(struct.pack("<I", num_heads))
    f.write(struct.pack("<I", hidden))
    f.write(struct.pack("<I", vocab))
    f.write(struct.pack("<I", weight_format))
    f.write(struct.pack("<II", 0, 0))            # reserved


def _write_header_v2(f, *, num_heads, hidden, weight_dtype):
    """v2 header (32 bytes, version=2). variant=1 (RESIDUAL_MLP)."""
    f.write(b"H1BM")
    f.write(struct.pack("<I", 2))                       # version
    f.write(struct.pack("<I", V2_VARIANT_RESIDUAL_MLP)) # variant
    f.write(struct.pack("<I", num_heads))
    f.write(struct.pack("<I", hidden))
    f.write(struct.pack("<I", weight_dtype))
    f.write(struct.pack("<II", 0, 0))                   # reserved


def _emit_synthetic_zero(out_path, *, num_heads, hidden, vocab,
                         weight_format):
    if weight_format != WEIGHT_FORMATS["halo_v2"]:
        raise SystemExit("synthetic-zero only emits HALO_V2 today")
    packed_cols = (hidden + 3) // 4
    per_head_w_bytes = vocab * packed_cols
    with open(out_path, "wb") as f:
        _write_header(f, num_heads=num_heads, hidden=hidden,
                      vocab=vocab, weight_format=weight_format)
        for _ in range(num_heads):
            f.write(b"\x00" * per_head_w_bytes)
            f.write(b"\x00" * (vocab * 4))
    print(f"wrote synthetic-zero {out_path} "
          f"({num_heads}h, hidden={hidden}, vocab={vocab})",
          file=sys.stderr)


def _load_safetensors_heads(hf_dir: Path) -> List[Tuple[int, "object"]]:
    """Return [(head_idx, weight_tensor_numpy), ...] sorted by head_idx.

    Tries safetensors first; falls back to torch.load for `medusa_*.pt`."""
    heads = []
    try:
        from safetensors.torch import load_file as _load_st
        st_files = sorted(hf_dir.glob("medusa_*.safetensors"))
        for stf in st_files:
            # Filename pattern: medusa_<idx>.safetensors
            idx = int(stf.stem.split("_")[1])
            tensors = _load_st(str(stf))
            # Heads typically expose a single weight key, but tolerate
            # `lm_head.weight` or just `weight`.
            for cand in ("lm_head.weight", "weight",
                         "medusa_head.weight"):
                if cand in tensors:
                    heads.append((idx, tensors[cand].cpu().numpy()))
                    break
            else:
                raise RuntimeError(
                    f"no recognized weight key in {stf} "
                    f"(saw {list(tensors.keys())})")
    except Exception as e:
        # safetensors path failed — try torch .pt
        try:
            import torch  # type: ignore
            pt_files = sorted(hf_dir.glob("medusa_heads*.pt"))
            if not pt_files:
                raise RuntimeError(
                    f"no medusa_*.safetensors or medusa_heads*.pt in {hf_dir}: {e}")
            blob = torch.load(str(pt_files[0]), map_location="cpu")
            # Accept both list[tensor] and dict-of-tensors layouts.
            if isinstance(blob, list):
                heads = [(i, t.cpu().numpy()) for i, t in enumerate(blob)]
            elif isinstance(blob, dict):
                for k, v in blob.items():
                    try:
                        idx = int(str(k).split("_")[-1])
                    except ValueError:
                        idx = len(heads)
                    heads.append((idx, v.cpu().numpy()))
            else:
                raise RuntimeError(f"unrecognized .pt blob layout in {pt_files[0]}")
        except Exception as e2:
            raise SystemExit(
                f"failed to read MedusaBitNet heads from {hf_dir}: {e2}")
    heads.sort(key=lambda kv: kv[0])
    return heads


def _load_residual_mlp_pt(pt_path: Path):
    """Read parrishcorcoran/MedusaBitNet-2B-4T's torch checkpoint and return
    (num_heads, hidden, w_in_bf16_bytes_list, w_out_bf16_bytes_list).
    Each list element is the head's tensor as raw bf16 bytes ready to write."""
    import numpy as np
    import torch  # type: ignore

    blob = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "heads" not in blob:
        raise SystemExit(
            f"{pt_path}: expected dict with key 'heads' (parrishcorcoran "
            f"checkpoint format), got {type(blob).__name__}")
    heads = blob["heads"]
    w_in = heads["w_in"]   # [1, num_heads, hidden, hidden] bf16
    w_out = heads["w_out"]
    if w_in.dim() != 4 or w_out.dim() != 4 or w_in.shape != w_out.shape:
        raise SystemExit(
            f"{pt_path}: unexpected heads shape "
            f"w_in={tuple(w_in.shape)} w_out={tuple(w_out.shape)}; "
            f"need 4D [1, H, hidden, hidden]")
    if w_in.shape[0] != 1:
        raise SystemExit(
            f"{pt_path}: leading batch dim != 1 (got {w_in.shape[0]})")
    num_heads = int(w_in.shape[1])
    hidden    = int(w_in.shape[2])
    if w_in.shape[3] != hidden:
        raise SystemExit(
            f"{pt_path}: w_in is not square ({w_in.shape[2]}x{w_in.shape[3]})")
    if w_in.dtype != torch.bfloat16 or w_out.dtype != torch.bfloat16:
        # Tolerate fp16 / fp32 by casting back to bf16; the on-disk dtype
        # is bf16 so the runtime cast path stays consistent.
        print(f"WARNING: w_in/w_out dtype = {w_in.dtype} / {w_out.dtype} — "
              f"casting to bf16 before writing.", file=sys.stderr)
        w_in  = w_in.to(torch.bfloat16)
        w_out = w_out.to(torch.bfloat16)
    # bf16 has no native numpy dtype — view as int16 so the byte stream is
    # bit-identical when we tofile() it. Each head: [hidden, hidden] in
    # row-major ([out_dim, in_dim]) — matches rcpp_fp16_gemv W shape.
    w_in_views  = [w_in[0,  h].contiguous().view(torch.int16).numpy()
                   for h in range(num_heads)]
    w_out_views = [w_out[0, h].contiguous().view(torch.int16).numpy()
                   for h in range(num_heads)]
    return num_heads, hidden, w_in_views, w_out_views


def _emit_residual_mlp_v2(out_path: Path, pt_path: Path):
    num_heads, hidden, w_in_views, w_out_views = \
        _load_residual_mlp_pt(pt_path)
    print(f"v2 residual-MLP: {num_heads} heads, hidden={hidden}, dtype=bf16",
          file=sys.stderr)
    with open(out_path, "wb") as f:
        _write_header_v2(f, num_heads=num_heads, hidden=hidden,
                         weight_dtype=V2_DTYPES["bf16"])
        for h in range(num_heads):
            w_in_views[h].tofile(f)
            w_out_views[h].tofile(f)
    sz = out_path.stat().st_size
    print(f"wrote {out_path} ({sz/2**20:.1f} MB)", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-dir", help="HF checkpoint dir (parrishcorcoran/MedusaBitNet-2B-4T) — "
                                    "v1 path. Ignored on v2 (which reads --pt-path).")
    p.add_argument("--pt-path",
                   help="Path to medusa_heads_step2000.pt (parrishcorcoran "
                        "torch checkpoint) — REQUIRED for --variant residual_mlp.")
    p.add_argument("--out", required=True, help="Output .h1b-medusa path")
    p.add_argument("--variant", default="vocab",
                   choices=["vocab", "residual_mlp"],
                   help="On-disk variant. vocab=v1 per-head ternary projection "
                        "(legacy). residual_mlp=v2 dense fp16 w_in/w_out "
                        "with shared base.lm_head.")
    p.add_argument("--weight-format", default="halo_v2",
                   choices=WEIGHT_FORMATS.keys(),
                   help="(v1 only) base ternary packing format")
    p.add_argument("--synthetic-zero", action="store_true",
                   help="Emit zero-filled heads (for end-to-end smoke "
                        "testing without real weights). Requires "
                        "--num-heads, --hidden, --vocab (v1 only).")
    p.add_argument("--num-heads", type=int)
    p.add_argument("--hidden", type=int)
    p.add_argument("--vocab", type=int)
    args = p.parse_args(argv)

    weight_format = WEIGHT_FORMATS[args.weight_format]

    if args.variant == "residual_mlp":
        if args.synthetic_zero:
            raise SystemExit("--synthetic-zero is v1-only today")
        if not args.pt_path:
            p.error("--variant residual_mlp requires --pt-path")
        pt_path = Path(args.pt_path).expanduser().resolve()
        if not pt_path.is_file():
            raise SystemExit(f"not a file: {pt_path}")
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _emit_residual_mlp_v2(out_path, pt_path)
        return

    # ── v1 path (legacy) ──
    if args.synthetic_zero:
        if args.num_heads is None or args.hidden is None or args.vocab is None:
            p.error("--synthetic-zero requires --num-heads, --hidden, --vocab")
        _emit_synthetic_zero(args.out,
                             num_heads=args.num_heads,
                             hidden=args.hidden,
                             vocab=args.vocab,
                             weight_format=weight_format)
        return

    if args.hf_dir is None:
        p.error("--hf-dir is required unless --synthetic-zero")

    hf_dir = Path(args.hf_dir).expanduser().resolve()
    if not hf_dir.is_dir():
        raise SystemExit(f"not a directory: {hf_dir}")
    heads_raw = _load_safetensors_heads(hf_dir)
    if not heads_raw:
        raise SystemExit(f"no Medusa heads found in {hf_dir}")
    num_heads = len(heads_raw)
    vocab, hidden = heads_raw[0][1].shape
    for idx, w in heads_raw:
        if w.shape != (vocab, hidden):
            raise SystemExit(
                f"head {idx} shape {w.shape} disagrees with head 0 ({vocab}, {hidden})")
    print(f"converting {num_heads} heads, vocab={vocab}, hidden={hidden} "
          f"→ weight_format={args.weight_format}", file=sys.stderr)

    if weight_format != WEIGHT_FORMATS["halo_v2"]:
        raise SystemExit(
            "only --weight-format=halo_v2 is implemented in v0; the engine's "
            "lazy-attach path supports the other formats via the same loader.")

    with open(args.out, "wb") as f:
        _write_header(f, num_heads=num_heads, hidden=hidden,
                      vocab=vocab, weight_format=weight_format)
        for idx, w in heads_raw:
            packed, scales = _ternarize_head(w, hidden, vocab)
            packed.tofile(f)
            scales.tofile(f)

    print(f"wrote {args.out} ({num_heads}h, vocab={vocab}, hidden={hidden})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
