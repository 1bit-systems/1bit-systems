# `.h1b-medusa` sidecar format

This file documents two on-disk variants:

- **v1** — per-head (vocab, hidden) ternary projection. Synthetic-zero only
  in tonight's measured runs; mismatch with parrishcorcoran upstream.
- **v2** — residual-MLP topology with shared base.lm_head. Mirrors
  parrishcorcoran/MedusaBitNet-2B-4T (4 heads, hidden=2560, w_in/w_out
  dense [hidden, hidden]). Loader auto-dispatches on the `version` u32.

# v1 (legacy)


Companion file for a base `.h1b` BitNet model carrying speculative-decoding
heads. Sits alongside the `.h1b` and is loaded by
`rcpp_medusa_load_h1b_sidecar()` only after the base model has been loaded
successfully (engine-side cross-checks `hidden_size` + `vocab_size` against
the base).

All multi-byte integers are little-endian. Floats are IEEE-754 fp32.

## Header (32 bytes)

```
offset  size   type    field
0       4      char[4] magic           = "H1BM"
4       4      u32     version         = 1
8       4      u32     num_heads       (typical: 4 for MedusaBitNet-2B-4T)
12      4      u32     hidden_size     (must equal base model)
16      4      u32     vocab_size      (must equal base model)
20      4      u32     weight_format   (rcpp_weight_format_t enum value)
24      4      u32     reserved        (must be 0)
28      4      u32     reserved        (must be 0)
```

`weight_format` mirrors the base-model `rcpp_weight_format_t`. Today only
`RCPP_WEIGHT_FORMAT_HALO_V2 = 0` is exercised; the other tags are accepted at
the loader level and dispatched through the same per-format decoder paths
already present for the base weights.

## Per-head payload

Repeated `num_heads` times, contiguously, in head-index order:

```
packed_weights : opaque bytes — same on-disk shape as a base-model
                 ternary-linear weight tensor with output dim = vocab_size,
                 input dim = hidden_size, in the format identified by
                 `weight_format`. Sizes:
                    HALO_V2     : vocab_size * ((hidden_size + 3) / 4) bytes
                    SHERRY_I8   : vocab_size * (hidden_size * 5 / 32) bytes
                    SHERRY_FP16 : vocab_size * (hidden_size * 5 / 32) bytes
                    TQ1         : vocab_size * (((hidden_size + 19)/20) * 20 / 5) bytes
                    BONSAI_Q1   : (no row_scales tensor — inline blocks; head
                                  payload is `vocab_size * blocks_per_row * 18` bytes)
                    BONSAI_TQ2  : same shape but * 34 bytes/block

row_scales     : f32 [vocab_size]   (omitted iff the format is one of the
                                     Bonsai inline-scale formats)
```

Each Medusa head is a per-token classifier that, given the model's
final-layer hidden state at position `t`, predicts the token at position
`t + 1 + h` for head index `h`. Heads are independent — there is NO
residual connection or LM-head sharing in this v1 wire format. The
synthetic-zero writer used for kernel smoke tests writes zero weights and
zero scales, which collapses every head's prediction to the argmax of the
all-zeros logit tensor (i.e. token id 0). That is intentional: it lets the
end-to-end glue (loader + dispatch + verify-prefix) be tested without real
weights, accepting exactly one token per round.

## Loader semantics

- `rcpp_medusa_load_h1b_sidecar(path, &base_model, &out_heads)` returns
  `RCPP_INVALID_ARG` on any of: missing magic, wrong version, mismatched
  hidden_size or vocab_size, malformed payload size.
- `RCPP_HIP_ERROR` indicates a device upload failed (rare; usually OOM).
- All weight + scale buffers are device-resident on success; release with
  `rcpp_medusa_free_heads(&heads)`.

## Python converter (offline, not a runtime path)

`tools/medusa-pack/medusa_safetensors_to_h1bm.py` consumes a HuggingFace
checkpoint dir matching the
[parrishcorcoran/MedusaBitNet-2B-4T](https://huggingface.co/parrishcorcoran/MedusaBitNet-2B-4T)
layout (`medusa_*.safetensors`, one file per head, `lm_head` is shared
with the base) and emits a v1 `.h1b-medusa`. Per Rule A this is offline
tooling only — the runtime never invokes Python.

# v2 (residual-MLP, shared lm_head)

Wire format for parrishcorcoran-style MedusaBitNet heads. Each head is a
single dense MLP layer with a residual connection, evaluated against the
backbone's final hidden state, then projected through the **shared**
`base.lm_head` to emit logits. There is no per-head vocab tensor.

Forward (per head k, applied to hidden h):
```
next_h = h + w_out_k @ SiLU(w_in_k @ h)
logits_k = base.lm_head @ next_h
```

## v2 header (32 bytes)

```
offset  size   type    field
0       4      char[4] magic           = "H1BM"
4       4      u32     version         = 2
8       4      u32     variant         = 1   (RESIDUAL_MLP; 0=legacy v1)
12      4      u32     num_heads             (4 for MedusaBitNet-2B-4T)
16      4      u32     hidden_size           (must equal base.hidden_size)
20      4      u32     weight_dtype          (0=bf16, 1=fp16,
                                              2=halo_v2_ternary, 3=sherry_i8)
24      4      u32     reserved        = 0
28      4      u32     reserved        = 0
```

`vocab_size` is intentionally absent — the engine reuses
`base.embedding_dev` (tied lm_head) on the v2 path, so the sidecar can be
loaded against any base with matching `hidden_size`.

## v2 per-head payload

Repeated `num_heads` times in head-index order. For each head:

```
w_in[hidden_size, hidden_size]   bytes_for_dtype(hidden_size, weight_dtype)
w_out[hidden_size, hidden_size]  bytes_for_dtype(hidden_size, weight_dtype)
```

`bytes_for_dtype`:
- `bf16`              : `hidden² · 2` bytes (cast host-side to fp16 at load).
- `fp16`              : `hidden² · 2` bytes (uploaded directly).
- `halo_v2_ternary`   : `hidden · ((hidden+3)/4)` bytes — reserved, no
                        runtime path yet.
- `sherry_i8`         : `hidden · (hidden·5/32)` bytes — reserved, no
                        runtime path yet.

For ternary / Sherry dtypes, per-row fp32 scales (size `hidden · 4` bytes
per tensor) follow each packed tensor. Today only bf16 + fp16 land on
the runtime path; the engine returns `RCPP_INVALID_ARG` on the reserved
tags.

## v2 loader semantics

`rcpp_medusa_load_h1b_sidecar()` dispatches on the `version` u32:
- `version=1` → legacy v1 path (above).
- `version=2` → v2 path. Cross-checks `hidden_size` against the base; the
  `vocab_size` slot is unused (engine reuses `base.embedding_dev`).

On success `out->variant == RCPP_MEDUSA_VARIANT_RESIDUAL_MLP` and each
`out->heads[k]` carries `w_in_dev` + `w_out_dev` as fp16 device buffers.
The legacy `packed_dev` / `row_scales_dev` slots stay nullptr on v2.

## v2 Python converter

`tools/medusa-pack/medusa_safetensors_to_h1bm.py --variant residual_mlp`
consumes parrishcorcoran's `medusa_heads_step2000.pt` (a torch dict with
`heads.w_in[1, 4, 2560, 2560]` and `heads.w_out[1, 4, 2560, 2560]` in
bf16) and emits a v2 sidecar with `weight_dtype=bf16`. The runtime casts
to fp16 at load time. Pure offline tooling — no Python at runtime.
