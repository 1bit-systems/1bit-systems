# DSpark/DFlash Deploy-Time Forward Pass Blocker — July 18, 2026

## Context

Requested task: extend `spec-decode/draft/mtp_draft.h` (currently Eagle3-only) to
also support DSpark and DFlash draft models (`spec-decode/configs/dspark_qwen3_0.6b.py`,
`spec-decode/configs/dflash_qwen3_0.6b.py`). Before porting, traced the actual
deploy-time reference implementation in `~/DeepSpec`
(`deepspec/eval/dspark/draft_ops.py::forward_dspark_draft_block` →
`Qwen3DSparkModel._forward_backbone` → `Qwen3DSparkAttention.forward`) — by
actually running it with synthetic weights/inputs, not just reading it. It
crashes. Stopped before implementing C++ against it: found two independent
problems in the Python reference, and neither DSpark nor DFlash has a trained
checkpoint anywhere on this machine to validate a guessed fix against.

`deepspec/modeling/dspark/qwen3/modeling.py` has exactly one commit ("first
init") in its `~/DeepSpec` git history — consistent with this deploy path
never having actually been exercised end-to-end by anyone yet. Eagle3's
C++ port (already working, trained to step 3370) doesn't hit either of these
issues because its architecture is materially simpler (single draft layer,
no separate target-context K/V injection, no block-diffusion masking).

## What was found

### Issue 1 — RoPE shape mismatch

`deepspec/modeling/dspark/qwen3/modeling.py:35-41`:

```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)          # no slicing on k
    return q_embed, k_embed
```

`k` (line 108) is `torch.cat([k_proj(target_hidden_states), k_proj(hidden_states)])`
— length `ctx_len + q_len`. But `cos`/`sin` come from
`self.rotary_emb(hidden_states, position_ids)` (line 375) where `hidden_states`
is the draft block's own `noise_embedding`, length `q_len` only. So
`k_embed = k * cos` multiplies a `[ctx_len+q_len]`-long tensor against a
`[q_len]`-long one. Reproduced directly (see script below): `RuntimeError: The
size of tensor a (7) must match the size of tensor b (4) at non-singleton
dimension 2` with `ctx_len=3, q_len=4`.

### Issue 2 — `draft_position_ids` length doesn't match `draft_input_ids` length

`deepspec/eval/dspark/draft_ops.py:32-34`:

```python
draft_position_ids = position_ids[
    :, past_key_values_draft.get_seq_length() : start + block_size
]
```

`draft_input_ids` (the actual token IDs — `evaluator.py:109-115`, `_propose()`)
is always exactly `block_size` long. For `draft_position_ids` to also be
`block_size` long, `past_key_values_draft.get_seq_length() == start` must hold
on every call. Tracing the outer loop (`deepspec/eval/base_evaluator.py:307-439`,
`generate_decoding_sample`):

- `start` initializes to `num_input_tokens` (prompt length) — **not 0**
  (line 359).
- `context.past_key_values_draft` is a single fresh `DynamicCache()` from
  `_init_context()` (`evaluator.py:92`) — length **0**.

So on the very first call, `draft_position_ids` has length `start + block_size`
(prompt length + block_size), not `block_size` — broken independently of
Issue 1, before any rounds even happen. Nothing in `_init_context()` or the
surrounding loop pre-populates `past_key_values_draft` to length `start`.

## Reproduction

```python
import os, copy, torch
os.environ['DSPARK_DENSE_MASK'] = '1'  # flex_attention needs Triton, crashes on gfx1151
from transformers import AutoConfig, DynamicCache
from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel
from deepspec.eval.dspark.draft_ops import forward_dspark_draft_block

torch.manual_seed(0)
target_layer_ids = [0, 1]; block_size = 4; mask_token_id = 999
target_cfg = AutoConfig.for_model("qwen3", hidden_size=32, num_attention_heads=4,
    num_key_value_heads=2, head_dim=8, intermediate_size=64, num_hidden_layers=3,
    vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1000000.0, attention_bias=False,
    layer_types=["full_attention"]*3, sliding_window=None)
draft_cfg = copy.deepcopy(target_cfg)
draft_cfg.architectures = ["Qwen3DSparkModel"]; draft_cfg.num_target_layers = 2
draft_cfg.num_hidden_layers = 2; draft_cfg.block_size = block_size
draft_cfg.tie_word_embeddings = False; draft_cfg.layer_types = ["full_attention"]*2
draft_cfg._attn_implementation = "eager"; draft_cfg.mask_token_id = mask_token_id
draft_cfg.target_layer_ids = target_layer_ids; draft_cfg.num_anchors = 4
draft_cfg.enable_confidence_head = False; draft_cfg.markov_rank = 0

model = Qwen3DSparkModel(draft_cfg).eval()
target_hidden_states = torch.randn(1, 3, 2 * 32)  # ctx_len=3
draft_input_ids = torch.full((1, block_size), mask_token_id, dtype=torch.long)
draft_input_ids[:, 0] = 5
position_ids = torch.arange(0, 20).unsqueeze(0)
past_kv = DynamicCache()

forward_dspark_draft_block(model, draft_input_ids=draft_input_ids,
    position_ids=position_ids, past_key_values_draft=past_kv,
    target_hidden_states=target_hidden_states, start=0, block_size=block_size)
# RuntimeError: The size of tensor a (7) must match the size of tensor b (4)
#   at non-singleton dimension 2
```

## Why this blocks the C++ port

Attempted a fix for Issue 1 (per-segment RoPE: context keys keep the absolute
positions they had when the target model produced them, draft keys/queries
get fresh block-relative positions, applied before concatenation — the most
natural read of the cross-attention structure). That fix alone surfaced
Issue 2 immediately on the very next test. Two independent, stacked
assumptions with nothing to verify either against (no trained checkpoint for
DSpark or DFlash exists anywhere) means any C++ port right now would encode
guesses about unwritten intended behavior as if they were verified — exactly
what this repo's `#294`/benchmark-honesty work has been trying to move away
from. Stopping here rather than compounding the guesswork.

## What's not blocked

- **Eagle3**: unaffected, already working (`spec-decode/draft/mtp_draft.h`,
  trained checkpoint at step 3370). This blocker is DSpark/DFlash-specific.
- The **training-time** forward pass (`Qwen3DSparkModel.forward()`, used by
  `Qwen3DSparkTrainer`) is a separate code path from the ones above and isn't
  known to be affected — training can presumably still proceed; it's
  specifically the post-training deploy/inference path that's broken.

## Suggested next steps

1. Confirm the intended RoPE design with whoever wrote
   `deepspec/modeling/dspark/qwen3/modeling.py` — is per-segment RoPE (context
   keeps original positions, draft gets fresh ones, applied pre-concat) the
   right fix, or is there a different intended design?
2. Clarify the intended `past_key_values_draft`/`start`/`position_ids`
   invariant in `forward_dspark_draft_block` — does `_init_context()` need to
   pre-seed the cache to length `start`, or should the slicing use a
   locally-relative index instead of the globally-relative `start`?
3. Once both are resolved and DSpark or DFlash has even a short/undertrained
   checkpoint, re-run the repro above (extend it into a regression test) to
   confirm, then this doc's blocker is cleared and the `mtp_draft.h`
   extension can proceed with a real reference to verify against.
