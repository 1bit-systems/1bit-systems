# v12 Correctness Blocker — July 3, 2026

## Context

Requested task: swap the production daemon's NPU backend from proxying to FLM (a
proprietary, closed-source binary) over to `npu_engine_cb.cpp` (v12, our own C++
engine, benchmarked at 97 tok/s — faster than FLM's 94 tok/s, and "Zero Python" per
the project's own branding). Built a persistent-server variant (`1bit.engine`,
`engine/npu/src/npu_engine_server.cpp`) so the daemon doesn't pay the ~8s model-load
cost per request.

Before wiring it into the daemon, sanity-checked actual chat output against FLM for
the same prompt ("What is 2+2?"). FLM: `" 4."` — correct. v12 (byte-identical
reproduction of the unmodified original): complete garbage
(`',W Tusśl各位alte=resultЛЬPCODE...'`). This surfaced that v12 was **never actually
validated for output quality** — every doc and benchmark in this repo checks tok/s
and "doesn't crash," never coherence. The 97 tok/s figure is real; the output behind
it was never real.

## What Was Found and Fixed

Three real, independently confirmed bugs, all still present as of this write-up
after fixing them — meaning at least one more bug remains (see below).

### 1. LM head weight substitution

`npu_engine_cb.cpp` correctly dequantizes `lm_head.weight` from the Q4NX file, then
immediately discards the result (`free(dequant_i8_to_float(i8p(lo_off),18992,&lr,&lc));`)
and computes the final vocabulary projection using the *input* embedding table
instead — implicitly assuming tied embeddings. Confirmed via the Q4NX header's
`data_offsets`: `model.embed_tokens.weight` (`BF16 [151936,1024]`, offset 0) and
`lm_head.weight` (`I8 [18992,5120]`, offset 311296000) are stored completely
separately. Qwen3-0.6B does **not** tie embeddings. The model computes a reasonable
final hidden state, then reads logits off the wrong matrix.

**Fix**: keep the dequantized `lm_head.weight`, use it for the final projection.

### 2. Weight-packing transpose

`dequant_i8_to_float(_ex)` returns row-major `[out_features, in_features]` (verified
from the tile-index arithmetic in `dequant_q4nx.c`: `linear_idx = (tile_row*32+lr) *
out_cols + (tile_col*256+col)` — classic row-major with `out_features` as the
slower-varying dimension). The GEMM dispatch (`packB()`/`go()`) needs the transpose
of that — `[in_features, out_features]`, since it computes `A[tokens,in] @
B[in,out]`. The original packing loop read the dequantized buffer with an
`out_features`-sized stride as if it were already `[in,out]`, silently scrambling
every weight matrix (Q/K/V/O/Gate/Up/Down, all of them) while still producing
finite, non-NaN, plausible-magnitude numbers — exactly why "doesn't crash" was never
sufficient validation. Confirmed empirically: dequanted Q-weight values at several
`(out_idx, in_idx)` positions matched the real HuggingFace model's weights (within
expected 4-bit quantization noise, ~5-20% relative error) when read as `[out,in]`,
and did not match when read as `[in,out]`.

A related, separate bug: O-proj and Down-proj dequant calls used the default
`dequant_i8_to_float()` wrapper, which hardcodes `in_features=1024`. O-proj's real
`in_features` is 2048 (`NH*HD`) and Down-proj's is 3072 (`IM`) — using the wrong
value scrambles the *dequant tiling itself*, before the transpose issue even
applies. Fixed by calling `dequant_i8_to_float_ex()` with the correct value for
these two projections.

**Fix**: `transpose_pack()` helper, correctly transposing dequant's `[out,in]`
output into the `[in,out]` layout `packB()`/`go()` expect, plus correct
`in_features` for O/Down.

### 3. Activation quantization clipping

`go()`'s activation-to-INT8 quantization used a hardcoded scale of `5.0f/127.0f`,
assuming activations stay within `[-5,5]`. Measured actual post-RMSNorm activations
at layer 0: range `[-8.24, 7.01]` — any value past ±5 silently clips to ±127 in the
INT8 quantization. This happens at every GEMM call, every layer, compounding across
all 28 layers.

**Fix**: `dynamic_ascale()` — per-call amax-based scale, matching the same approach
`packB()` already uses for weights, instead of a fixed constant.

## What's Still Broken

With all three fixes applied (individually and combined, in various combinations
with both RoPE conventions below), chat output is still incoherent — degenerating
into repetitive tokens (`iaux`, `也不例外`) rather than plausible English/Chinese.
At least one more bug remains.

**Ruled out** (verified correct via ground-truth comparison against the real
HuggingFace `Qwen/Qwen3-0.6B` model):
- Embedding table lookup — matches HF model within expected BF16 rounding
- RoPE theta (1000000) and head config (H=1024, NH=16, NKV=8, HD=128) — match
  `AutoConfig` exactly
- GQA head mapping (`kvh = hh / GQA`) — matches HF's `repeat_kv` convention
- K/V extraction offsets from the packed QKV GEMM output — consistent with the
  packing offsets

**Tested, inconclusive**:
- RoPE rotation convention — tried both the original interleaved-pairs convention
  (`(d, d+1)` rotate together) and HuggingFace's actual "rotate_half" convention
  (`(i, i+head_dim/2)` rotate together, which is what real Qwen3 models use,
  confirmed during the earlier Eagle3 draft-model work this session). Neither
  alone, nor combined with the three fixes above, restored coherence. `ra_interleaved`
  and `ra_rothalf` are both left in `npu_engine_server.cpp` (currently wired to
  interleaved) for whoever picks this back up.

**Not yet checked** — most likely remaining suspects:
- The compiled NPU kernel binaries (`.xclbin` files) themselves. These are opaque
  MLIR-compiled binaries; debugging them would need the AI Engine Simulator, which
  is blocked on this machine (missing `aie2p_8x4_device.json` for NPU2 — see
  `docs/FUSED-INTEGRATION-BLOCKER.md`'s aiesimulator section from the same
  investigation thread).
- Attention softmax scaling or numerical precision within the attention loop itself
  — not yet traced layer-by-layer against a Python reference.

## Status

**Do not wire `1bit.engine` (or any v12 variant) into the production daemon.** FLM
proxy stays in production (`daemon/npu-gpu-cpud.py`, port 9090) until this is fully
resolved and re-verified against real chat prompts, not just dispatch speed.

The three fixes above are real, confirmed, and applied to all three copies of this
dispatch logic in the repo:
- `engine/npu/src/npu_engine_cb.cpp` (production v12 reference)
- `engine/npu/src/npu_engine_server.cpp` (new persistent-server variant, not yet
  used anywhere)
- `spec-decode/engine/npu_target_model.h` (target-model dispatch for the
  speculative-decoding work) — important since this affects the eventual
  acceptance-rate measurements from that separate effort.

## Next Steps (for whoever picks this up)

1. Layer-by-layer numerical trace: build a from-scratch Python reference for one
   transformer layer (RMSNorm → QKV → RoPE → attention → O-proj → SwiGLU MLP) using
   the *dequantized* Q4NX weights (bypass the NPU kernel entirely, pure CPU/numpy),
   and diff v12's intermediate activations against it at each stage. This will
   localize whether the remaining bug is host-side (attention math, RoPE) or in the
   NPU kernel binaries.
2. If the trace shows host-side math is correct through the full layer, the bug is
   almost certainly in the `.xclbin` kernels — would need either the source MLIR (if
   it still exists somewhere) or a working AI Engine Simulator to progress further.
3. Given the effort already sunk here (this session) and in the fused-xclbin
   investigation (a separate, earlier session) — both point at real gaps in this
   machine's NPU debugging tooling. Getting the AI Engine Simulator working for
   NPU2 (or acquiring/building the missing device JSON) would unblock multiple
   stalled investigations at once, not just this one.
