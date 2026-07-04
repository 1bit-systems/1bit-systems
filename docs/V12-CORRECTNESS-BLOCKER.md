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

With all three original fixes plus the RoPE convention fix applied, the host-side
math in all 4 dispatch copies now matches the HuggingFace Qwen3 reference:
- LM head → correctly uses untied lm_head.weight (not embed tokens)
- Weight packing → correctly transposes dequant [out,in] to [in,out] for NPU GEMM
- Activation scale → dynamic per-call amax, no hardcoded clipping
- RoPE → uses HuggingFace rotate_half convention (verified against HF modeling source)

**The remaining risk is in the compiled NPU kernel binaries (.xclbin files).**
These are opaque MLIR-compiled binaries; debugging them would need the AI Engine
Simulator, which is blocked on this machine (missing `aie2p_8x4_device.json` for
NPU2 — see `docs/FUSED-INTEGRATION-BLOCKER.md`'s aiesimulator section).

If the host-side math is correct (as we believe it now is), and output is still
incoherent, the bug must be in the xclbin kernels themselves — either the INT8
GEMM matrix multiply or the quantization/dequant logic inside the NPU compute tiles.
Verification approach: run `tools/layer_trace.py` to produce a Python reference
trace of any layer's intermediates, and diff the C++ engine's intermediates against
it. If the C++ intermediates match through the full layer, the xclbin kernels are
the only remaining variable.

## Status (July 4, 2026, after RoPE fix)

**Do not wire `1bit.engine` (or any v12 variant) into the production daemon** until
the xclbin kernels have been validated against a Python reference trace. FLM
proxy stays in production (`daemon/npu-gpu-cpud.py`, port 9090) until this is resolved.
