# Fused XCLBIN Integration — Final Status July 2, 2026

## What's Working

| Component | Status |
|-----------|--------|
| Fused xclbin compiled (374KB full-layer) | ✅ |
| Kernel groups mapped (0-7 all OK) | ✅ |
| Kernel dispatches (no crash) | ✅ |
| BOs allocated correctly (9.4MB weights) | ✅ |
| AUX prefix correct (1216 dwords) | ✅ |

## What's Blocking

The Q4NX weight stream requires chunk scheduling via FLM's
_projection_stream_from_schedule(). This function maps
(phase, block, input_chunk, group, patch, row_in_patch) 
to a specific chunk index in the model's Q4NX storage.

Our packer reads chunks correctly (1600 chunks, 5120 bytes each)
but replicates them 8× (4 columns × 2 patches) instead of
indexing them by schedule entry. Correct output: 9.4MB per layer.

## Path Forward

FLM's Python infrastructure can pack correctly but requires:
- Running FLM's layer_weight_stream() with 0.6B model
- Saving weights_i32 to disk
- mmap'ing into our C++ engine

This is ~200 lines of Python glue code + ~50 lines of C++ dispatch.
Estimated 4-8 hours to produce working fused decode.

## Action Items
1. Write Python script using FLM's qwen3_model.layer_weight_stream()
2. Save packed weights as binary file  
3. C++ engine: mmap file → XRT BO → dispatch
4. Benchmark: expected <5ms/tok batch step at M=32

## v12 Engine (Production)
97 tok/s. 24× speedup. Beats FLM Kraken Point (66.5 tok/s).
Standalone INT8 GEMM xclbins. OpenMP attention + LM head.
Zero Python. Pure C++.


## UPDATE (07:30): Q4NX Packer v2 — Correct Size, DMA Timeout

Weight packer v2 produces correct-size output (9,835,264 bytes = 2,458,816 dwords)
matching the fused xclbin MLIR exactly:

```
AUX: 4864 bytes (1216 dwords)
Stream: 9,830,400 bytes (2,457,600 dwords)
Total: 9,835,264 bytes (2,458,816 dwords) ✅
```

Packer at: `/home/bcloud/npu-sandbox/npu-infer/tools/pack_fused_v2.py`

**Algorithm:**
1. Dequantize I8 weights → f32
2. Re-quantize to Q4NX format (symmetric 4-bit per group)
3. Pack in FLM schedule order (group→patch→schedule_entry→row_in_patch)
4. Prepend AUX prefix (norm weights + RoPE angles in bf16)

**Why DMA still times out (63s):**
The fused xclbin's main16 kernel (`qwen3_decode_kernels.cc`) implements FLM's
proprietary Q4NX dequant algorithm with specific scale/zero semantics:
- Scales are per-row per-group (bf16)
- Zeros are per-row per-group (bf16)  
- Weights are packed as two U4 lanes of 16 rows each
- Dequant: `(w * scale + zero)` with float accumulation

Our Q4NX quantizer produces valid values but the specific scale/zero
computation likely differs from FLM's training pipeline, causing NaN/Inf
accumulations that stall AIE compute tiles.

**Resolution:** The Q4NX format is not a generic 4-bit quantization — it's FLM's
proprietary format tuned to their training pipeline. Correct integration requires
either using FLM's weight generator directly or matching their quantization
parameters exactly. This is a model-converter project, not an engine project.

**Action:** Continue production engine (v12 standalone INT8, 97 tok/s).
Schedule Q4NX format conversion as a separate workstream.

---

## UPDATE (2026-07-03): This Diagnosis Was Wrong — It Was a Schedule Bug, Not a Format Bug

The "requant produces NaN, wrong quantization pipeline" theory above applied to `pack_fused_v2.py`
(dequant → re-quant, genuinely mismatched vs. FLM's training-time parameters). It does **not**
apply to `pack_fused_v3.py` / the current `engine/npu/src/q4nx_stream.cpp`, both of which read raw
Q4NX chunks directly from `model.q4nx` with zero dequant/requant — using FLM's own bytes verbatim.

The real remaining bug in `q4nx_stream.cpp` was schedule ordering (naive per-column replication
instead of the group/patch/row-indexed schedule), now fixed and verified byte-identical to the
validated `pack_fused_v3.py` output. See `docs/FUSED-INTEGRATION-BLOCKER.md`'s 2026-07-03 update
for the full schedule formula and for what's still blocking (an O/UP/GATE/DOWN dataflow deadlock
and a separate QKV numeric-correctness bug, both isolated to the Chess-compiled kernel — not the
weight format).
