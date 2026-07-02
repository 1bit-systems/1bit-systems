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

