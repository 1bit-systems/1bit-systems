# Fused XCLBIN Integration Blocker — July 2, 2026

## What Works

| Component | Status | Detail |
|-----------|--------|--------|
| QKV prefix xclbin | ✅ Compiled | 253KB, 4 runtime args |
| Full-layer xclbin | ✅ Compiled | 374KB, 5 runtime args |
| Kernel group mapping | ✅ Verified | Groups 0-7 all accessible |
| Kernel dispatch | ✅ Runs | ~63s DMA timeout (not crash) |
| C++ engine | ✅ Compiles | BOs alloc + dispatch OK |

## What's Blocking

The **Q4NX weight format** required by the fused xclbin is fundamentally
different from our flat INT8 weight format. The xclbin's runtime sequence
uses 512 specific DMA descriptors with precise byte offsets and lengths
that read Q4NX-compressed weight chunks in FLM's streaming format.

Our model stores weights as dequantized flat bf16 → packed as flat INT8.
FLM's format is Q4NX: 5120-byte chunks (scales + zeros + packed 4-bit data)
laid out per-column, per-patch, per-phase, using row-major interleaving.

## What's Needed for Full Integration

1. **Q4NX weight packer** — Port FLM's `_projection_stream_from_schedule()`
   from Python to C++. ~200 lines of complex weight interleaving logic.

2. **KV cache layout** — FLM uses block-major (128-token blocks). Our engine
   uses linear layout. The fused xclbin DMA descriptors expect block-major.

3. **AUX data prefix** — RMS norm weights + QK norm + RoPE cos/sin at specific
   byte offsets within the weights BO.

4. **Runtime sequence arguments** — The kernel needs exactly 5 buffers with
   exact sizes matching the MLIR generation.

All of this is implemented in FLM's `qwen3_8b_decode_layer_runner.py`.
Porting to C++ is ~500 lines of careful weight format conversion.

## Decision

**Keep standalone INT8 GEMM xclbins for production** (v12 engine at 97 tok/s).
Fused xclbin integration is a separate project requiring Q4NX format support.
The fused xclbins compile and prove the toolchain works — the blocker is
weight format, not hardware or toolchain.

## Path Forward

1. Write a Python script that uses FLM's runner directly with our 0.6B model
2. Once validated numerically, port the weight packer to C++
3. Integrate into multi-layer decode loop
4. Expected: ~5ms/tok batch step, ~3ms/tok effective at M=32
