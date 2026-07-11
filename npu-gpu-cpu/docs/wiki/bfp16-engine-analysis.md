# BFP16 Engine Analysis

## Finding: BFP16 fused engine was always degenerate

After fixing the O projection dequant bug, the BFP16 fused engine STILL produces
degenerate output (repeating token 77653 with hidden state blowup).

### Root Cause

The `flashlf3/bfp16` format (`v8bfp16ebs8`) used by the BFP16 xclbins has
fundamental precision limitations:

- **7-bit mantissa** + shared 8-bit exponent per group of 8 values
- **Truncation rounding** (not round-to-nearest) in all quantize/dequant paths
- **17% max relative error** measured per single GEMM (128×1024×2048)
- **112 cumulative GEMM calls** (4 GEMMs/layer × 28 layers) compound error
- Hidden state diverges from ~1000 (L19) to ~19000 (L28)

### Verification

All BFP16 variants tested:
- `npu_engine_v3`, `v4`, `v5`, `v6` (original BFP16 engines) — ALL degenerate
- `npu_engine_fused.cpp` (O projection fixed) — STILL degenerate
- `test_128_rand` — 17% max relative error on single BFP16 GEMM

### Working Alternative

The INT8 engine (`npu_engine_i8.cpp`) uses a different format:
- INT8 xclbins (`matmul_i8_i16`) with per-matrix quant scale
- 8-bit precision per value independently (no shared exponent)
- Verified: 174ms/tok, 8 diverse tokens, no NaN
- Fused INT8 variant: `npu_engine_fused_i8.cpp` at 248ms/tok

### Background: Why was BFP16 used?

The torch2aie toolchain template only supported BFP16 output types for AIE
matrix multiplication. The INT8 format required custom MLIR generators
(`n1_core_i8.py`, `n1_core_i8_v2.py`) that produce `matmul_i8_i16` xclbins.
These were built during the INT8 inference work and work correctly.

### Scaling: Multi-token and 2-layer batch

BFP16 precision limits mean this will never work for full model inference.
Focus all effort on INT8-based approaches:
1. Multi-token batch (4 tokens per NPU call): ~55ms/tok target
2. 2-layer fused xclbins: ~170ms/tok via fewer kernel launches
3. INT8 DMA stride fix: ~100ms/tok single-token
