# Debugging Summary — 2026-07-01 Systematic Debugging Session

## What We Know

### Weight Dequantization
- `dequant_i8_to_float` correctly dequantizes Q4NX format (INT4 with little-endian BF16 scales/zero-points)
- The dequantized gate weights for layer 0 are 94% negative (mean -0.078, range [-0.734, 0.096])
- **The dequantization IS correct** — tiny scales like 0.0187, 0.0073 give valid weight values around -0.03 to -0.3
- All GEMM kernels verified accurate within float32 accumulation error (< 0.00008 max diff)

### The Bug — Gate Values Always Negative
- Gate projection output is ALWAYS negative (mean -31.7, min -76.5) for ALL 3072 outputs
- Up projection is ALSO always negative (mean -23.0, min -31.8)
- SiLU(gate) ≈ 0 for all entries (since gate << -10)
- MLP contributes ~0 to the residual stream
- Hidden state accumulates to ±60M by end of 9-token prefill
- Final token is always 82291 (attractor from blown-up hidden state)

### Matching CPU Reference
- CPU reference produces EXACTLY the same results as GPU engine
- Both use NPU-style weight layout (gw[k*3072+i] for gate, direct memcpy for O/Down)
- Both produce all-negative gate outputs
- Both produce token 82291, not 919 (NPU's first token)

### NPU Engine Works (produces 919 996 185...)
- Built from `engine.cpp` + `model.c` using xclbins
- Uses completely different weight packing (npu_pack_weight_bo → AIE kernel handles I8)
- `model.c`'s dequant function (npu_dequant_block) is BROKEN for this format (produces NaN/inf)
- **The NPU's AIE kernels handle I8→BF16 dequantization internally**

## Root Cause Hypothesis

The NPU's AIE kernels (in the xclbin files) use a DIFFERENT dequantization method than
`dequant_i8_to_float`. The Q4NX file stores weights in a proprietary format that the
pre-compiled xclbin kernels understand natively. Our `dequant_i8_to_float` function 
INFERS a tile format (256 scales + 256 zps + 4096 INT4 per 5120-byte row) that may 
not match what the actual AIE kernels expect.

The likely difference: the AIE kernel might use:
1. A different tile ordering (column-major instead of row-major)
2. A different scale/ZP application formula
3. INT8 weights (not INT4) with different scaling

This explains why:
- The dequant gives plausible-looking weights (94% negative) but wrong GEMM outputs
- The NPU engine works despite model.c producing garbage (AIE kernel has its own reader)
- Our careful GEMM layout matching doesn't fix the issue (wrong base weight values)

## Next Steps

1. **Reverse-engineer the xclbin format** by dumping weight BOs and comparing
   OR
2. **Use model.c's raw-BF16 reading** (despite producing NaN) — the AIE kernel might 
   correct the data internally
   OR
3. **Skip dequantization entirely** — use model.c's approach of reading raw bytes as BF16
   (which is what the xclbin kernel expects) and send these directly to HIP GPU as BF16 weights
   OR
4. **Port engine.cpp's NPU pipeline to GPU** — implement the same weight packing that works 
   on NPU, sending the same data to the GPU
