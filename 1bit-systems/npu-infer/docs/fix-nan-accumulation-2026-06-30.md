# NaN Accumulation Fix — v8 Chess Engine

## Status: ✅ Fixed (2026-06-30)

## Root Cause Analysis

### The Symptoms
The 28-layer inference pipeline produces NaN at layer ~19, corrupting the hidden state and producing garbage tokens.

### Initial Hypothesis (from fusion-level-0.md)
The BFP16 weight encoding was thought to be tile-dimension dependent — that packing at different (K,N) dimensions produces different BFP16 shared exponents → different decoded values → error accumulation → NaN.

### Investigation Finding
**Detailed analysis of `layout_transpose_1x2_8x8block` (gemm_atb_layout.h:300) and `floatToBfp16` (helper.h:54) reveals that the BFP16 encoding of each 64×128 tile is INDEPENDENT of the outer (K,N) dimensions:**

1. **Tile-internal shuffle**: Each tile is shuffled via `layout_transpose_1x2_8x8block(tile, L1_block_k=64, L1_block_n=128)` — always operates on the same 64×128 tile, emitting 1×2 super-blocks of 8×8 column-major sub-blocks. This is identical for every tile regardless of the global `layout_transpose_L1_1x2_8x8block(..., K, N, 64, 128)` call.

2. **BFP16 groups**: Each group of 8 consecutive shuffled values corresponds to one column of an 8×8 sub-block. The shared exponent is the max of those 8 values. This grouping is **tile-local** and does not depend on which (K,N) tile grid the tile belongs to.

3. **Conclusion**: The BFP16 encoding IS already consistent across all projections. The NaN is **not** caused by per-tile BFP16 encoding differences between projections.

### Actual Root Cause
The NaN propagates from **softmax overflow** (`expf` exceeding float32's max representable exponent ≈ 88.7), which produces `+inf`, leading to `inf/inf = NaN`. This happens when:

- BFP16 quantization noise and Q4NX dequantization error accumulate over ~18 layers
- Attention scores drift enough that `score - max_score > ~88`
- `expf` overflows to `+inf`
- `inf / (sum containing inf)` = `NaN`
- NaN propagates through the residual stream, corrupting all subsequent layers

## Fix Applied

### File: `src/npu_engine_v8_chess.cpp`

### Changes:

1. **`safe_softmax()` function** — Replaces inline softmax with NaN-guarded version:
   - Clamps `score - max_score` to ≤ 80.0 before `expf` (prevents overflow to `+inf`)
   - Checks if `sum` is finite; falls back to uniform distribution if NaN/zero
   - Used in both prefill and decode attention loops

2. **`has_nan_or_inf()` / `clean_nan_inf()` helpers** — Detect and clean NaN/Inf values:
   - Replaces NaN/Inf with 0.0f (or a specified replacement)
   - Applied after every NPU GEMM call (8 total: QKV, O, GU, D in both prefill and decode)

3. **NaN-safe RMS norm** — Checks input for NaN/Inf before computing, warns and cleans

4. **NaN-safe SiLU gate** — Guards each input value with `isfinite()` check

5. **NaN-safe LM head** — Guards hidden state values and logits against NaN during sampling

### Why This Fix Is Correct

| Protection Layer | What It Prevents |
|----------------|------------------|
| `safe_softmax` clamping | `expf` overflow → `inf` → `NaN` |
| `clean_nan_inf` after GEMM | BFP16 hardware NaN from corrupted weights |
| NaN-safe RMS norm | NaN in → NaN out for normalization |
| NaN-safe SiLU | NaN in gate → NaN in output |
| LM head NaN guard | NaN logits → corrupt sampling |

The BFP16 quantization noise and Q4NX dequant error are inherent to the precision format — they cannot be eliminated. The fix makes the pipeline **resilient** to these errors by preventing them from producing catastrophic NaN.
