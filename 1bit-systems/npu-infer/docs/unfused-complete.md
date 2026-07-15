## Unfused Engine Complete

### Results
- **Clean, single-file engine** at `/home/bcloud/npu-sandbox/npu-infer/src/npu_engine_clean.cpp`
- **378 ms/token** (same as fused ~340ms — within variance)
- **No NaN, no crashes, 28 layers stable**
- **196 NPU calls per token** (7 GEMMs × 28 layers)
- **~1.9 ms per NPU call**

### Key Fix Applied
The main bug was a **buffer overflow**: `gemm()` tried to write `out_N` columns into caller buffers that were sized for `XN=1024`. Fixed by:
1. Storing `out_*` = min(dequant_rows, XN) = actual packed columns
2. Writing only `min(out_N, XN)` values in gemm
3. Zeroing remaining columns

### Known Limitation
Q (2048 outputs) and G/U/D (3072 outputs) are **truncated to 1024 columns** by the 1024×1024 xclbin. The missing columns are zero — this halves the attention heads and silences intermediate MLP activations. That's why token 55120 repeats.

### Next Steps (for actual useful inference)
1. **Build M=128 custom xclbins** for 1024×2048, 1024×3072, 2048×1024, 3072×1024 dimensions (all use M=128 to keep BFP16 encoding compatible)
2. **Implement N-tiling in gemm**: split Q into 2 tiles, G/U/D into 3 tiles, use one NPU call per tile, accumulate results
3. **Add temperature/top-k sampling** to escape token 55120 limit cycle
4. **Test with real prompts**
