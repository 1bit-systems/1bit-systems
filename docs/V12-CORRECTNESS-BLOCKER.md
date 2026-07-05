# v12 Correctness — Audited July 5, 2026

## Discovery (July 3, 2026)

The open-source C++ engine (v12) was benchmarked at 97 tok/s across all project docs,
but **never validated for output coherence**. Sanity-checking the actual output against
FLM for the same prompt revealed complete garbage — the speed numbers were real, the
output behind them never was.

## Bugs Found and Fixed (7 Rounds)

| Round | Bug | Fix |
|-------|-----|-----|
| 1 | Activation quant clipped outside [-5,5] | `dynamic_ascale()` — per-call amax |
| 2 | Weight matrices packed wrong orientation | `transpose_pack()` + correct `in_features` |
| 3 | LM head used embed table instead of untied weights | `lm_head_f32` separate from `emb_f32` |
| 4 | RoPE used wrong convention | half-split `rotate_half` |
| 5 | Causal mask off-by-one in prefill | V-loops use `sp+pi+1` |
| 6 | Spec decode KV cache not written for draft tokens | Fix in 3 spec engines |
| 7 | Hardcoded paths | `NPU_XCLBIN_DIR`/`NPU_MODEL_PATH` env vars |
| 8 | NaN on LM head softmax | NaN guards added |

## Current Status (July 5, 2026)

### ✅ Host-side math: CORRECT
The universal engine (`npu_engine_universal.cpp`) compiles and runs. The host-side math
(dequant → transpose → INT8 quant → dispatch) has been verified against the fused
xclbin reference path.

### ✅ Fused xclbin: VALIDATED
The torch2aie fused xclbin (single-layer dispatch) produces output matching the CPU
oracle within `max_abs=0.0078`. This confirms:
- Q4NX dequantization is correct
- NPU INT8 GEMM kernels produce correct output
- Weight packing for the fused format is correct

### ⚠️ Standalone INT8 xclbins: NOT RE-VERIFIED
The 4 standalone INT8 xclbins (QKV, O, GU, D) used by `npu_engine_universal.cpp`
are opaque MLIR-compiled binaries. After all host-side fixes, they have NOT been
re-verified for coherent end-to-end output. The `npu_engine_fused_i8.cpp` engine
(using the fused xclbin) IS verified — it runs at 4 tok/s with valid output.

### Next Steps
- Re-validate the universal engine output after all host-side fixes
- Or switch production to the fused xclbin path (which requires the torch2aie toolchain)
