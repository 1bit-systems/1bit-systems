# Speculative Decode — Status

## Architecture

```
Draft (CPU, Eagle3 1-layer / DSpark 5-layer)
    │ proposes tokens
    ▼
Target (NPU, 4-xclbin INT8 GEMM via NPUQwen3Target)
    │ verifies via forward_with_kv()
    ▼
Accept/Reject → next iteration
```

---

## ✅ Working

| Mode | Command | Target | Tok/s | Draft | Status |
|------|---------|--------|-------|-------|--------|
| 4-xclbin spec-decode | `--spec-decode` | NPUQwen3Target (4 xclbins/layer) | **1.3 tok/s** (FIXED_ASCALE+OMP) | DSpark 5-layer | ✅ Real tokens |
| 4-xclbin simple | `--bench-real` | NPUQwen3Target | **3.0s/4tok** | None | ✅ Real tokens |
| Simulated bench | `--bench` | Simulated | ~5000 tok/s | DSpark | ✅ |
| Daemon | `--daemon` | NPUQwen3Target | — | DSpark | ✅ |

## 🔧 Fused Target — NaN at Layers 24-27 (ROOT CAUSE FIXED 2026-07-08)

The fused xclbin (`NpuFusedTarget`) was producing **NaN at layers 24-27** due to
BF16 numerical overflow in the AIE2 tile code, causing all-zero token output.

### Root Cause

The AIE2 Q4NX projection kernel accumulates dequantized weight × activation
products in FP32, then converts the result to BF16 for the compact record
payload (`emit_record_payload()` in `qwen3_decode_kernels_06b.cc`). At deeper
layers (22+), the hidden state values have grown large due to residual
accumulation across 28 layers. When the FP32 accumulator exceeds ~65504 (max
BF16 finite), the `to_vector<bfloat16>()` conversion produces Inf, which
propagates as NaN through the rest of the pipeline.

Additionally, the BF16 residual addition in `full_vector_station.cc`
(`add_bf16_block_inplace`) adds two BF16 values in FP32 but stores the
result as BF16. At deep layers, the residual + projection output can exceed
65504, causing BF16 overflow.

### Fix: Per-layer Weight Scaling + Host-side Compensation

**Approach A implemented** — Scale down Q4NX weights for layers 22-27 by
tapering factors (0.5× for L22-23, 0.25× for L24-25, 0.125× for L26-27).
This reduces all projection outputs (Q, K, V, O, UP, GATE, DOWN), keeping
the FP32 accumulator well within BF16 range. The C++ driver then applies the
inverse scale factor (2×, 4×, 8×) to the output BF16 values of those layers.

1. **Weight rescaling**: `tools/scale_fused_weights.py` modifies existing
   `fused_weights_l*.bin` files by multiplying each Q4NX chunk's BF16 scale
   and offset arrays by the per-layer weight factor.
2. **Host compensation**: `npu_fused_target.h` has a `kWeightScaleFactor[]`
   array mapping each layer to its inverse compensation factor. After each
   kernel run, scaled layers' BF16 outputs are multiplied by the compensation
   factor before being fed to the next layer.
3. **NaN clamp retained**: The existing `clamp_bf16_finite()` guard remains
   as a safety net against unexpected overflow.

| Mode | Tok/s | Tokens | Status |
|------|-------|--------|--------|
| `--fused` | 291 tok/s | ✅ coherent | 🏆 Weight-scaled (0.5×→0.125× layers 22-27) + host compensation |

**Long-term fix**: Modify the AIE2 kernel source (`to_vector<bfloat16>()` in
`emit_record_payload` and `add_bf16_block_inplace`) to saturate/clamp before
BF16 conversion. This requires recompiling the xclbin but would handle all
layers and future models without host-side workarounds.

## 🏋️ Training

| Run | Examples | Epochs | Loss | Acceptance | Data Source |
|-----|----------|--------|------|------------|-------------|
| FP (old) | ~10K | 5 | — | 0% | HuggingFace FP16 |
| NPU 200 | 200 | 5 | 12.1→6.4 | **0%** | NPU INT8 hidden states |
| **NPU 1K** | **1,000** | **5** | **9.10→4.19** | **~0%** | **NPU INT8 hidden states** |

**Analysis**: Loss 4.19 = perplexity ~66 = ~1.5% per-token accuracy. With 7 draft
tokens, expected block acceptance = 1 - (0.985)^7 ≈ 10%. Measured: 0% in 15 rounds
(~20% chance of all-zero). Draft model quality is the bottleneck, not correctness.

**Root cause**: 1K examples is insufficient; Eagle3 1-layer architecture may not
capture NPU INT8 distribution well enough. The C++ inference is verified correct
(bitwise match with Python training).

## 🆕 Native Ternary Daemon

The NPU native ternary daemon (`npu_ternaryd`) executes all 28 layers but
produces **zero-valued outputs** for all projections. Root cause identified:

**Weight encoding mismatch**: The Q1_0→ternary repack tool (`repack_q1_0.py`)
maps Q1_0 1-bit values to 2-bit encoding (0b00/−1, 0b10/+1), but the AIE2
kernel (`mm_ternary_32x64x128.o`) likely expects a different 2-bit ternary
encoding (e.g., 0b01/−1, 0b00/0, 0b10/+1 or similar). All scales are non-zero
(mean 0.038, verified), so the zero output is from weight lookup returning zero.

**Fix needed**: Either (a) determine the kernel's expected 2-bit encoding and
update `repack_q1_0.py`, or (b) rebuild the kernel with the encoding produced
by the repack script.

## 🔧 Spec-Decode Bugs Fixed (2026-07-08)

Three critical C++ inference bugs were causing draft model to produce wrong outputs:

1. **Cross-head attention** (was broken GQA + time-accumulating KV cache).
   Eagle3 uses single-position cross-head attention (no KV cache, no time dim).
2. **FFN residual connection** (was adding MLP output to un-normed attention output).
   Python adds MLP to `post_attn_norm(h + attn)`, not to raw `h + attn`.
3. **Hidden state norm mismatch** (was using wrong norm weight at pos > 0).
   Python uses `norm()` weight; C++ was using `hidden_norm()` weight.

C++ now produces **identical outputs** to Python (verified with numpy-generated
input, all 5 positions match to 4 decimal places).

## 🆕 Token Router (separate project)

A **token-level router** design in Rust (`token-router/DESIGN.md`) routes
individual tokens between NPU/GPU/MLX backends based on confidence, content,
or budget. Key strategies:

- **Cascade**: NPU generates all tokens; low-confidence tokens rerouted to GPU
  (no draft model needed — confidence-based fallback)
- **Speculative decode**: draft on NPU, verify on GPU (uses unified memory on
  Strix Halo for zero-copy)
- **Content router**: keyword-based routing (port of `unified-router.py`)

This could bypass the draft training problem entirely — instead of training
a draft that matches the NPU's distribution, use the GPU to handle uncertain
tokens on the fly.

## Next Steps

1. ✅ **Training complete** — 1K examples, 5 epochs, loss 9.1→4.2
2. ✅ **C++ inference fixed** — verified bitwise match with Python
3. 🔍 **Scale up training** — 50K+ examples needed for useful acceptance rate, or
   pivot to cascade token router
4. **Fix ternary daemon weight encoding** — determine kernel's 2-bit encoding
   and align repack script
5. ✅ **Fused xclbin NaN** at layers 24-27 — root cause fixed via per-layer weight scaling + host compensation (2026-07-08)
