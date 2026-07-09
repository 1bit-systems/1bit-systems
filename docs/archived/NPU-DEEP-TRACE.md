# NPU Engine Deep Per-Sublayer Trace — July 5, 2026

## Run command

```
./build/npu_engine_cb --trace 1 1   # npt=1 prompt (token 100), ng=1 decode
```

→ dumps layer-0 substage activations to `/tmp/cb_trace/*.bin`,
compare vs HuggingFace `Qwen/Qwen3-0.6B` reference.

## Result

Token 100 ("when"), layer 0 prefilled alone:

| Activation     | Eng norm | HF norm  | cos_sim | ratio | Status |
|---------------|---------:|---------:|--------:|------:|:------:|
| input_embedding | 1.0397 | 1.0396 | 0.999840 | 1.0001 | ✅ |
| h_ln1 (RMSNorm)  | 6.3854 | 6.3868 | 0.999866 | 0.9998 | ✅ |
| **q_flat (QKV Q)**   | **14.6144** | **1.7881** | **0.834606** | **8.1729** | ❌ |
| **k_flat (QKV K)**   | **11.9603** | **1.3817** | **0.790079** | **8.6565** | ❌ |
| v_flat (QKV V)   | 4.5591 | 0.8970 | 0.968563 | 5.0826 | ✅ |
| q_normed        | 111.09 | 92.29 | 0.817743 | 1.2037 | ❌ |
| k_normed        | 565.30 | 309.30 | 0.922436 | 1.8277 | ❌ |

## Findings

1. Embedding + input_layernorm RMSNorm match HuggingFace (cos_sim ≈ 1.0).
   So embeddings lookup, BF16→float conversion, RMSNorm, and the clipped
   norm weights are **correct**.

2. The **first divergence** is at `q_flat` — the GEMM output for Q. The
   kernel output (Cm=2778 → Cm × ascale × Bscale = 0.2584 per element)
   has **norm ratio 8.2×** and **cos_sim 0.83** relative to HF.
   Cos_sim 0.83 is too low for quantization noise — the engine's Q
   vector points ~33° away from HF's Q vector.

3. **K** is similarly divergent (cos_sim 0.79). **V** has cos_sim 0.97 —
   much closer. The Q-vs-V quality gap directly mirrors the **weight
   magnitude distribution**: Q has wider outliers than V (more of Q's
   rows need a different effective scale to be correctly captured than
   V's rows).

4. The **kernel itself is verified correct** (earlier hardware
   dump-and-compare proved Cm == Am @ Bm bit-for-bit). The
   wrong direction (cos_sim < 1) therefore comes from the **scale
   being applied to the kernel's raw output**.

## Root cause (located, not fixed)

The HF cache packer at `tools/layer_trace.py` packs the fused QKV
  weight tensor with **one single per-tensor scale**:

    qkv_scale = max(abs(np.concatenate([Q, K, V]))) / 127.0

That single Bscale is then stored in `wsc[l].qk` and applied inside
`go()`:

    C_f32[j] = Cm[j] × ascale × qkv_scale

This is only correct when the entire QKV weight matrix has a *uniform*
weight distribution. In reality:
  - Subset_max(Q) ≠ Subset_max(K) ≠ Subset_max(V)
  - The HF packer's `qkv_scale` is the *largest* of the three max-abs
    values (it's max over the concatenation), so Q-flat-per-row dynamic
    range gets under-scaled (effectively all Q values get hit by the
    largest scale, while their *own* per-element range is much narrower).

The fix is per-projection packing (or per-column quantization, matching
what the experimental per-column INT8 path attempted before being
abandoned earlier this investigation).

The other agent has continued work on this in parallel — they're
validating per-column INT8 packing with a search inside `n1_core_i8_v2.py`
to see which (K_tile, scale) binding the AIE columns expect.

## Earlier false alarms (retracted)

- ~~dequant_q4nx.c produces 800× too large weights~~ — measurement
  error: C dequant test used data_section offset instead of absolute
  file offset (8+34584 byte header). dequant is correct.
- ~~xclbin kernel internal data tiling mismatch~~ — hardware
  dump-and-compare proved kernel bit-exact versus numpy.

## Files Changed

| File | Change |
|------|--------|
| `engine/npu/src/npu_engine_cb.cpp` | Stride fix pi*NH*HD → pi*4096, decode off-by-one fix, RMSNorm clipping, i32 Cm |
| `engine/npu/xclbins/n1_core_i8_v2.py` | matmul_i8_i16 → matmul_i8_i32, dtype_out int16 → int32 |
| `engine/npu/src/i4_loader.h` | New: raw I4 expander (for the per-column path) |
| `tools/layer_trace.py` | Layer-by-layer trace plus /tmp/hf_weights_cache packer |
| `tools/chunk_dequant.py` | Python Q4NX dequant byte-matching reference |
| `docs/archived/NPU-DEEP-TRACE.md` | This document |
| `docs/archived/NPU-ENGINE-CORRECTNESS-STATUS.md` | Earlier investigation summary |

## Recommended next step

Stop packing Q+K+V with one shared scale. Two viable options:

1. Pack Q, K, V into separate BOs (separate xclbin dispatches, separate
   Bscale per subsystem). Simple, correct. Slower (3 GEMM × layer instead
   of 1 fused).
2. Per-column INT8 packing inside the fused QKV BO. Swaps to **one**
   Bscale per AIE 128-wide output column instead of single tensor scale.
   Preserves the single-dispatch path; was the original plan that the
   prior "dequant 800×" investigation abandoned. Now that the dequant is
   confirmed correct, this can be revisited.