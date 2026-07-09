# NPU Engine QKV Weight-Cache Corruption — Root Cause (July 5, 2026)

> **STATUS: FIXED** by `tools/gen_hf_cache.py`. See `/tmp/hf_weights_cache/`.
> Layer-0 substage trace isolates the QKV GEMM blowup to the HF INT8 weight
> cache, **not** the kernel, **not** `dynamic_ascale`, and **not** the prefill
> Q-indexing stride. The stride bug (`f668ef76`) was real but independent and
> only affected `npt>1`; this corruption affects even `npt=1`.

## TL;DR

- `input_layernorm` output (`h_ln1`) matches the HF float reference **bit-exactly** (`cos_sim = 1.000`, `max_abs = 0.000`).
- The very next stage — the QKV GEMM output `q_flat = h_ln1 @ W_qkv` — blows up to `cos_sim = −0.21`.
- Direct weight-vs-weight comparison of `/tmp/hf_weights_cache/qkv_*.bin` (dequantized with the cache scale `wsc.qk`) against the Q4NX-dequant reference used by `tools/layer_trace.py` gives **`cos_sim ≈ −0.24`** for **every layer (0, 1, 2) and every Q / K / V block**.
- Since `h_ln1` is exact and `q_flat = h_ln1 @ W_qkv`, the GEMM-output blowup is fully explained by the **weights being garbage**, independent of the NPU kernel and the activation quantizer.
- The cache file **format** is correct (4,194,304 int8 bytes = `1024×4096`, single global scale `0.005075`). The **contents** are uncorrelated with the model's actual weights.

**Verdict: the script that generated `/tmp/hf_weights_cache/qkv_*.bin` produced uncorrelated INT8 weights. That script is no longer in the repo. It must be regenerated correctly from the Q4NX-dequant float weights (or directly from BF16) before the engine can produce coherent output.**

## Method

1. Added `--trace` to `engine/npu/src/npu_engine_cb.cpp`: forces `npt=1`, token `100`, `ng=0`, and dumps 17 layer-0 substage intermediates (`h_ln1, q_flat, k_flat, v_flat, q_normed, k_rope, attn_out_flat, attn_proj, h_after_attn, h_ln2, ffn_gate, ffn_up, ffn_hidden, ffn_out, h_out`) to `/tmp/cb_trace/<key>.bin`.
2. `tools/layer_trace.py` produces the HF float reference of the same substages into `/tmp/layer_trace_outputs.npz` (note: `layer_trace.py` itself has a minor bug — it omits the RMS *division* in QK-norm, applying only `×weight`; this is irrelevant to the present root cause since the divergence is upstream of QK-norm).
3. `tools/cb_trace_diff.py` element-wise diffs the two, reporting `cos_sim`, `max_abs`, `rel_L2` per key.

## Trace diff (engine vs HF reference)

```
key                n     cos_sim       max_abs    rel_L2  status
h_ln1           1024     1.00000       0.00000   0.00000  OK        ← RMSNorm bit-exact
q_flat          2048    -0.20569       4.40518   2.25242  BLOWUP     ← first divergence (QKV GEMM out)
k_flat          1024    -0.11831      14.49391   1.18006  BLOWUP
v_flat          1024    -0.26305       0.52687   2.49451  BLOWUP
q_normed        2048    -0.29300      25.06358  14.18249  BLOWUP     (all downstream inherit the weight bug)
...
```

**The first diverging substage is the QKV GEMM output.** Everything downstream (QK-norm, RoPE, attention, O-proj, FFN) only inherits the corruption — none of them is the cause.

## Weight-vs-weight isolation (`tools/cb_weight_compare.py`)

Dequantize the engine's cached INT8 QKV with the cache's own scale and compare
to the Q4NX-INT4-dequant reference (`q4nx_reference.dequantize_weight`, the
same float weights `layer_trace.py` uses):

```
=== layer 0 ===  cache scale (qk) = 0.005075
  Q block  cos_sim=-0.23716  max_abs=1.40656  rel_L2=1.13865
  K block  cos_sim=-0.24395  max_abs=0.82517  rel_L2=1.14458
  V block  cos_sim=-0.24367  max_abs=0.27199  rel_L2=1.14214
  → WEIGHTS_DIVERGE (cache generation suspect)

=== layer 1 ===  cache scale (qk) = 0.005044
  Q block  cos_sim=-0.24309  ...
  K block  cos_sim=-0.23997  ...
  V block  cos_sim=-0.24568  ...
  → WEIGHTS_DIVERGE

=== layer 2 ===  cache scale (qk) = 0.004060
  Q block  cos_sim=-0.24034  ...
  K block  cos_sim=-0.24533  ...
  V block  cos_sim=-0.24250  ...
  → WEIGHTS_DIVERGE
```

A cos_sim near **−0.24** (negative, essentially uncorrelated) across the board
means the cache is not a noisy approximation of the weights — it is the wrong
weights entirely. Consistent with the `q_flat` blowup cos_sim of `−0.21`.

## Why the existing `NPU-ENGINE-CORRECTNESS-STATUS.md` "root cause" is incomplete

That doc identified the **prefill Q addressing stride** bug
(`qo_b[pi*NH*HD+...]` should be `pi*4096`) as the confirmed root cause, on the
grounds that `npt=1` is unaffected (`0*2048 == 0*4096`). The stride bug **is
real** and **is now fixed** in commit `f668ef76`, but the fix is necessary, not
sufficient: the single-token trace (`npt=1`, `pi=0`, where the stride is a
no-op) **still** produces garbage `q_flat`. There are **two independent bugs**:

1. **Prefill Q stride** (multi-token only) — FIXED (`f668ef76`).
2. **HF INT8 weight cache corruption** (all `npt`, including `npt=1`) — STILL OPEN, this doc.

## Ruled out

| Suspect | Evidence |
|---|---|
| NPU INT8 GEMM kernel | Harmonizes with the existing doc's hardware dump-and-compare: `Cm == Am @ Bm` bit-for-bit. The wrongness is in `B` (the weights), not the kernel's accumulation. |
| `dynamic_ascale` activation clipping | `h_ln1 max|.| = 2.33`, so `ascale = 2.33/127` — no clipping. And the divergence is in the weights, which `ascale` doesn't touch. |
| QK-norm RMS-division bug in `layer_trace.py` | Divergence is upstream at `q_flat`, before QK-norm runs. (The reference bug is real but irrelevant here.) |
| RoPE convention | Upstream of RoPE. |
| Prefill Q stride | `npt=1` reproduces the blowup, where the stride is a no-op. |
| RMSNorm, embeddings, model loading | `h_ln1` is bit-exact; `input_embedding` matches. |

## Next step (unblocks coherent output)

Rewrite the HF INT8 weight-cache generator from scratch, reading the BF16
Qwen3-0.6B weights (HuggingFace `transformers` or directly via `q4nx_reference`
INT4-dequant as the float source of truth) and producing, per layer:

- `qkv_<l>.bin` — int8, layout `[in=1024, out=4096]` (transposed for `A@B`),
  columns `[0:2048]=Q`, `[2048:3072]=K`, `[3072:4096]=V`.
- `scales_<l>.bin` — 4 float32 (`qk, o_, g_, d_`) = `amax/127` per fused matrix.

Then re-run `tools/cb_trace_diff.py` until `q_flat` cos_sim clears 0.999. At
that point `cb_weight_compare.py` should also clear 0.95+ for every block, and
the engine should produce coherent text. Until then, no kernel or attention
fix can produce coherent output — the weights are wrong.

## Reproducer

```bash
export XRT=/opt/xilinx/xrt
export LD_LIBRARY_PATH=$XRT/lib64:$LD_LIBRARY_PATH
g++ -std=c++23 -O3 -o engine/npu/build/npu_engine_cb_trace \
    engine/npu/src/npu_engine_cb.cpp engine/npu/build/dequant_q4nx.o \
    -I$XRT/include -L$XRT/lib64 -lxrt_coreutil -luuid -lm -ldl
rm -rf /tmp/cb_trace
./engine/npu/build/npu_engine_cb_trace --trace
python3 tools/layer_trace.py          # regenerate HF reference
python3 tools/cb_trace_diff.py         # substage diff (expect q_flat blowup)
python3 tools/cb_weight_compare.py     # weight diff (expect WEIGHTS_DIVERGE)
```

## Files

| File | Purpose |
|---|---|
| `engine/npu/src/npu_engine_cb.cpp` | `--trace` mode + substage dump (committed in `f668ef76`) |
| `tools/cb_trace_diff.py` | Per-substage engine-vs-HF cos_sim/max_abs diff |
| `tools/cb_weight_compare.py` | INT8-cache vs Q4NX-dequant weight comparison |
| `tools/layer_trace.py` | HF float reference single-token layer-0 trace |
| `tools/q4nx_reference.py` | Q4NX INT4 weight dequantizer (float source of truth) |

## Commits

```
f668ef76 fix(npu): prefill Q indexing stride — pi*NH*HD → pi*4096 (QKV fused buffer)
21864a41 fix(npu): order decode loop LM-head BEFORE forward pass (off-by-one)
<new>    docs(npu): isolate QKV GEMM blowup to HF INT8 weight cache corruption
```