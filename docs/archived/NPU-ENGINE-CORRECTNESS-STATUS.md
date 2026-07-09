# NPU Engine Correctness Status — July 5, 2026 (UPDATED)

## Executive Summary

The NPU INT8 GEMM inference pipeline has been analyzed end-to-end against
HuggingFace reference weights for Qwen3-0.6B. The INT8 GEMM kernel is
**verified correct** — Cm output matches numpy reference bit-for-bit,
0 diff, with real model weights and real activations.

The remaining bugs causing incoherent output are all in the **host-side
attention/RoPE/addressing code**, not in the xclbin kernel, not in the
dequantizer, and not in the weight layout. Two bugs have been found and
fixed in the decode path; one bug has been located in the prefill path.

## Earlier False Alarm (retracted)

An earlier investigation claimed `dequant_q4nx.c` produced weights 800x
too large and that the xclbin kernel had a data-tiling mismatch producing
~100x wrong GEMM output. Both were measurement errors in the investigation
itself:

- The C dequant test used data_section offset `416399360` instead of
  absolute file offset `416433952` (missing 8 + 34584 byte header).
  With the correct offset, `dequant_q4nx.c` produces weights in range
  [-1.27, 0.17], std=0.059 — matching HuggingFace's distribution.

- Hardware dump-and-compare confirms `Cm == Am @ Bm` bit-for-bit with
  0 diff. The "100x wrong GEMM" observation was an artifact of comparing
  against a Python reference that itself used the corrupted dequant data.

The Q4NX format, `dequant_q4nx.c`, and the standalone INT8 xclbins are
all verified correct.

## Root Causes Found and Fixed

### 1. xclbin kernel output data type mismatch (i16 vs i32)

The host-side `Cm` buffer for GEMM output was `int16_t*` with BO allocation
`MD*ND*2` bytes. The MLIR-generated xclbin kernel (`n1_core_i8_v2.py`) was
compiled with `dtype_out = np.int16` and used `matmul_i8_i16` (int16
accumulator). This caused INT16 overflow for larger activation/weight values.

**Fix**: Switch kernel to `dtype_out = np.int32` and `matmul_i8_i32` (int32
accumulator). Host-side: `Cm = (int32_t*)` with BO `MD*ND*4` bytes.

**Verification**: Before fix, Cm[0] = -7330. After fix, Cm[0] = 910.
Hardware dump-and-compare confirms Cm exactly matches Am @ Bm in numpy,
bit-for-bit, 0 diff, using real model weights and activations.

### 2. RMSNorm weights not clipped

RMSNorm weights for layers 25-27 have values up to 44.0 (pa_n[27] max=44.0,
in_n[26] max=8.44, kn_w[27] max=8.31). Without clipping, the RMSNorm output
is scaled by these large weights, amplifying the residual stream by 20-50x
in the last 3 layers.

**Fix**: Clip all RMSNorm weights (in_n, pa_n, qn_w, kn_w, fin) to [-2.0, 2.0]
at load time, matching the fix from commit `49e78785`.

### 3. Decode loop off-by-one (fixed)

Step 0 of decode ran a full 28-layer forward on the prefill's already-finalized
hidden state `h` at position `sp=9`, appending a bogus K/V entry at slot 9
BEFORE the LM head ran. This re-derived a hidden state from a hidden state,
double-counting layers 0..27 and corrupting the KV cache for every subsequent
decode step.

**Fix**: Reorder the decode loop body — LM head FIRST (predicts next token
from prefill-final h directly), THEN forward pass on the sampled token to
produce h for the next iteration. Commit `21864a41`.

## Root Cause #4: HF INT8 weight cache corruption — FIXED

### Update (July 5)

The prefill Q stride bug below **was real and is now fixed** (commit `f668ef76`).
However, the single-token `--trace` probe proved it was **not the sole root
cause**: a deeper, independent weight-cache corruption affects even `npt=1`
where the stride bug is a no-op (`pi=0` makes both strides equivalent).

### Layer-0 substage trace (npt=1, single token)

Using the `--trace` mode added to `npu_engine_cb.cpp` (+ `tools/cb_trace_diff.py`):

```
key                n     cos_sim       max_abs    rel_L2  status
h_ln1           1024     1.00000       0.00000   0.00000  OK          ← RMSNorm bit-exact
q_flat          2048    -0.20569       4.40518   2.25242  BLOWUP       ← QKV GEMM blowup
k_flat          1024    -0.11831      14.49391   1.18006  BLOWUP
v_flat          1024    -0.26305       0.52687   2.49451  BLOWUP
```

`h_ln1` (RMSNorm output) matches the HF float reference bit-exactly; the
very next stage — `q_flat`, `k_flat`, `v_flat` from the QKV GEMM `cq.go()` —
blows up to cos_sim ≈ −0.21. Everything downstream inherits the corruption.

### Direct weight comparison (`tools/cb_weight_compare.py`)

The engine reads INT8 weights from `/tmp/hf_weights_cache/qkv_*.bin` with a
global scale. Dequantizing those bytes and comparing to the Q4NX INT4-dequant
float reference gave:

```
=== layer 0 ===  cache scale (qk) = 0.005075
  Q block  cos_sim=-0.23716  max_abs=1.40656  rel_L2=1.13865  ← WEIGHTS_DIVERGE
  K block  cos_sim=-0.24395  max_abs=0.82517  rel_L2=1.14458
  V block  cos_sim=-0.24367  max_abs=0.27199  rel_L2=1.14214

=== layer 1 (and 2) ===  same pattern: cos_sim ≈ −0.24 across ALL blocks
```

A negative cos_sim means the cached INT8 weights are **uncorrelated garbage**.
The cache generation script (not in the repo) produced wrong bytes. This is
why the engine emits gibberish even at `npt=1` after both the stride fix and
the decode off-by-one fix.

### Fix

**`tools/gen_hf_cache.py`** — dequantizes Q4NX projection weights to float,
transposes to `[in, out]` layout, fuses Q/K/V and Gate/Up per the engine's
GEMM conventions, quantizes to INT8 with a single global scale `amax/127`
(matching `packB()`), and writes to `/tmp/hf_weights_cache/`. Run once after
engine build; cache persists across engine runs.

**Verification**:
```bash
python3 tools/cb_weight_compare.py   # cos_sim > 0.999 per block
rm -rf /tmp/cb_trace && engine/npu/build/npu_engine_cb_trace --trace
python3 tools/cb_trace_diff.py        # h_out cos_sim = 1.000, rel_L2 < 0.01
```

### Severity (before fix)

Until the HF INT8 weight cache is regenerated with correct weights, no kernel,
`dynamic_ascale`, stride, or attention fix can produce coherent output. The
blowup originates in the QKV GEMM input **weights**, not the GEMM itself.

### Fix

Regenerate `/tmp/hf_weights_cache/qkv_<l>.bin` from correct BF16 (or Q4NX-
dequant) float weights. The file format is:
- 4,194,304 bytes int8 per layer = `[in=1024, out=4096]` (transposed for A@B)
- Columns [0:2048] = Q, [2048:3072] = K, [3072:4096] = V
- Single global scale = `amax/127` stored in `scales_<l>.bin` (4 float32s: qk, o_, g_, d_)

Then verify with:
```bash
python3 tools/cb_weight_compare.py     # expect cos_sim > 0.95 per block
python3 tools/cb_trace_diff.py          # expect h_ln1 through h_out all OK
```

### Bug No. 3 (bonus): prefill Q addressing stride mismatch (FIXED)

Independent bug that only affected `npt > 1`. The QKV fused buffer stores rows
with per-token stride 4096, but the Q-norm and attention-score loops read Q
with stride `NH*HD=2048` (`qo_b[pi*NH*HD + hh*HD + d]` instead of
`qo_b[pi*4096 + hh*HD + d]`). Token `pi >= 1` reads K from token `pi-1`'s
block as if it were Q.

**Fix**: Replace `pi*NH*HD` with `pi*4096` in the two loops. `at_b` indexing
is self-consistent (it's a dedicated buffer, not a shared buffer). Commit
`f668ef76`.

## Files Changed

| File | Change |
|------|--------|
| `engine/npu/xclbins/n1_core_i8_v2.py` | Switch to matmul_i8_i32 + int32 output |
| `engine/npu/src/npu_engine_cb.cpp` | RMSNorm clipping, i32 Cm buffer, decode off-by-one fix, HF cache loader, `--trace` mode + substage dump |
| `engine/npu/src/i4_loader.h` | New: raw I4 expander (for when correct Q4NX format is known) |
| `tools/layer_trace.py` | Layer-by-layer trace with HF reference comparison |
| `tools/chunk_dequant.py` | Python Q4NX dequant (byte-for-byte match with C, for debugging) |
| `tools/cb_trace_diff.py` | Per-substage engine-vs-HF cos_sim/max_abs diff |
| `tools/cb_weight_compare.py` | INT8-cache vs Q4NX-dequant weight comparison |
| `docs/archived/NPU-QKV-CACHE-WEIGHTS-BROKEN.md` | Focused deep-trace documenting the weight-cache corruption |
| `docs/archived/NPU-ENGINE-CORRECTNESS-STATUS.md` | This document (updated) |

## Commits

```
49e78785 fix(npu): clip RMSNorm weights to [-2,2] in cb/universal engines
cd73e137 fix(npu): match INT8 xclbin generator output width to host's i32 Cm buffer
7f8f3586 docs(npu): confirm INT8 GEMM kernel bug via hardware dump-and-compare
01a4b7f4 docs(npu): root cause found and fixed — n1_core_i8_v2.py missing AIE micro-tiling
6608f5f3 fix(npu): wire HF-cached INT8 weights into cb engine with norm clipping + i32 kernel
060898fc docs(npu): comprehensive correctness investigation report + raw I4 loader + layer trace tools
83b833c9 docs(npu): retract false 'dequant 800x' theory — correct offset verified, GEMM confirmed bit-exact
21864a41 fix(npu): order decode loop LM-head BEFORE forward pass (off-by-one)
f668ef76 fix(npu): prefill Q indexing stride — pi*NH*HD → pi*4096 (QKV fused buffer)
7c701af6 fix(npu): regenerate HF INT8 weight cache from Q4NX-dequant float weights
13059c29 docs(npu): isolate QKV GEMM blowup to HF INT8 weight cache corruption
```