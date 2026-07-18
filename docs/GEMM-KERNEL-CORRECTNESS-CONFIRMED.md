# INT8 GEMM Kernel Correctness — Confirmed Hardware Bug (July 5, 2026)

> **Status as of 2026-07-18**: superseded in part — see the bottom-most
> "Status" section for the current, most accurate picture. Short version: the
> single-core-vs-multi-core bug this doc originally described has fixes for
> all four shapes now, but they were developed on two separate branches and
> reconciled after the fact; the *combined* result (all 4 shapes at 8-core
> together) has not been independently re-verified on real hardware. Read the
> final "Status" section before trusting any throughput/correctness claim in
> the body of this doc.

## TL;DR

Started chasing a "repeated token" decode symptom in `npu_engine_cb.cpp`/
`npu_engine_universal.cpp`. Found and fixed three real, independent host-side bugs
along the way (int16/int32 kernel-output width mismatch, a malformed smoke-test
prompt, unbounded RMSNorm weights). None of them were the root cause.

**With all host-side math now provably correct** (verified via direct hardware
dump-and-compare against the exact bytes sent to the NPU), the compiled INT8 GEMM
xclbin kernels themselves — all four of them (QKV, O, GU, D) — **do not compute
`A·B` on their own real inputs.** This is not approximation error or quantization
noise; the hardware's output has no measurable correlation with the correct answer.

This confirms, with hard numbers, what `docs/V12-CORRECTNESS-BLOCKER.md` suspected
back on July 3 but never proved: *"If the host-side math is correct... and output is
still incoherent, the bug must be in the xclbin kernels themselves."* It's not
fixable from the host side. The next step is inside the MLIR generator
(`engine/npu/xclbins/n1_core_i8_v2.py`) / kernel source (`mm.cc`) / `aiecc`
compilation pipeline, or requires the AI Engine Simulator (previously blocked — see
`docs/FUSED-INTEGRATION-BLOCKER.md` — by a missing real NPU2 device-topology file;
a substitute Versal device file now exists but models the wrong chip, so simulation
results from it would not be trustworthy without independent validation).

## Investigation Timeline

### 1. Starting symptom: decode repeats the same token every step

`npu_engine_cb.cpp` and `npu_engine_universal.cpp`'s hardcoded smoke-test prompt
(`int pt[]={151643,872,198,11852,151644,198,151643,77091,198}`) decodes via the real
Qwen3 tokenizer to garbage: `<|endoftext|>user\nems<|im_start|>\n<|endoftext|>assistant\n`
— `<|endoftext|>` and `<|im_start|>` are swapped, and the "user message" is the
stray fragment "ems". **Fixed**: replaced with the real tokenization of
`<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n`
(`{151644,872,198,13048,151645,198,151644,77091,198}`), verified against the actual
Qwen3-0.6B tokenizer. Landed in both files, on `1bit-systems` PR #32.

**This did not fix the repeated-token symptom.** Re-ran with the corrected prompt;
decode still collapsed to a single token with the exact same near-100%-confidence
logit distribution as with the garbage prompt.

### 2. Fixed: INT16/INT32 kernel-output width mismatch

Separately (see PR #32's first commit): the MLIR generator
(`n1_core_i8_v2.py`) declared `dtype_out=np.int16` and linked `matmul_i8_i16`/
`zero_i16`, so every xclbin it built emitted 16-bit accumulator output. But
`npu_engine_cb.cpp`'s host code reads that buffer as `int32_t` (`bC` sized
`MD*ND*4`, comment `// kernel outputs i32`). Reading a 16-bit kernel result as
32-bit produced a ~120,000x magnitude error (`Cm[0]` was `-782,000,000` vs. an
expected `~-6540`).

**Fix**: switched the generator to `dtype_out=np.int32`, linked
`matmul_i8_i32`/`zero_i32` (which `mm.cc` already defines for `-Di8_i32_ONLY`
builds — no kernel source changes needed). Rebuilt all four production xclbins.
Verified: `Cm[0]` went from `-782,000,000` to `-6770`, matching a Python reference
(`-6540`) to within normal quantization rounding. This was a real, now-resolved bug
— but it was masking the deeper issue documented below, not causing it.

### 3. Fixed (partial): unbounded RMSNorm weights

Dumped the post-prefill hidden state for three different 9-token prompts (a
tokenizer-valid "Hi" prompt, a "No" prompt, and the original garbage prompt).
All three converged to a hidden-state norm of **~4,050,000**, with cosine
similarity **~1.0** between them — i.e. the residual stream was almost entirely
input-independent by the end of the 28-layer prefill stack, swamping any real
per-token signal (expected O(1-100)).

Root cause: `input_layernorm`/`post_attention_layernorm`/final-norm weights are
loaded directly from the bf16 model file with no bound. Measured directly:
`in_n[26]` max = **106.5**, `pa_n[27]` max = **192** — matching
`docs/journey.md`'s prior note ("Norm weights grow to 106x by layer 26"). Since
`rn_c()` (RMSNorm) rescales the input by this weight regardless of the input's own
scale, and the result is added into the residual stream every layer, weights this
large make each layer's contribution almost entirely weight-driven rather than
input-driven — compounding over 28 layers into near-input-independent magnitude.

A fix for this exact issue ("clip input_norm/post_norm to max 2.0") was previously
applied only to the (separate, unmerged) fused-xclbin path per PR #23 / journey.md,
never ported to `npu_engine_cb.cpp`/`npu_engine_universal.cpp`. **Fix**: ported the
clip (`[-2, 2]`) to `in_n`/`pa_n`/`fin`(`_v`) at load time in both files. Landed on
PR #32.

**Verified improvement, not resolution**: decode stopped locking onto one single
token every step (started oscillating among the top few candidates instead). But
re-checking hidden states post-clip across different prompts still showed cosine
similarity ~1.0 — a second, unidentified contributor remained.

### 4. Isolated the remaining explosion to the down-projection (D) GEMM

Instrumented `npu_engine_cb.cpp`'s batched prefill loop to print the hidden-state
norm/maxabs after every stage of layer 0 (for the corrected "Hi" prompt):

| Stage | norm | maxabs |
|---|---|---|
| post-RMSNorm, pre-QKV | 6.53 | 1.85 |
| QKV output | 14.67 | 1.29 |
| attention output | 30.36 | 4.41 |
| O-proj output | 24.56 | 2.46 |
| post-RMSNorm 2, pre-GU | 15.60 | 3.59 |
| GU (gate/up) output | 164.41 | 4.64 |
| SiLU-gated (su_b) | 270.47 | 18.65 |
| **D (down-proj) output** | **2250.62** | **223.35** |
| after residual add | 2264.18 | 223.99 |

Everything up through the SiLU-gated activation is sane. The down-projection GEMM
alone amplifies norm by ~8x and maxabs by ~12x in a single matmul — and this
compounds every layer, producing the ~4,000,000 norm seen after 28 layers.

### 5. Found and fixed a real bug in the (uncommitted) per-column quantization scheme — did not resolve the explosion

At this point in the investigation, a separate/parallel session had rewritten
`npu_engine_cb.cpp`'s weight quantization from a single global per-tensor scale to
a per-AIE-column scheme (`packB_col`/`go_col`, uncommitted, live in
`/home/bcloud/engine/npu/src/npu_engine_cb.cpp` at time of writing — not yet on any
branch/PR). Found a real, demonstrable bug in it: `ncol` (fixed at `init()` time)
didn't match the actual weight buffer size for the O and D contexts.

- **D-proj**: `col_sz = K*128` with `K=DOUT=1024` → `col_sz=131,072`. `ncol=8`
  (hardcoded). Total packed = `1,048,576`. Real buffer size = `DOUT*DIN =
  1024*3072 = 3,145,728`. **Only 1/3 of the weight buffer was being packed** — the
  rest was whatever was already in that device buffer.
- **O-proj**: same pattern, packing exactly half the buffer.
- QKV and GU happened to come out exactly right by coincidence of their specific
  dimensions.

Tried the obvious fix (swap the `packB_col` call's argument order for O/D so
`ncol*col_sz` exactly equals the true buffer size — verified via arithmetic). **This
made D's explosion worse** (norm 2250 → 7195), not better. That regression revealed
the deeper problem: `packB_col`'s per-chunk scale groups by contiguous
memory offset, which corresponds to **input-feature row groups**, but
`go_col`'s dequant looks up `Bcol[l][n/128]` indexed by **output column** — these
are different axes of the same buffer and never corresponded to the same thing,
regardless of argument order.

**Fix applied**: reverted `packB_col`/`go_col` to a single global per-layer weight
scale (provably correct — verified the full buffer is scanned/packed, no
under-coverage). This is a real, verified correctness fix over the buggy per-column
scheme. It did not resolve the explosion either: D-proj output was still
**norm=7259, maxabs=434** for the identical layer-0 input that should produce
`su_b`'s norm=270 input scaled by a sane weight matrix.

This fix currently lives uncommitted in the live checkout
(`/home/bcloud/engine/npu/src/npu_engine_cb.cpp`), mixed in with the rest of that
session's broader (unrelated, in-progress) rewrite across ~25 NPU engine files, so
it hasn't been cleanly isolated into its own commit yet.

### 6. Ruled out a host-side data bug: raw down-projection weights are clean

Dumped the raw dequantized `down_proj` weight for layer 0 *before* any quantization:
**norm=130.7, maxabs=0.6855**, dims `1024×3072` (correct). This is a completely
normal weight magnitude for a trained model. The explosion is not caused by bad
weight data or a dequant/tiling bug upstream of quantization.

### 7. Confirmed via hardware-in-the-loop testing: the compiled kernels don't implement `A·B`

Two independent tests, both against real NPU2 hardware:

**Test A — synthetic isolated GEMM** (adapted `npu-sandbox/npu-infer/tools/test_gemm_i8.cpp`
for int32 output, run directly against each production xclbin with a known,
deterministic INT8 pattern for A and B):

| Shape | M,K,N | Errors (of 10,000 checked) | Max diff | Avg diff |
|---|---|---|---|---|
| QKV | 128,1024,4096 | 10,000 | 34,263 | 11,063 |
| O | 128,2048,1024 | 10,000 | 60,588 | 25,409 |
| GU | 128,1024,6144 | 10,000 | 27,492 | 9,052 |
| D | 128,3072,1024 | 10,000 | 58,044 | 28,662 |

All four fail completely. (This test uses a naive row-major `A[M,K]@B[K,N]` fill
pattern that may not exactly match what the kernel expects at the byte level, so on
its own this result is suggestive but not conclusive — see Test B.)

**Test B — real bytes, direct comparison** (instrumented `go_col` to dump the exact
`Am` (quantized activation) and `Bm` (quantized weight, from `layerB[l]`) buffers
the real model sends to hardware during an actual forward pass, plus the real
`Cm` readback, for both QKV and D at layer 0):

For **D** (`MD=128,KD=3072,ND=1024`, 9 real prefill tokens):
```
Cm[8,0:8]  = [ -94565  -83862  -81273 -111747  -84766  -88403  -87565  -84303]
ref[8,0:8] = [-1252751 -681834 -1339176 -1363487 -732522 -1117851 -1361552 -1521576]
mean|diff| over real rows 0..8: 611,833   max|diff|: 2,666,044
```

For **QKV** (`MD=128,KD=1024,ND=4096`):
```
Cm[0,0:8]  = [ 4676 -7139 10518  1716 -1922 -2209  7988   801]
ref[0,0:8] = [ -652 -1014   337  2501   767   106   350   423]
mean|diff| over real rows 0..8: 5,394     mean|ref| magnitude: 1,834
```

`ref` here is `Am.astype(int64) @ Bm.astype(int64)` computed in numpy on the *exact*
bytes read back from the device buffers before/after the real kernel call — no
guessing about layout conventions, since these are the identical bytes the host
code already prepared and sent.

For QKV, the mean error is *larger than the mean reference magnitude itself* — this
is uncorrelated noise, not imprecise-but-close computation.

**Ruled out simpler explanations for Test B**:
- **Transpose convention** (`A·Bᵀ` instead of `A·B`): tried re-interpreting `Bm` as
  `[N,K]` and computing `A @ B.T` — mean diff 5,160 vs. 5,393, no meaningful
  improvement.
- **Row-tile permutation** (kernel computes correctly but writes output rows out of
  order due to an M-tiling/scheduling bug): checked correlation of every hardware
  output row against every reference row (0-19); best correlation found across all
  pairings was **0.0155** — i.e., none. Value ranges also differ between `Cm` and
  `ref` (roughly ±30,000 vs ±12,000 for QKV), ruling out a pure permutation.

## What This Means

The compiled xclbin kernels for all four INT8 GEMM shapes used by
`npu_engine_cb.cpp` (QKV, O, GU, D — built via
`npu-sandbox/npu-infer/bf16_kernel_dev/build_all_int8_i32.sh`, using
`mm.cc`'s `matmul_i8_i32` and `engine/npu/xclbins/n1_core_i8_v2.py`'s generated
MLIR) do not correctly compute `A[M,K] @ B[K,N]` for real inputs on real NPU2
hardware. This is upstream of, and independent of, every host-side bug found and
fixed in this repo so far (dequant tiling, weight transpose, activation clipping,
RoPE convention, causal masking, residual saves, LM head substitution, the
int16/int32 width mismatch, unbounded norm weights). All of those fixes were
real and necessary, but none of them touch the actual arithmetic the compiled
kernel performs on the AIE cores.

This directly confirms (with reproducible, quantitative hardware evidence) the
unresolved question `docs/V12-CORRECTNESS-BLOCKER.md` ended on:

> If the host-side math is correct (as we believe it now is), and output is still
> incoherent, the bug must be in the xclbin kernels themselves — either the INT8
> GEMM matrix multiply or the quantization/dequant logic inside the NPU compute
> tiles.

## What's NOT Yet Known

- Whether the bug is in `mm.cc`'s kernel source itself (the `matmul_vectorized_8x8x8_i8_i32`
  template, or the `matmul_i8_i32` wrapper `mm.cc` generates via its `combos()` macro),
  in the MLIR generated by `n1_core_i8_v2.py` (buffer/DMA descriptor setup, tiling,
  `objectFifo` wiring), or somewhere in the `aiecc` compilation/lowering pipeline
  between the two.
- Whether this is a longstanding bug that's *always* been present in every prior
  "97 tok/s" benchmark of this kernel family (plausible — `V12-CORRECTNESS-BLOCKER.md`
  already established that no prior benchmark ever validated output correctness,
  only speed and "doesn't crash"), or a regression introduced by a more recent
  change to `mm.cc`/the generator.
- Whether the AI Engine Simulator would actually help pin this down further. It's
  the natural next tool (cycle-accurate, can dump every AIE core's internal state),
  but per `docs/FUSED-INTEGRATION-BLOCKER.md`, real NPU2 topology data isn't
  available — the `aie2p_8x4_device.json` that exists on this machine
  (`~/Xilinx/2026.1/data/aie2ps/devices/`) is a symlink to a generic Versal AI Edge
  device file (`XC2VE3804.json`), not Strix Halo's actual topology. A simulation run
  against it could be useful as a coarse sanity check of the kernel's *logic* (does
  it even attempt the right sequence of operations) but its cycle-level/numeric
  fidelity to this specific chip is unverified and shouldn't be trusted without
  independent confirmation.

## Reproduction

Build the i32 kernel and xclbins:
```
bash /home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/build_all_int8_i32.sh
```

Run the isolated synthetic GEMM test (Test A) against any production xclbin:
```
g++ -std=gnu++17 -O3 -o test_gemm_i32 test_gemm_i32.cpp \
    -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil -luuid -lm
./test_gemm_i32 build/int8/final_i8_D_v.xclbin build/int8/insts_i8_D_v.txt 128 3072 1024
```
(adapted from `npu-sandbox/npu-infer/tools/test_gemm_i8.cpp`, changed to int32
output/accumulation to match the current kernel)

Reproduce Test B (real-bytes dump-and-compare): add a one-shot dump of `Am`,
`layerB[l]`, and `Cm` inside `I8Ctx::go_col()` right after
`bC->sync(XCL_BO_SYNC_BO_FROM_DEVICE)`, guarded on `l==0 && strcmp(name,"D")==0`
(or `"QKV"`), then in Python:
```python
import numpy as np
Am = np.fromfile('dump_Am.bin', dtype=np.int8).reshape(MD, KD)
Bm = np.fromfile('dump_Bm.bin', dtype=np.int8).reshape(KD, ND)
Cm = np.fromfile('dump_Cm.bin', dtype=np.int32).reshape(MD, ND)
ref = Am.astype(np.int64) @ Bm.astype(np.int64)
# compare Cm[:npt] against ref[:npt]
```

## Status (Updated 2026-07-17)

- ✅ **Single-core xclbins verified CORRECT on real hardware!** All four production
  xclbins (QKV, O, GU, D) pass the INT32 oracle with 0 errors across 10000+ elements.
  Built using AMD's reference `single_core.py` generator from mlir-aie.
- ✅ Landed on `1bit-systems`: int16/int32 width fix, prompt fix, RMSNorm weight clip.
- ✅ `backend_manager.cpp`: `npu_xrt` backend re-enabled (`auto_selectable = true`).
- 🔧 **Multi-core 8-tile data distribution still buggy.** The 8-core (`n1_core_i8_v2.py`)
  xclbins produce wrong output due to a DMA `dims_to_stream` / kernel micro-tile layout
  mismatch. The `matmul_i8_i32` function from `mm.cc` reads data in 8×8 micro-tile order,
  but the MLIR generator streams data in row-major order. Fixing the MLIR generator's
  data tiling to match the vectorized kernel's expectations is the next step.
  See `bf16_kernel_dev/CORRECT-v6-VECTORIZED-ANALYSIS.md` for detailed analysis.
- `backend_manager.cpp` now auto-selects `npu_xrt` (single-core path). Throughput is
  ~12 tok/s on Strix Halo (vs 97 tok/s multi-core target).

## Status (Updated 2026-07-18) — reconciled branch state, unverified in combination

This work continued past the 07-17 status above, but split across two separate,
divergent local branches that were only discovered and reconciled after the
fact (both had gone dangling — orphaned by unrelated `git reset --hard`
operations elsewhere in the repo's history — and were recovered via `git
reflog`). Recording exactly what was independently verified vs. what is a
claim from the reconciliation, since the two are not the same thing:

- ✅ **O and D shapes at 8-core, independently verified.** One branch fixed
  and verified the 8-core xclbins for the `O` and `D` GEMM shapes specifically
  (2.47x faster than single-core), leaving `QKV` and `GU` on the single-core
  path because 8-core still had the multi-N-group bug described above for
  those two shapes.
- ✅ **Root cause of the multi-N-group bug identified and independently
  fixed for all 4 shapes**, on the *other* branch: for `N > 1024` (multiple
  N-groups per output column), the broadcast-A fifo's B-descriptor BDs for
  different N-groups were firing concurrently on the same fifo, causing the
  core to read B subtiles from the wrong N-group mid-K-iteration. Fix:
  sequence A + B + C strictly per N-group, awaiting each N-group's C
  completion before starting the next N-group. This branch's own testing
  reported all 4 shapes (QKV, O, GU, D) passing the INT32 oracle at 8-core.
- ⚠️ **These two branches were never combined until this reconciliation.**
  The second branch's N-group sequencing fix was built on top of an older
  base that did NOT include the first branch's O/D 8-core xclbin work — so
  its "all 4 shapes PASS" result was demonstrated against its own (single-core
  QKV/GU, differently-built O/D) xclbin set, not the specific xclbin/instruction
  combination now sitting in this repo after reconciliation. The reconciliation
  took the first branch's binary `.xclbin` files (including its 8-core O/D
  builds) as the base and layered the second branch's newer `insts_i8_*.txt`
  instruction streams on top, since the instruction-stream fix is what
  actually encodes the N-group sequencing correction. **This exact combined
  set of files has not been run against the INT32 oracle on real hardware.**
  Until that re-run happens, treat "all 4 shapes correct at 8-core" as a
  plausible-but-unverified claim for this specific combination, not a
  confirmed one — even though each half was separately confirmed on its own
  differently-based branch.
- **Before trusting this in production**: re-run the Test A/Test B
  reproduction steps above against all four current `.xclbin` +
  `insts_i8_*.txt` pairs in `engine/npu/xclbins/` and confirm 0 errors across
  all four, on real Strix Halo hardware, in this exact combined form.
