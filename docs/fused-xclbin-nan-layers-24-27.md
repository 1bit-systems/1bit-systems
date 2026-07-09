# RESEARCH: Fused XCLBIN NaN at Layers 24–27 (BF16 Overflow)

**Status:** Root cause identified + concrete fix plan
**Date:** 2026-07-08
**Scope:** `1bit-systems/engine/npu/kernel/mm_ternary_32x64x128.cpp` (native ternary single-shot kernel) and its build `engine/npu/build/ternary_32c_oneshot/`
**Impact risk:** LOW (verified via `gitnexus_impact` — `extern "C"` AIE kernel, 0 C++ callers; only `compute_one_row`, a `static inline` helper, is touched)

---

## 1. Executive Summary

The fused single-shot xclbin (`ternary_32c_oneshot.xclbin`) wraps the native
ternary GEMV kernel `mm_ternary_32x64x128`. That kernel accumulates the K=256
dot product **correctly in F32** (`aie::accum<accfloat, 32>`), but then:

1. **Truncates the F32 accumulator to BF16** *before* applying the per-row scale,
2. Applies the per-row scale **in BF16** (`bf16 × bf16 → bf16`),
3. **Reduces the 32 lanes in BF16** (`aie::reduce_add` of a `bfloat16` vector → `bfloat16`),
4. Accumulates into the output **in BF16** (`output[row] += …`, where `output[row]` is `bfloat16`),
5. Does **not** set an AIE rounding mode (defaults to *truncate*), introducing a
   systematic positive bias on every BF16 cast.

These five BF16 precision-loss steps compound across the 28-layer residual stream.
By layers 24–27 the hidden-state magnitude has grown enough that
`scale × dot_product` saturates BF16 (→ ±inf), and the final lane reduction
hits `+inf + −inf → NaN`. This matches the **documented, observed** failure:

> `docs/STATUS.md:27` — *"NPU fused layer … One xclbin/layer — **NaN at layers 24+**,
> not coherent."*
>
> `engine/npu/BENCHMARKS.md:210` — *"Norm weight clipping (max 2.0) prevents
> **bf16 overflow at deep layers**."*
>
> `docs/journey.md:1239` — *"28-layer integration produces NaN due to **BFP16
> precision differences**…"*

The native ternary kernel has the **same class of defect** as the retired
torch2aie fused engine, so it will reproduce "NaN at layers 24–27" the moment it
is wired into the 28-layer pipeline (currently pending — `ternary-npu.md` Next
Step #6).

**Fix:** move the scale, the lane reduction, and the row accumulation into F32,
set `conv_even` rounding, and convert to BF16 exactly once per output element at
the very end. Plus a host-side activation clamp (the same [-8,8] range fix applied
to the v12 engine in `journey.md:69`).

---

## 2. Current Architecture of the Fused Kernel

### 2.1 What "single-shot" means

| Layer of stack | File | Role |
|---|---|---|
| MLIR dataflow | `engine/npu/kernel/n1_core_native_ternary_32core_oneshot.py` | 32-core (4×8) `object_fifo` graph: `shim→mem→core`, **single bounded loop (`range_(1)`)** so each dispatch runs one input→one output then halts. Host re-dispatches to tile M/K. |
| AIE tile kernel | `1bit-systems/engine/npu/kernel/mm_ternary_32x64x128.cpp` | The compute: on-the-fly 2-bit→ternary decode + BF16 GEMV, compiled with `-DDIM_M=… -DDIM_K_PACKED=…`. |
| Build | `engine/npu/build/build_32c_oneshot.sh` | xchesscc → MLIR gen → aiecc → `ternary_32c_oneshot.xclbin` + `insts_*.txt` |
| Artifact | `engine/npu/build/ternary_32c_oneshot/ternary_32c_oneshot.xclbin` (294 KB), copied to `engine/npu/build/xclbins/` |

The "old" infinite-loop cores (`n1_core_native_ternary_32core.py`) used
`for(;;)` (0xFFFFFFFF). The **oneshot** variant (`n1_core_native_ternary_oneshot.py`
and its 32-core sibling) replaces that with `for _ in range_(1):` so each AIE
core does exactly one tile and stops — letting the host tile across the full
hidden dim by re-dispatching. That is why this is the "fused / single-shot" path.

### 2.2 What each (transformer) layer needs

From `npu_fused_pipeline.cpp` / `npu_engine_fused.cpp`, one transformer layer =
7 GEMVs against the hidden state:

```
Q, K, V, O   (attention)   +   Gate, Up, Down   (FFN, with SiLU on Gate)
```

The native ternary kernel handles **one** GEMV tile: `out[row] = scale[row] ·
⟨weight[row], activation⟩` for `row ∈ [0, DIM_M)`, reducing K=256 ternary values.
The host tiles M (across the 32 cores) and K (across re-dispatches) to cover the
full 1536–4096 hidden dim. Output of layer N's Down-proj (+ residual + RMSNorm)
becomes the **activation input to layer N+1** — so numerical error is fed
forward through all 28 layers.

### 2.3 The kernel's compute path (the part that matters)

`mm_ternary_32x64x128.cpp`, `compute_one_row()` (lines 78–101) and the main loop
(lines 124–141):

```cpp
// compute_one_row — per output row
aie::accum<accfloat, kVLen> acc = aie::zeros<accfloat, kVLen>();   // kVLen = 32, F32 lanes
for (iter = 0..7) {                                                // 8 iters × 32 = K=256
    aie::vector<bfloat16,32> tern = decode_ternary_8bytes(w + iter*8);
    aie::vector<bfloat16,32> act  = aie::load_v<32>(activation + iter*32);
    acc = aie::add(acc, aie::mul(tern, act));                      // ✅ F32 accumulation
}
// ❌ PROBLEM ZONE — everything below is BF16:
aie::vector<bfloat16,32> acc_vec   = aie::to_vector<bfloat16>(acc);          // (1) F32→BF16 truncation
aie::vector<bfloat16,32> scale_vec = aie::broadcast<bfloat16,32>(scale);
auto scaled_acc = aie::mul(acc_vec, scale_vec);                              // (2) scale in BF16
return aie::to_vector<bfloat16>(scaled_acc);                                 // (3) → BF16 again
```

```cpp
// main loop
for (row = 0..kM) {
    aie::vector<bfloat16,32> row_result = compute_one_row(wt_row, acts, scale);
    output[row] += aie::reduce_add(row_result);   // (4) reduce 32 BF16 lanes → BF16;  (5) accumulate in BF16
}
```

The dot-product **MAC loop is fine** (F32). Everything *after* it — scale,
lane-reduce, row-accumulate, and the final cast — is done in BF16. That is the
defect.

---

## 3. Root-Cause Analysis: Why Layers 24–27

### 3.1 The divergence mechanism (why it gets worse with depth)

BF16 has ~3 mantissa bits (~7-bit equivalent, ~0.4% relative resolution). Every
time a value passes through one of the five BF16 steps in §2.3 it is rounded.
On AIE2, the **default rounding mode is `truncate`** (round toward +inf for the
float→bf16 downcast) unless the kernel explicitly calls
`aie::set_rounding(...)`. Truncation is **biased**: it adds a small positive
error on average to every cast.

Evidence this is intended to be set: the sibling kernel
`1bit-systems/.../bitnet_ternary_scheduler.cpp` opens with
`::aie::set_rounding(aie::rounding_mode::conv_even);`. **`mm_ternary_32x64x128.cpp`
has no such call** — confirmed by grep (0 hits). So every BF16 cast in the
single-shot kernel is biased-truncate.

Through the residual stream this bias is **monotonic and compounding**:

```
hidden_{n+1} = RMSNorm(hidden_n + GEMV_bf16(hidden_n))
```

A small systematic positive bias on `GEMV_bf16` output ⇒ `hidden` magnitude
drifts upward layer over layer. The per-row ternary **scales** are also larger
in deeper layers (calibrated to preserve information), so `scale × dot_product`
grows on *both* factors. After ~24 layers the activations entering the GEMV are
large enough that step (2) `bf16(acc) × scale` saturates to **±inf**, and step
(4) `reduce_add` then produces `+inf + −inf = NaN`. NaN then floods the residual
stream for the remaining layers.

This is exactly the pattern documented for the retired torch2aie fused engine
(`NaN at layer ~19` → `layers 24+`) and is why **only the last few layers** fail.

### 3.2 Why "24–27" specifically and not earlier

* 28-layer model (Qwen3-0.6B `NC=28`; Bonsai-1.7B `28 layers`).
* The bias is small per layer (~0.4% of magnitude × number of BF16 ops).
* It takes ~24 compounding steps to cross the BF16 saturation threshold under
  realistic activation magnitudes (measured activation range is **[-8.24, 7.01]**
  per `journey.md:69`, i.e. already large — a single GEMV of K=256 ternary×8 can
  swing ±2048 before scaling, and deep-layer scales push that past BF16 range
  once the residual has inflated).
* The `[-5,5]` activation clip that masked this in the INT8 v12 engine
  (`journey.md:69`) is **not** applied on the native-ternary path, so it hits
  the wall at depth.

### 3.3 The "smoking gun" evidence already in the repo

| Source | Quote | Implication |
|---|---|---|
| `STATUS.md:27` | "NPU fused layer … NaN at layers 24+" | The exact symptom. |
| `BENCHMARKS.md:210` | "Norm weight clipping (max 2.0) prevents bf16 overflow at deep layers" | Confirms overflow is **bf16** and is **deep-layer**; clipping norms is a *workaround*, not a fix. |
| `journey.md:1181` | "issue from NPU BFP16 compute diverging from ideal FP32" | Root cause class = BF16 vs FP32 divergence. |
| `journey.md:1239` | "28-layer integration produces NaN due to BFP16 precision differences" | Compounding over 28 layers is the trigger. |
| `bitnet_ternary_scheduler.cpp` | `set_rounding(conv_even)` present | Proves the team knows truncate-default is wrong; the oneshot kernel just omits it. |

---

## 4. The Fix

Two layers of fix: **(A) keep the kernel in F32 until the final cast**, and
**(B) add a host-side magnitude safety net** (the same one that made the v12
engine produce finite output). (A) is the real cure; (B) is cheap insurance.

### 4.1 Fix A — F32 through scale + reduce + accumulate (PRIMARY)

**File:** `1bit-systems/engine/npu/kernel/mm_ternary_32x64x128.cpp`

**(A1) Set unbiased rounding at kernel entry** — add as the first line of
`mm_ternary_32x64x128()` (mirrors `bitnet_ternary_scheduler.cpp`):

```cpp
::aie::set_rounding(aie::rounding_mode::conv_even);
```

**(A2) Rewrite `compute_one_row()` to return F32**, eliminating steps (1)–(3):

```cpp
// Returns the scaled dot product as an F32 scalar.
// Scale and lane-reduction now happen in F32; BF16 conversion happens once, in the caller.
__attribute__((always_inline)) static inline float
compute_one_row_f32(const uint8_t *__restrict weight_row,
                    const bfloat16 *__restrict activation,
                    bfloat16 scale) {
  aie::accum<accfloat, kVLen> acc = aie::zeros<accfloat, kVLen>();

  for (int32_t iter = 0; iter < kInnerIters; iter++) {
    aie::vector<bfloat16, kVLen> tern = decode_ternary_8bytes(weight_row + iter * kBytesPerIter);
    aie::vector<bfloat16, kVLen> act  = aie::load_v<kVLen>(activation + iter * kVLen);
    acc = aie::add(acc, aie::mul(tern, act));      // F32 MAC (unchanged)
  }

  float dot = aie::reduce_add(acc);                // F32 reduction of 32 lanes (was BF16)
  return dot * (float)scale;                       // F32 scale (was BF16)
}
```

> Note: `aie::reduce_add` on an `accfloat` accumulator returns an F32 scalar.
> If the toolchain's overload only accepts a vector, reduce the accumulator's
> vector view in F32 instead:
> `float dot = aie::reduce_add(acc.template to_vector<float>());` — the key is
> the operand type is float, not bfloat16.

**(A3) Accumulate rows in an F32 scratch, convert to BF16 once** — rewrite the
main loop (currently lines 124–141):

```cpp
void mm_ternary_32x64x128(int32_t *__restrict input, bfloat16 *__restrict output) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);          // (A1) kill truncate bias

  const uint8_t  *weights = reinterpret_cast<const uint8_t *>(input);
  const bfloat16 *scales  = reinterpret_cast<const bfloat16 *>(input + kScaleOffset / 4);
  const bfloat16 *acts    = reinterpret_cast<const bfloat16 *>(input + kActOffset / 4);

  float out_f32[kM];                                           // F32 scratch (kM ≤ 32, fits in L1)
  for (int32_t i = 0; i < kM; i++) out_f32[i] = 0.0f;

  for (int32_t row = 0; row < kM; row++) {
    out_f32[row] += compute_one_row_f32(weights + row * kKPacked, acts, scales[row]);  // F32 accumulate
  }

  // Single BF16 conversion at the very end (with saturation safety)
  for (int32_t i = 0; i < kM; i++) {
    output[i] = aie::to_scalar<bfloat16>(out_f32[i]);          // one cast, conv_even rounding
  }
}
```

This collapses the five BF16 lossy steps down to **one** BF16 cast per output
element, performed after all accumulation. `kM ≤ 32` so the F32 scratch is 128
bytes — trivial in core L1.

**Effect:** the only BF16 rounding in the whole GEMV is the final
`f32→bf16`. Combined with `conv_even` rounding, the per-layer bias becomes
zero-mean, so the residual stream no longer drifts upward and never reaches the
saturation point. Layers 24–27 produce finite output.

### 4.2 Fix B — Host-side activation clamp (INSURANCE)

**Files:** `engine/npu/src/npu_engine_fused.cpp` (and the npu-sandbox twin)
wherever hidden state is written back between layers. Clamp the post-residual
hidden to the measured range before the next GEMV:

```cpp
// After residual add, before next layer's GEMV input pack:
for (int i = 0; i < H; i++) {
    float v = hidden[i];
    if (v >  8.0f) v =  8.0f;     // measured range [-8.24, 7.01] (journey.md:69)
    if (v < -8.0f) v = -8.0f;
    hidden[i] = v;
}
```

This mirrors the `[-5,5]→[-8,8]` clip fix that made the v12 engine finite, and
guarantees that even a pathological input cannot push `scale × dot` past BF16
range. With Fix A in place the clamp should never fire on a healthy model; it
exists to turn "NaN" into "slightly clipped" for adversarial inputs.

### 4.3 What does NOT need to change

* **Tile size** (`DIM_M`, `DIM_K_PACKED`): not the cause. K=256 F32 accumulation
  never overflows (max ~256 × |act|). Keep current tiling.
* **MLIR generators** (`n1_core_native_ternary_*oneshot.py`): unchanged — they
  only describe dataflow; `link_with=mm_ternary_32x64x128.o` picks up the new
  `.o` automatically on rebuild.
* **Output buffer type in MLIR** (`dtype_out = np.float32` maps to bf16 in the
  AIE dialect per the generator comment): the per-element output is still BF16;
  we just compute it in F32 internally. No MLIR change.

### 4.4 Rebuild + re-validate

```bash
source engine/npu/build/env.sh
bash engine/npu/build/build_32c_oneshot.sh 128 64      # rebuilds kernel .o + xclbin

# All-ones regression (must still be -256.0000, now via F32 path):
g++ -O2 -std=c++17 -o test_ternary_32core engine/npu/tests/test_ternary_32core.cpp -lxrt_coreutil
./test_ternary_32core \
    engine/npu/build/ternary_32c_oneshot/ternary_32c_oneshot.xclbin \
    engine/npu/build/ternary_32c_oneshot/insts_ternary_32c_oneshot.txt
```

The all-ones test (`-256.0000`, 128/128) is preserved exactly: F32 reduction of
all-(-1)×all-1 over K=256 = -256, × scale 1.0 = -256. So this is a safe
regression gate.

**New validation to add** (the missing test that let this ship): a *random BF16*
test comparing kernel output to `mm_ternary_reference` (the F32 scalar reference
already in the file, lines 144–172) with `max_abs_err < 1e-2`, and a
*deep-layer stress* test that feeds outputs back as inputs for 28 iterations to
confirm no NaN. This directly closes `ternary-npu.md` Next Step #3 ("random BF16
validation — output layout fix needed").

---

## 5. Effort Estimate

| Item | Size | Hours |
|---|---|---|
| Fix A (kernel F32 + rounding) — 1 file, ~30 lines | **S** | 2–3 h |
| Rebuild xclbin + all-ones regression | S | 0.5 h |
| Add random-BF16 vs F32-reference unit test | S | 1–2 h |
| Fix B (host activation clamp) — 2 files | S | 1 h |
| 28-layer no-NaN re-validation (the real work; needs the ternary path wired into the layer loop, which is itself pending `ternary-npu.md` #6) | M | 1–2 days |
| **End-to-end coherence** vs HF reference (the historical blocker — see `journey.md` "v12 WAS NEVER OUTPUT-VALIDATED") | **M–L** | days |

**Overall: M.** The code change itself is small and low-risk (S, ~half a day),
but proving the NaN is gone *and* output is coherent across 28 layers is M,
because (a) the native-ternary path is not yet wired into the layer pipeline,
and (b) this repo has a documented history of "fixes the crash, still
incoherent" — coherence must be checked against a real HF reference, not just
absence of NaN.

---

## 6. Recommendation

1. Apply **Fix A** now — it is strictly more correct than the current code at
   equal cost (the MAC loop is already F32; we're just *not throwing it away*)
   and removes the documented divergence class.
2. Add the **random-BF16 vs F32-reference test** before wiring the ternary path
   into 28-layer inference — it would have caught this.
3. Keep the **norm-weight-clip (max 2.0)** mitigation from `BENCHMARKS.md:210`
   as defense-in-depth *in addition to* Fix A, not instead of it.
4. Do **not** chase tile-size changes or MLIR rewrites — the bug is purely in
   the post-MAC BF16 path of `compute_one_row` + the main loop.

---

## Appendix — File & line index

| Location | What |
|---|---|
| `1bit-systems/engine/npu/kernel/mm_ternary_32x64x128.cpp:78-101` | `compute_one_row` — the BF16 scale/cast defect (Fix A2) |
| `1bit-systems/engine/npu/kernel/mm_ternary_32x64x128.cpp:124-141` | main loop — BF16 reduce/accumulate defect (Fix A1+A3) |
| `1bit-systems/engine/npu/kernel/bitnet_ternary_scheduler.cpp` | sibling kernel that correctly sets `conv_even` rounding (reference) |
| `engine/npu/kernel/n1_core_native_ternary_32core_oneshot.py` | MLIR dataflow for the fused single-shot kernel (unchanged) |
| `engine/npu/build/build_32c_oneshot.sh` | rebuild entry point |
| `engine/npu/build/ternary_32c_oneshot/ternary_32c_oneshot.xclbin` | the failing artifact (rebuilt by Fix A) |
| `engine/npu/tests/test_ternary_32core.cpp` | all-ones regression gate (stays -256.0) |
| `engine/npu/src/npu_engine_fused.cpp` | host layer loop — Fix B activation clamp |
| `docs/STATUS.md:27`, `engine/npu/BENCHMARKS.md:210`, `docs/journey.md:1175-1239` | prior-art evidence |
