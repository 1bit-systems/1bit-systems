# Spec-Decode 0% Acceptance — Root Cause Analysis & Fix Plan

**Date:** 2026-07-08  
**Scope:** `/home/bcloud/spec-decode/` — speculative decoding for XDNA-2 NPU, Qwen3-0.6B INT8 target.  
**Verdict:** The task's stated premise (FP32-vs-INT8 hidden-state mismatch) is **correct for the production default checkpoint** (`dspark_draft.bin`), but it is **not the whole story**. The experiment that tried to fix it (`eagle3_draft_npu_1k.bin`) introduced *different* bugs and also yields 0%. This report disentangles the two.

---

## 1. Current Architecture

### 1.1 System flow

```
Draft (CPU: DSpark 5-layer *default* | Eagle3 1-layer)
    │  proposes block_size=7 candidate tokens
    ▼
Target (NPU: NPUQwen3Target — 28-layer Qwen3-0.6B, 4×INT8-GEMM xclbins)
    │  verifies candidates in one forward_with_kv() pass
    ▼
Rejection sampling → accept/reject → next iteration
```

- **Orchestrator:** `engine/spec_decode.h` — `SpeculativeDecoderT<DraftModelT>`. Templated; defaults to **DSpark** (`DefaultDraftModel = DSparkDraftModel`). Runs the draft autoregressively for `block_size` steps, feeds `[last_token | draft_tokens…]` to `target.forward_with_kv()`, does greedy rejection, takes a bonus token, rolls back rejected KV via `commit_accepted()`.
- **Target:** `engine/npu_target_model.h` — `NPUQwen3Target`. Real NPU dispatch (QKV/O/GU/D INT8 xclbins, hand-rolled RMSNorm/RoPE/attn/SwiGLU on host, `FIXED_ASCALE = 8.0/127`). Exposes layer hidden states via `get_layer_hidden()` in the **feature-major interleaved** layout the draft's `fc` projection expects: `[l0_h0, l1_h0, …, l4_h0, l0_h1, …]`.

### 1.2 The two draft models

| Model | File | Layers | Params | Extra heads | Notes |
|-------|------|:------:|:------:|:------------|-------|
| **DSpark** (default) | `draft/dspark_draft.h` | 5 | ~1.4B | Markov (rank 128) + confidence | Production draft. Cross-attention to target features + autoregressive KV cache. |
| **Eagle3 (MTP)** | `draft/mtp_draft.h` | 1 | ~336M | none | Archived; single-position cross-head attention. |

Both match the DeepSpec Python reference (`Qwen3DSparkModel` / `Qwen3Eagle3Model`) exactly. C++ inference was verified **bitwise** against Python (STATUS.md, 2026-07-08). **The C++ draft math is correct.**

### 1.3 The two training pipelines

| Pipeline | Trainer | Hidden-state source | Per-position? | Loss target |
|----------|---------|---------------------|:-------------:|-------------|
| **A. DSpark (production)** | `train_dspark.py` | **HuggingFace** `AutoModel.from_pretrained("Qwen/Qwen3-0.6B")` (BF16) | ✅ yes, `[S, NTL·H]` | assistant tokens only (loss-masked) |
| **B. Eagle3-NPU (experiment)** | `train_from_cache.py` | **NPU INT8** (via `tools/capture_npu_hidden.cpp`) | ❌ **last position only** | question tokens (no assistant!) |

- Pipeline A (`train_dspark.py`) is **architecturally correct**: per-position hidden states, chat template (`<|im_start|>user/assistant`), loss masked to assistant tokens only. It produces `checkpoints/dspark_draft.bin` (1.7 GB).
- Pipeline B was built to *test* the FP32/INT8 fix. It captures NPU hidden states and trains `checkpoints/eagle3_draft_npu_1k.bin` (1.3 GB, loss 9.1→4.19).

---

## 2. Root Cause Analysis

There are **two independent 0%-acceptance causes**, one per checkpoint. Both must be understood.

### 2.1 Production DSpark checkpoint — **FP32/INT8 distribution mismatch** (the task's premise ✅)

- `dspark_draft.bin` (the default loaded by `engine/npu_single_binary.cpp:46` `kDraftCheckpoint`) was trained on **HuggingFace BF16** hidden states (`train_dspark.py:217,398` `AutoModel.from_pretrained`).
- At inference the draft receives **NPU INT8** hidden states (dequantized to FP32) from `NPUQwen3Target::get_layer_hidden()`.
- The NPU's INT8 weight+activation quantization (`FIXED_ASCALE`, per-GEMM rescale) produces a materially different distribution from the full-precision HF reference. The draft's `fc` projection + 5 decoder layers were fit to the FP32 manifold, so on NPU features its logits are off-distribution → 0% greedy match.
- **This is exactly the cause the task describes, and it is real for the production checkpoint.**

### 2.2 Eagle3-NPU experiment — **broken training objective** (a *second*, independent 0% cause)

This checkpoint used NPU INT8 features (so 2.1 is solved for it), but still gets ~0% because of three compounding data/objective bugs in pipeline B:

1. **Wrong capture position.** `tools/capture_npu_hidden.cpp` runs the NPU on the full tokenized input and snapshots `layer_hidden_snapshot_[l]` = the hidden state at the **last token only**. `train_from_cache.py` then seeds the draft's `h = fc(target_features)` with that end-of-sequence state, but teacher-forces on tokens starting from **token 0**. The seed position and the supervised tokens are uncorrelated. (The team discovered this on 2026-07-08 and wrote `tools/recapture_prompt_features.cpp` + `tools/recapture_split.cpp` to capture at the prompt boundary instead — **but no training script consumes that corrected format yet**.)

2. **No assistant responses in the training set.** `tools/prepare_npu_inputs.py` tokenizes **only the `user` content** (the math question). Verified: `npu_inputs_1000.bin` = 1000 examples, **avg 45 tokens** — a question, with no assistant answer. The draft is therefore trained to do **next-token-prediction on questions**, i.e. to reconstruct the prompt. It never learns to predict the assistant continuation it must produce at inference.

3. **Insufficient scale + weak architecture.** Only 1,000 examples × 5 epochs; loss plateaued at ~4.19 (perplexity ~66, ~1.5% top-1). With block_size=7 the expected *block* acceptance from that is only ~10%, and 0-in-15 rounds is plausible noise. `train_from_cache.py` also uses the 1-layer Eagle3, which STATUS.md notes "may not capture NPU INT8 distribution well enough."

**Net:** Pipeline B's 0% is dominated by the objective bugs (2.2.1–2.2.2), *not* by FP32/INT8. Fixing only the quantization gap (the task's framing) would **not** rescue pipeline B; and pipeline A — which *would* be rescued by closing the quantization gap — has the correct objective already.

### 2.3 What is *not* the problem

- ✅ C++ draft inference is bitwise-correct vs Python (verified).
- ✅ Feature **layout** is consistent: capture→build_cache→train and inference `get_layer_hidden` both use feature-major interleave (`[l0_h0,l1_h0,…]`). Confirmed by code + size math (`npu_hidden_1000.bin` = 20.7 MB matches exactly).
- ✅ Spec-decode orchestrator (rejection, bonus token, KV rollback) is sound.
- ✅ The NPU target produces real, coherent tokens standalone (`--bench-real`: 3.0s/4tok, "Real tokens").

---

## 3. Fix Plan

**Recommended strategy: fix the *quantization gap* on the architecturally-correct pipeline A**, because pipeline A already has per-position features, chat templating, loss masking, and the stronger 5-layer DSpark model. Two complementary approaches — do **B first** (cheap, fast), then **A** (principled).

### Fix B (PRIMARY — small effort, bridges the gap directly): Input calibration layer

Keep the HF-trained `dspark_draft.bin`. Insert a tiny learnable adapter between the NPU's raw hidden states and the draft's `fc` projection so the draft sees an FP32-like distribution.

**B.1 — Collect a paired calibration set (cheap, ~200–1k examples).**
For each prompt, capture the **same position's** hidden states from *both* backends and store them aligned:
- HF:   `h_hf   ∈ ℝ^{S×NTL·H}` (full precision)
- NPU:  `h_npu  ∈ ℝ^{S×NTL·H}` (INT8-dequantized)

New tool `tools/capture_paired_hidden.cpp`: runs `NPUQwen3Target::forward()` (NPU) and a one-shot HF forward (or reuse `TargetFeatureExtractor` from `train_dspark.py`) on identical inputs, dumps both. **No assistant responses needed** — this is a pure distribution-alignment dataset.

**B.2 — Train a calibration head (minutes).**
New script `tools/train_calibration.py` learns an affine map per target-layer block:
```
h_cal = γ ⊙ LayerNorm(h_npu) + β        # γ,β ∈ ℝ^{NTL·H}
MSE / cosine loss:  h_cal ≈ h_hf
```
~5120 parameters. Trivially fast on CPU.

**B.3 — Wire it into inference.** Add the calibration as the first step of `DSparkDraftModel::forward()` (and `MTPDraftModel::forward()`).

`draft/dspark_draft.h` — add fields + apply in `forward()`:
```cpp
struct DSparkDraftConfig {
    // ... existing ...
    bool use_input_calibration = false;     // NEW
};
// In DSparkDraftWeights, add:
std::vector<float> cal_gamma;   // [num_target_layers * hidden]   NEW
std::vector<float> cal_beta;    // [num_target_layers * hidden]   NEW

// At top of forward(), right after step "1. On new round", before linear(fc,...):
//   cached_target_features_ = γ ⊙ RMSNorm(trunk_hidden) + β   (when use_input_calibration)
```
Export path: append `cal_gamma`, `cal_beta` to the `.bin` in `train_dspark.py::export_to_binary()`.

**Why this works:** INT8 quantization is roughly a monotonic, layer-localized rescaling; a per-feature affine on RMSNormed inputs absorbs the bulk of the mean/variance drift. This is the standard "quantization-aware input adapter" trick. Expected to move acceptance **0% → 20–50%** immediately, and it's the smallest possible change to the verified-correct draft math.

**Effort: S–M.** ~1–2 days.

---

### Fix A (PRINCIPLED — large effort, removes the gap at the source): NPU per-position cache → `train_dspark.py`

Retrain the DSpark draft end-to-end on **NPU-generated** hidden states, reusing the correct `train_dspark.py` framework. This eliminates 2.1 entirely (no domain gap) because train and inference use the *same* NPU INT8 path.

**A.1 — Make the NPU target emit per-position layer features.**
Currently `NPUQwen3Target::batch_forward()` only snapshots the **last** position (`layer_hidden_snapshot_[l]`, a single vector). Capture all positions for the target layers.

`engine/npu_target_model.h` — extend the snapshot:
```cpp
// Add member:
std::vector<std::vector<float>> layer_hidden_allpos_;  // [NC][max_n*H]  NEW
// In batch_forward(), per layer l, after h_b is finalized for that layer, copy ALL n positions:
for (int pi = 0; pi < n; pi++)
    memcpy(&layer_hidden_allpos_[l][pi*H], &h_b[(size_t)pi*H], H*4);
// Add accessor:
void get_layer_hidden_positions(int32_t layer_id, int32_t n_positions, float* out /*[n*H]*/);
```

**A.2 — New capture tool `tools/capture_npu_cache.cpp`** (replace `capture_npu_hidden.cpp`):
- Read full chat-formatted sequences (prompt **+ assistant response**) from a JSONL via a Python tokenizer pre-step (extend `prepare_npu_inputs.py` to emit the *full* conversation with a loss mask, like `train_dspark.py:295-318` does — user=0, assistant=1).
- Run `NPUQwen3Target::forward()` once per example.
- Emit a **DeepSpec-format cache** matching what `train_dspark.py::train_from_cache()` already expects:
  ```python
  {"input_ids": LongTensor[S],
   "loss_mask":  UInt8Tensor[S],                 # 1 on assistant tokens
   "target_hidden_states": BFloat16Tensor[S, NTL·H],   # per-position, NPU INT8
   "target_last_hidden_states": BFloat16Tensor[S, H]}
  ```

**A.3 — Train.** `train_dspark.py` already supports a cache path; point `CACHE_PATH` at the NPU cache and run its existing `train_from_cache()`. Scale to **≥10K examples, 10 epochs** (the DSpark config's intended `global_batch_size=512`). Note the feature-major vs layer-major layout must match what the trainer's `Qwen3DSparkModel` consumes — verify against `train_dspark.py::extract()` (`[B,S,NTL*H]` concat-along-hidden, i.e. layer-major per position). **Double-check this ordering** between capture and the model's `fc` input; a transpose bug here would silently reproduce 0%.

**A.4 — Export & swap** the checkpoint into `checkpoints/dspark_draft.bin`.

**Throughput note:** capture is one NPU prefill/example (~20–30 ex/min observed), so 10K examples ≈ **7–10 h overnight** — feasible. Per-position storage ≈ 10K × 100 tok × 5 × 1024 × 2 B ≈ 10 GB — fine.

**Effort: L.** ~3–5 days (capture rewrite + overnight capture + retrain + validate).

---

### Fix C (FALLBACK — bypass the draft): confidence-cascade token router

If A/B don't reach useful acceptance, STATUS.md documents a **cascade router** (`token-router/DESIGN.md`): NPU generates all tokens; only low-confidence tokens are re-verified on GPU. No draft model, no acceptance problem. Trade-off: needs the iGPU/dGPU as a verifier.

**Effort: M** (separate project). Use only if A+B stall.

---

## 4. Recommended Sequence & Effort

| Step | Action | Files | Effort | Expected Δ acceptance |
|------|--------|-------|:------:|:---------------------:|
| **1** | Fix B (calibration layer) on production `dspark_draft.bin` | `tools/capture_paired_hidden.cpp` (new), `tools/train_calibration.py` (new), `draft/dspark_draft.h`, `draft/mtp_draft.h`, `train_dspark.py` (export) | **S–M** | 0% → 20–50% |
| **2** | Validate with `tools/test_acceptance.cpp` on a fresh prompt | — | S | confirm >0% |
| **3** | Fix A (NPU per-position cache → retrain DSpark) | `engine/npu_target_model.h`, `tools/capture_npu_cache.cpp` (new), `tools/prepare_npu_inputs.py` (full conv.), `train_dspark.py` (CACHE_PATH) | **L** | → 60–85% |
| **4** | If 1+3 still <50%, pivot to cascade router (Fix C) | `token-router/*` | M | n/a (no draft) |

**Start with Fix B.** It is the smallest change to *verified-correct* code, directly attacks the task's identified root cause, and will tell you within a day whether the quantization gap is the dominant factor or whether deeper issues (architecture/scale) remain.

---

## 5. Key Code References

- Orchestrator: `engine/spec_decode.h` — `SpeculativeDecoderT::generate()` (draft loop, verify, rejection).
- NPU target + feature layout: `engine/npu_target_model.h` — `NPUQwen3Target::batch_forward()`, `get_layer_hidden()` (feature-major interleave), `FIXED_ASCALE`.
- Drafts: `draft/dspark_draft.h` (`forward()` step 1 = `fc(hidden_norm)` seed point for calibration), `draft/mtp_draft.h`.
- Correct trainer (HF source, but right objective): `train_dspark.py` — `TargetFeatureExtractor.extract()` (per-position hooks), `build_conversation_collator()` (chat template + loss mask), `prepare_target_cache()`, `train_from_cache()`.
- Broken trainer (NPU source, wrong objective): `train_from_cache.py`, `tools/capture_npu_hidden.cpp` (last-position-only), `tools/prepare_npu_inputs.py` (user-content-only).
- In-progress position fix (no consumer yet): `tools/recapture_prompt_features.cpp`, `tools/recapture_split.cpp`.
- Acceptance test: `tools/test_acceptance.cpp`.
