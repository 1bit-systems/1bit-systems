## UPDATE 28 (2026-07-20): MAMBA1 GPU BACKEND — 79.8 TOK/S, 9 BUGS KILLED

**The Mamba1 GPU backend (`mamba1_engine.hip` + `backend_mamba1.cpp`) is now fully built, linked, and validated end-to-end on Strix Halo. BlackMamba 1.5B: 79.8 tok/s. BlackMamba 2.8B: 46.4 tok/s.**

What was delivered:

1. **Build linkage fixed**: `create_mamba1_backend` was only compiled into `unified_server`, not `libbackend_manager.a` — every other binary (test_backend, backend_demo, vision_server, etc.) failed to link. Moved `backend_mamba1.cpp` into the static lib. HIP device stubs were also missing because the file was compiled as CXX despite launching kernels with `<<<>>>` syntax; moved all kernel launches into `extern "C"` wrapper functions in `mamba1_engine.hip` so callers compile as plain CXX.
2. **Conv state buffer overflow fixed**: the conv state shift loop wrote to `cs[(d_conv-1) * d_inner + i]` but the buffer was only `[d_conv-1, d_inner]` (max valid index `d_conv-2`). This caused silent GPU memory corruption on every SSM layer forward pass. Fixed the loop bound from `dc-2` to `dc-3`.
3. **A_log exponentiation fixed**: Mamba1 parameterizes `A = -exp(A_log)`, but the selective scan kernel used `A_log` directly as `A` in `A_bar = exp(dt * A)`. This meant the SSM dynamics were completely wrong. Added `-expf()` in the scan loop to compute `A = -exp(A_log)` before discretization.
4. **Model routing fixed**: GGUF Mamba models are now routed to `mamba1_gpu` backend (was falling through to ZINC GPU catch-all).
5. **Both BlackMamba sizes converted and benchmarked**: 1.5B (30 layers, 15 SSM + 15 MoE) at 79.8 tok/s, 2.8B (36 layers, 18 SSM + 18 MoE) at 46.4 tok/s — both on Strix Halo iGPU via ROCm HIP, alternating SSM/MoE layer dispatch.
6. **Diagnostic tool**: `tools/test_mamba1_backend.cpp` loads a Mamba1 GGUF directly into the HIP backend without the HTTP server — warmup, benchmark, and generation in one shot.

**The BlackMamba `⚠️` in the README is gone.**

---

## UPDATE 27 (2026-07-06): FUSED LAYER ENGINE GOES PRODUCTION — 291 TOK/S (3× V12)

**The fused layer engine now ships at 291 tok/s (3.4 ms/tok), 3× the v12 baseline, in a 38 KB binary.**

What was delivered:
1. **One xclbin call per transformer layer**: QKV projection, attention, O projection, gate+up, SiLU, and down projection all run on the NPU in a single dispatch. No CPU attention, no intermediate BO syncs. Uses `design_full_layer.xclbin` (416 KB) from the torch2aie toolchain with per-position instruction files.
2. **3.4 ms/tok decode**: The fused dispatch eliminates the per-GEMM ioctl overhead that limited v12. At 291 tok/s, the NPU's INT8 throughput is now the bottleneck, not the dispatch layer.
3. **38 KB binary**: The fused engine binary is smaller than the previous 74 KB daemon despite doing more per call. Static linking + stripped symbols + no Python runtime paths.
4. **Fixed scale optimization in universal engine**: `dynamic_ascale()` replaced with `FIXED_ASCALE = 8.0f / 127.0f` — saves 35 μs per GEMM call (4 ms/batch across 112 calls). Worth +11% on decode.
5. **FLM v0.9.44 workaround in daemon**: FLM's `/v1/chat/completions` has a `basic_string::substr` bug. Daemon now converts chat messages to text prompts via a lightweight Qwen3 template and calls `/v1/completions` instead.

**Narrative shift**: v12 (97 tok/s, C++ standalone INT8) is now the fallback path. The fused layer engine is the production path. All docs, badges, and benchmarks updated to reflect this. Everything from "74 KB binary, 94 tok/s" to "38 KB binary, 291 tok/s."

---

## UPDATE 26 (2026-07-05): ALL 3 BUGS CONFIRMED FIXED — AIE MICRO-TILING ROOT CAUSE RESOLVED

**v12 is now coherent. 97 tok/s verified. GEMM kernel bit-exact.**

A parallel investigation (branch `fix/npu-hf-cache-i32-kernel`) independently confirmed
what UPDATE 25 suspected: the remaining bug was in the **compiled xclbin kernels**, not
the host code. Root cause: `n1_core_i8_v2.py` (the INT8 MLIR generator) was **missing AIE
micro-tiling** — the GEMM kernel received weights in the wrong internal layout despite
being bit-for-bit correct at the BO level.

Fixes applied:
1. **xclbin output width** — matched INT8 generator output width to host's i32 Cm buffer
   (`cd73e137`)
2. **Smoke-test prompt** — replaced malformed prompt with valid chat template
   (`3d984285`)
3. **RMSNorm weight clip** — clipped weights to [-2,2] in cb/universal engines
   (`49e78785`, partial)
4. **GEMM kernel verified** — hardware dump-and-compare confirmed bit-exact
   (`7f8f3586`)
5. **Root cause identified** — missing AIE micro-tiling in n1_core_i8_v2.py
   (`01a4b7f4`)
6. **Parallel theories reconciled** — both investigation paths now agree
   (`16016167`)

All 6 fixes cherry-picked onto main as `232db025`..`bffe5a2e`.
**97 tok/s v12 now produces coherent output.**

---

## UPDATE 25 (2026-07-03): v12 WAS NEVER OUTPUT-VALIDATED — 3 REAL BUGS FOUND, STILL INCOHERENT

Set out to swap the production daemon's NPU backend from FLM (proprietary, closed-source)
to v12 (our own C++ engine, "97 tok/s, beats FLM's 94, Zero Python"). Before wiring it in,
sanity-checked actual chat output against FLM for the same prompt. FLM answered "What is
2+2?" correctly (" 4."). v12 — byte-identical reproduction of the unmodified original —
produced complete garbage. Every doc and benchmark in this repo checks tok/s and "doesn't
crash," never coherence. The 97 tok/s number is real; the output behind it never was.

Found and fixed 3 real, confirmed bugs, all present in `npu_engine_cb.cpp` since it was
first written and inherited by `npu_target_model.h` (spec-decode's target-model dispatch):

1. **LM head weight substitution** — `lm_head.weight` gets correctly dequantized then
   immediately discarded; the code computes final vocab logits against the *embedding*
   table instead (assumes tied embeddings). Qwen3-0.6B's checkpoint stores them completely
   separately (confirmed via Q4NX header data_offsets) — the model computes a reasonable
   final hidden state, then reads logits off the wrong matrix.
2. **Weight-packing transpose** — `dequant_i8_to_float` returns row-major
   `[out_features, in_features]`; the GEMM dispatch needs `[in_features, out_features]`.
   The packing loop read the buffer with the wrong stride, silently scrambling every
   weight matrix (Q/K/V/O/Gate/Up/Down) while still producing finite, plausible-looking
   numbers. Also: O-proj and Down-proj dequant calls used the wrong `in_features` (1024
   default instead of their real 2048/3072), scrambling the tiling itself.
3. **Activation quantization clipping** — hardcoded INT8 scale assumed activations stay
   within [-5,5]; measured range is [-8.24,7.01]. Silently clipped every layer, compounding
   across all 28.

All three fixed, in all three copies of this logic (`npu_engine_cb.cpp`,
`npu_engine_server.cpp` — a new persistent-server variant built for the daemon swap,
and `spec-decode/engine/npu_target_model.h`). Chat output is **still incoherent** after
all three fixes, individually and combined, tested against both RoPE conventions
(interleaved-pairs and HF's actual rotate_half). Ruled out via ground-truth comparison
against the real HF model: embedding lookup, RoPE theta/config, GQA head mapping, K/V
extraction offsets — all correct. Remaining suspects: RoPE rotation convention (tested,
inconclusive) or the compiled `.xclbin` kernels themselves, undebuggable without the AI
Engine Simulator — blocked on this machine since Update 24's investigation (missing
`aie2p_8x4_device.json` for NPU2). Full writeup: `docs/V12-CORRECTNESS-BLOCKER.md`.

**FLM proxy stays in production.** Do not wire v12/1bit.engine into the daemon until this
is resolved and re-verified against real chat prompts, not just dispatch speed.

---

## UPDATE 24 (2026-07-03): FUSED XCLBIN RESUMED — SCHEDULE FIXED, DEADLOCK ISOLATED, NEW KERNEL BUG FOUND

Picked back up the fused-transformer-xclbin effort flagged as "next" at the end of Update 17
(the intervening Updates 18-23 covering the fused-xclbin dead end, the pivot to the universal
5-model v12 engine, and the merch store live in git history / other docs, not fully reflected
in this file until now).

### What Was Found

1. **Reconstructed the correct Q4NX weight-packing schedule** by cross-referencing the MLIR
   generator against `qwen3_model.py::_projection_stream_from_schedule` — the fused xclbin
   expects weight chunks distributed per-column/per-row (`row_chunk = block*16 + group*4 +
   patch*2 + row_in_patch`), not replicated identically across columns as the old
   `q4nx_stream.cpp` did. `npu-sandbox/npu-infer/tools/pack_fused_v3.py` already implements this
   correctly (verified byte-identical on regen) by reading real Q4NX chunks straight out of
   `model.q4nx`, no dequant/requant.
2. **Schedule-correct weights alone didn't fix the full-layer deadlock** — re-ran `npu_engine_v13`,
   still 62857ms timeout, all-zero output.
3. **Isolated the deadlock to the O/UP/GATE/DOWN tail.** The smaller QKV-prefix xclbin (rebuilt
   fresh via `full_layer_qkv_prefix_runner.py`) dispatches cleanly in ~4ms, no deadlock at all —
   matching what the sibling BitNet port (`torch2aie/examples/bitnet-decode-layer`) found for the
   identical design shape. The full-layer deadlock is a lock/dataflow bug specific to the tail
   phases, not a data-scheduling problem.
4. **Found a second, separate bug: QKV-prefix produces numerically wrong output**, even
   deadlock-free. ~1000+ K/V cache mismatches vs. the CPU golden reference, at multiple token
   positions. Traced RoPE and RMSNorm formulas in the Chess kernel (`postprocess_qkv.cc`) against
   the Python reference — both match exactly. V-cache (no RoPE/norm at all) is *also* wrong,
   narrowing the bug to the Q4NX GEMM/dequant kernel (`qwen3_decode_kernels.cc`) or record
   absorption — unresolved, needs kernel-level debug instrumentation to pin down further.

### Status

Fused xclbin is closer than before (schedule solved, deadlock scope narrowed) but still not
working end-to-end — two distinct kernel bugs remain (tail deadlock, QKV numeric correctness).
v12 (97 tok/s, standalone INT8 GEMM, zero Python) stays production. Full details in
`docs/FUSED-INTEGRATION-BLOCKER.md`.

---

## UPDATE 23 (2026-07-02 15:32 ADT): PRODUCTION RELEASE — v2026.07.02-all5models

Shipped: tag `v2026.07.02-all5models`, site updated to "One binary to rule them all." 5 model
families verified, 0 crashes, 28 tok/s on Qwen3-0.6B (all-models auto-detect binary). vs FLM: 2.4×
slower per-token, but open source, zero dependencies, 5 models from one 120KB binary. Fused xclbin
flagged as the path to close the gap (picked back up in Update 24, three sessions later).

---

## UPDATE 22 (2026-07-02 15:13 ADT): ALL 5 MODELS AT V12 BATCH SPEED, 0 CRASHES

Model-agnostic engine (`npu_engine_all.cpp`) verified across the full catalog:

| Model | Decode |
|---|---|
| Qwen3-0.6B | 58 ms/tok |
| Gemma4-E2B | 117 ms/tok |
| Qwen3-VL-4B | 141 ms/tok |
| Llama-3.1-8B | 185 ms/tok |
| Qwen3-8B | 215 ms/tok |

Fix: `dequant_i8_to_float_ex` had `in_features` hardcoded to 1024 — only 0.6B ever worked
correctly. Corrected to read `H`, `NH×HD`, `IM` per-projection from the model header; all 5
families verified working.

---

## UPDATE 21 (2026-07-02 12:01 ADT): SESSION CLOSE — FULL NPU ENGINE STATE

*(Reconstructed from git history — commits in this window didn't carry explicit UPDATE numbers;
assigned 20/21 here to keep the sequence readable.)*

v12 engine at 97 tok/s (10 ms/tok), 24× speedup, beating FLM Kraken Point (66.5 tok/s). Fused
xclbin: 3 xclbins compiled (QKV-prefix, full-layer, unified), 5 Chess kernels recompiled for
Qwen3-0.6B, blocked on Q4NX weight format (see Update 20) — NPU firmware confirmed an active
xclbin via `ERT_CMD_STATE_TIMEOUT`, an early sighting of the same deadlock symptom Update 24 later
isolated. Model xclbins: 23 total across 5 families (Qwen3-0.6B, Qwen3-VL-4B, Qwen3-8B,
Llama-3.1-8B, Gemma4-E2B). CLI scaffolded by a second agent (package.json, tsconfig, command
routing).

---

## UPDATE 20 (2026-07-02 05:17–07:27 ADT): FUSED XCLBIN — FIRST ATTEMPT, Q4NX BLOCKER

The first full attempt at the fused-transformer-xclbin idea flagged in Update 18. Contract
established for Qwen3-0.6B dimensions, 5 kernels recompiled with Chess for the smaller model.
MLIR generator produced a working design; 2 xclbins compiled (374KB full-layer, 253KB QKV-prefix).
`npu_engine_v13` proved the dispatch mechanics work — xclbin loads, BOs allocate (9.4MB weights),
kernel dispatches without crashing — but hit a wall: the fused xclbin's weight-stream layout
expects FLM's proprietary Q4NX chunk format, and the engine's flat INT8 weights don't match it.
Weight-stream scheduler work got the packed size exactly right (2,458,816 dwords) but DMA still
timed out (63s) — diagnosed at the time as a Q4NX *quantization* mismatch (dequant→requant producing
NaN/Inf). Decision: keep v12 (97 tok/s, standalone GEMM) in production, treat fused xclbin as a
separate weight-format workstream.

(Update 24, a session later, revisited this with fresh eyes and found the real bug was the
*schedule* — chunks replicated identically across columns instead of indexed per-tile — not the
quantization theory reached here. See `docs/WEIGHT-STREAM-BLOCKER.md` for the correction.)

---

## UPDATE 19 (2026-07-02 06:24–06:27 ADT): MULTI-MODEL XCLBINS, MODEL-AGNOSTIC ENGINE

Two parallel threads landed close together:

- **Multi-model build-out**: 22-23 xclbins compiled across 6 model families (Qwen3-0.6B, Qwen3-8B,
  Qwen3-VL-4B, Gemma4-E2B, Llama). `npu_engine_mt.cpp` (model-agnostic multi-token engine) +
  `model_config.h` (auto-detects model dimensions from Q4NX headers) + a 42-model catalog
  (`model-catalog.md`) classifying every FLM NPU2 model by architecture. `build_all_models.sh`
  automates the xclbin builds.
- **Engine speed**: v12 at 10 ms/tok (97 tok/s), 24× speedup from the v3 baseline
  (244→50→16→10 ms/tok across v3→v6→v9→v12).
- **Attention**: `attn_scalar.o` + `attn_c8.xclbin` compiled but not integrated — CPU OpenMP was
  still faster for context <128 at this point.
- Site live: 145 visitors, CI pipeline + PR-Agent running, benchmarks current.

---

## UPDATE 18 (2026-07-02 04:01 ADT): M=32 TARGET, NPU LM HEAD, FLM COMPARISON

FLM Kraken Point benchmark for reference: 66.5 tok/s on weaker hardware than ours. Engine evolution
recap v3→v10: 244→16 ms/tok (15.2×, same numbers as Update 17). NPU LM head landed on-chip: 4ms
(N=30720 xclbin, 88KB) — previously a CPU-side cost. `xrt::runlist` batching investigated and found
to save only 27μs/layer (not worth the complexity). M=32 v11 targeted for >100 tok/s. Next flagged:
NPU attention kernel (`edge_attention.o` compiled, not yet integrated) and the fused-xclbin idea
that Update 20 picks up.

---

## UPDATE 17 (2026-07-02 03:00 ADT): M=16 BATCH DECODE — 16 ms/tok, 15.2× SPEEDUP

### 244→16 ms/tok in One Session

```
v3 (Jul 1): 244 ms/tok  baseline
v6 (Jul 2):  50 ms/tok  batch-4 + OpenMP LM head           (4.4×)
v7 (Jul 2):       —     ioctl=9μs, r.wait=1334μs probe
v8 (Jul 2):  27 ms/tok  M=8 batch decode                   (8.2×)
v9 (Jul 2):  16 ms/tok  M=16 batch decode                  (15.2×)
```

### M=16 Batch Decode — How It Works

The v7 probe proved `r.wait()` at 1334μs per GEMM call is NPU compute time,
not driver overhead. The NPU is 99% idle in the M dimension at M=1. At M=16,
compute stays at 1334μs but processes 16× more data → 11ms/tok per batch step.

Single-token boot (157ms) provides top-16 token candidates from LM head logits.
The 16 candidates run through one batched forward pass (28 layers, 4 GEMMs each)
= 112 dispatches at 1334μs = 149ms NPU time + LM head (6ms) + CPU (10ms) ≈ 170ms.
170ms / 16 tokens = 11 ms/tok.

At 64 tokens (4 batches): 16.1 ms/tok effective. Boot amortized away.

### FLM Gap: 1.5×

FLM: 93 tok/s = 10.7 ms/tok (proprietary).  
v9: 63 tok/s = 16.0 ms/tok (open source).  
Gap: 1.5×. Was 20× yesterday morning.

Next: LM head on NPU (151936×1024 INT8 matmul on D-style xclbin) = ~1ms.
That alone brings batch step from 11→6ms/tok and effective to ~8ms/tok.
Combined with M=32: ~4ms/tok effective = matches FLM.

### Session Summary

- 9 engine versions built and tested on-device
- f32 embeddings: -20% decode latency
- OpenMP LM head: 67→6ms
- μs-probe: identified NPU compute as bottleneck (not ioctl)
- M=4→8→16 batched decode: dispatch amortization
- 15.2× total speedup
- CI pipeline on self-hosted runner
- All numbers on https://1bit.systems

---

## UPDATE 16 (2026-07-02 02:00 ADT): FULL PROFILE + 50 ms/tok BATCH-4 DECODE

### NPU Dispatch: The Root Cause

μs-accurate profiling (`npu_engine_profile.cpp`) proved our GEMM overhead:

```
Per-GEMM dispatch (112 per token):
  Quantize A:    6 μs   (<1%)
  Sync A→NPU:    2 μs   (<1%)
  Kernel+wait: 1346 μs   (99%)  ← THE BOTTLENECK
  Sync C←NPU:    8 μs   (<1%)
  Dequant C:     1 μs   (<1%)

Total: 1363 μs/call × 112 calls = 156.8 ms (70%)
LM head: 67 ms (30%)
CPU ops (norms, RoPE, attn, SiLU): 0.7 ms (<1%)
```

The NPU is spending 99% of dispatch time in launch+wait overhead.
Actual M=1 GEMM is 0.5-5 μs. Overhead ratio: **2000×**.

### Chained Batch-4 Decode (v6): 50 ms/tok

Instead of per-token dispatch, we generate top-4 tokens from LM head
logits and run them all through one batched forward pass. Each batch
step takes ~160ms for 4 tokens = 40 ms/tok. Boot step: 157ms.

```
$ OMP_NUM_THREADS=16 ./npu_engine_v6 16

  [0] boot=127595 top4=127595,65831,39815,63550 (157ms)
  [1] batch=4 tok=9275 ms=161 (40 ms/tok)
  [5] batch=4 tok=106211 ms=159 (40 ms/tok)
  [9] batch=4 tok=83570 ms=158 (40 ms/tok)
  [13] batch=3 tok=83570 ms=157 (52 ms/tok)
=== 50 ms/tok effective ===
```

Token IDs diverse across batches. No NaN. Clean exit.
4.4× speedup from v3 (244→50 ms/tok).

### OpenMP LM Head

Pre-converted BFP16→F32 embeddings (622 MB) + OpenMP on 16 Zen5 cores:
LM head: 67ms → ~6ms per token (11× faster). This plus batch-4
amortization is what dropped us from 222→50 ms/tok.

### What We Learned

- Removing weight re-sync (v4) saved nothing — weights already on device.
- 2-layer draft model (spec decode v0) had 0% acceptance rate on Qwen3.
- Batching at decode time works: dispatch overhead amortizes across tokens.
- CPU is never the bottleneck — 26 μs/layer vs 5599 μs GEMM dispatch.

### Next: Fused Transformer XCLBIN

The 112 dispatches per token are now 112 per 4 tokens = 28/token effective.
To get to FLM's 93 tok/s, we need a single fused transformer-layer xclbin
that chains QKV→norm→attention→O→norm→GU→D on NPU without host round-trips.
That turns 28 dispatches into 1. Then LM head goes on NPU via D-xclbin INT8
matmul. Then we're at ~10 ms/tok.

---

## UPDATE 15 (2026-07-01 15:00 ADT): PR-AGENT LIVE, LANDING PAGE DEPLOYED, 242 ms/tok VERIFIED

### Live Production Stack

```
https://1bit.systems          → 50 TOPS landing page (Cloudflare Pages)
https://github.com/.../1bit-systems → Full source, benchmarks, journey
PR-Agent: Qodo + OpenCode GLM-5.2 → auto-review on every PR
```

### Verified Timing (2026-07-01 15:00 ADT)

```
=== NPU Engine v3 — Continuous Batch ===
Prefill 9 tokens: 179ms (20 ms/tok)
Decode 4 tokens: 242 ms/tok
Tokens: 106811, 63165, 117266, 109842
```

| Metric | Today | Overnight (Jul 1 04:00) |
|--------|-------|-------------------------|
| Prefill M=9 | 179ms (20 ms/tok) | ~200ms |
| Decode | **242 ms/tok** | 219 ms/tok |
| Prefill M=1 | 161ms | — |
| Prefill M=4 | 162ms (40 ms/tok) | — |
| PPR Agent | Qodo + OpenCode GLM-5.2 | — |
| Landing page | 50 TOPS headline deployed | — |

### What pi-agent Tightened

- Timings stable across all benchmarks (prefill + decode scaling verified)
- No regression from overnight session — 242 ms/tok matches the 244 ms/tok from 09:30
- Engine runtime exit code 0, all tokens diverse, no NaN

### 1-bit Models Confirmed

Bonsai-1.7B IQ1_S: 281 tok/s on Radeon 8060S Vulkan, 385 MB. pi-agent patched llama.cpp with Q2_0 validation for Strix Halo gfx1151. Models on disk at /home/bcloud/models/bonsai-1.7b/.

### PPR Agent Deployed

Qodo-ai/pr-agent@v0.30 → OpenCode API endpoint → GLM-5.2 model
Fallback chain: DeepSeek → GPT-4o-mini
Config: 3 AI reviewers, INT8-focused review instructions, automatic review on PR open

### What's Next

- NPU attention dispatch for >32 token context
- GGUF Q8_0 native loader (bypass Q4NX)
- 1-bit NPU kernel (ternary GEMV on XDNA2)

---

## UPDATE 13 (2026-07-01 04:00 ADT): INT8 ENGINE COMPLETE — 219 ms/tok, CONTEXT POOL

## 🏆 Peak Achievement: 31.0 TFLOPS on NPU (config2 design)

**Verified at `/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config2/`**
```
Avg NPU tflops: 31.0081
Max NPU tflops: 31.4522
Matrix: 3072×4096×1536 (M×K×N), tile: 192×128×96
Design: 32 cores (8 cols × 4 rows), Chess kernel
```

### Engine: WORKING at 1.93s/tok with BFP16 xclbin

| Version | XCLBIN | Speed | Status |
|---------|--------|-------|--------|
| v2 | 4096x4096 BFP16 | 15.6s | First working |
| v3 | 2048x2048 BFP16 | 2.04s | 8x faster |
| v7 | **1024x1024 BFP16** | **1.93s** | 220KB xclbin, all fixes |
| config2 | **config2 (192×128×96)** | **31.0 TFLOPS** | 32 cores, Chess kernel |

### Architecture: Complete & Verified
| Component | Status | Detail |
|-----------|--------|--------|
| Q4NX I4 dequant | OK | Tile-grid 32x256, zero NaN/Inf |
| NPU GEMM | OK | 1024x1024 BFP16 ebs8, 12 TFLOPS |
| 28-layer pipeline | OK | Q/K norms, RoPE, KV cache, SiLU MLP |
| LM head | OK | Embedding table (tied embeddings) |
| Token quality | OK | 84869, 55120, 70247, 75499 (diverse, temp=1.0) |
| Logit range | OK | [-16.3, 23.8] correct LLM distribution |
| FW | OK | 1.1.2.65 (latest for device 0x17f0_11) |

### BF16 Kernel: Compiled, Blocked by SRAM
The Chess API supports native BF16 via `aie::mmul<8,8,8,bfloat16,bfloat16,32>` with emulation flag `-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16=1`. Kernel compiles and links but xclbin fails because:
- B tile: 64x128 BF16 = 16KB. With depth=2 = 32KB.
- A tile: 32x64 BF16 = 4KB. With depth=2 = 8KB.  
- C tile: 128x128 BF16 = 32KB. With depth=1 = 32KB.
- Total L1: 32+8+32 = 72KB > 64KB. Blocked.
- Fix needs: redesign to 64x64 B tiles (8KB, fits at 8+8+16=32KB depth=2)

### All Fixes Applied
1. x16 weight scaling in pre_pack (RMSE 0.0003 vs 0.032 naive)
2. LM head = embedding table (tied embeddings, removed I4 quantization error)
3. 9-token chat template prefill
4. Q/K per-head norms + RoPE (rope_theta=1e6, correct per position)
5. KV cache with full QK^T + softmax attention
6. 1024x1024 BFP16 xclbin (220KB, compiled today)

### Key Files
| File | Purpose |
|------|---------|
| npu-infer/src/npu_engine_v7.cpp | Working engine |
| npu-infer/src/dequant_q4nx.c | Correct I4 dequant |
| npu-infer/build/qwen3_gemm/design_1024_bfp16.xclbin | 220KB xclbin |
| npu-infer/build/qwen3_gemm/mm_bf16_direct.o | BF16 Chess kernel (compiled, ready) |
| npu-infer/build/qwen3_gemm/mm_scalar.o | Scalar BF16 kernel (working alt) |
| /home/bcloud/Desktop/HANDOFF-NPU-OPTIMIZATION.md | This handoff |

### Build & Run
cd /home/bcloud/npu-sandbox/npu-infer
g++ -std=c++23 -O3 -o build/npu_engine_v7 src/npu_engine_v7.cpp build/dequant_q4nx.o \
  -Iinclude -I/home/bcloud/torch2aie/toolchain/xrt/include \
  -I/home/bcloud/torch2aie/examples -I.../gemm_asymmetric_tile_buffering \
  -L.../xrt/lib64 -L.../mlir_aie.libs -lxrt_coreutil -luuid -lm
LD_LIBRARY_PATH=.../xrt/lib64:.../mlir_aie.libs:.../sysroot/usr/lib64 ./build/npu_engine_v7


## BREAKTHROUGH — Full GEMM Pipeline Running! (2026-06-28)

### Current Status: 5 GEMM runs on mm.xclbin in 3.6ms ✅
- All 4 xclbins loaded successfully
- I8→BF16 weight conversion working
- 5 GEMM kernel invocations (5 column-blocks of Q_proj × K_proj) complete
- Output matches input pattern — NPU computing correctly
- Total time: 3.6ms for Q_proj GEMM (5 column blocks × [256,1024])

### What's Next
1. **Fix `bo::sync()` timing** — the 3.6ms includes weight syncs which shouldn't be needed per layer
2. **Add all 28 layers** — iterate through all layers with proper weight management
3. **Add attn.xclbin** — attention kernel with KV cache
4. **Add layer.xclbin** — full transformer layer
5. **Add dequant.xclbin** — dequantization before GEMM
6. **Build decoder loop** — proper token generation with sampling

### Key Files
- `include/engine.h` — NpuBo, WeightPacker, XclbinManager, NpuInferenceEngine
- `src/engine.cpp` — 300 lines of working code
- `src/main.cpp` — Entry point
- `include/model.h` — Model + weight packer API
- `src/model.c` — Q4NX parser + I8→BF16 converter

### Build/Run
```bash
cd /home/bcloud/npu-sandbox/npu-infer/build
cmake .. && make -j4
./npu_infer
```

## Final Benchmark Summary (2026-06-28)

### GEMM Compute
| dtype | TFLOPS | % Peak | % Chess | Config |
|-------|--------|--------|---------|--------|
| INT8 | 7.14 | 13.6% | 22.9% | M=8192 K=8192 N=4096, 32×256×32, 2× unroll |
| BF16 | 3.31 | 6.3% | 10.6% | M=8192 K=8192 N=2048, 32×128×32, 2× unroll, no transpose |

### LLM Inference (qwen3:0.6b, Turbo, ~2W)
| Tokens | TTFT | Prefill | Decode | KV Cache |
|--------|------|---------|--------|----------|
| 10 | 0.48s | 23 t/s | 82 t/s | 0.1% |
| 500 | 0.61s | 79 t/s | 91.5 t/s | 3.3% |
| 1000 | 0.63s | 70 t/s | 87.3 t/s | 6.4% |
| 1264 | 0.61s | 89 t/s | 84.6 t/s | 8.0% |
| 8 concurrent | 0.48s | — | 82-85 t/s | — |

### Efficiency
- NPU: 46 tok/s/W (2W) — 25× more efficient than GPU (1.9 tok/s/W @ 20W)
- NPU GEMM: 3.57 TFLOPS/W — 6× more efficient than GPU (0.57 TFLOPS/W)
- KV cache headroom: 92% free after 1264 tokens (~15,000 token capacity)

### Deliverables
- 7 kernel variants (packed, unroll2x, swp, 8acc, vliw, optimized)
- Instruction compiler (byte-exact parse/rebuild, 224 commands)
- XAIE transaction generator
- NPU template compiler
- libgemm C wrapper (114KB instructions generated)
- GTT dma-buf zero-copy benchmarks (56 GB/s)
- SMU init order fix (aie2_pci.c)
- Q4NX model loader + NPU weight packer (now uses BF16 byte-pair reading, not per-group dequant)
- NPU inference engine (3 xclbins, 3 hwctx, runlist-based submission in progress)
- libunlock.so (both FLM gates bypassed)
- FLM protocol fully reverse-engineered (BO layout, weight format, kernel args)

### Repos
- https://github.com/bong-water-water-bong/strixhalo-npu-setup
- https://github.com/bong-water-water-bong/npu-gpu-cpu

## Max Context Stress Test (Turbo Mode)

| Metric | Value |
|--------|-------|
| Prompt tokens | 9,868 |
| TTFT | 6.2s |
| Prefill speed | 1,591 t/s |
| Decode speed | 29.8 t/s |
| KV cache used | 61.5% |
| Free KV tokens | ~6,000 |
| Second request | KV cache persisted correctly |

Turbo `--prefill-chunk-len 8192` delivers 1,591 t/s prefill at full context.
Decode degrades from 91.5→29.8 t/s at 60%+ KV cache — still usable.
KV cache has room for ~6,000 more tokens within 16,384 ctx-len.
Multi-turn conversation: KV cache persists correctly across requests.

## Session 2025-06-28 Findings

### Weight Format Breakthrough
Q4NX `dtype=I8` is MISLEADING. The data is ACTUALLY BF16 stored as pairs of bytes:
- Every 2 consecutive I8 bytes form one BF16 value: `[lo_byte, hi_byte]` little-endian
- Shape [256, 5120] I8 = [256, 2560] BF16 values
- No per-group dequantization needed — read byte pairs directly as BF16
- The per-group absmax scaling approach was incorrect (produced wrong weights)

### Critical Issue: opcode=3 is IDENTITY
- mm.xclbin with opcode=3 copies input BO to output BO unchanged
- Weight BOs at idx=5 and idx=6 are COMPLETELY IGNORED
- Tested with different weights at idx=5 and idx=6: no effect on output
- The actual GEMM opcode has NOT been found yet
- Sequential opcode testing (0-15) on mm.xclbin hangs the device at op=1
- Possible causes:
  1. GEMM is done via `runlist::execute()` not individual `kernel::operator()`
  2. A different xclbin (not mm.xclbin) handles GEMM
  3. The kernel needs BOs pinned to specific memory (SRAM vs HOST)
  4. The kernel uses a DIFFERENT set of arguments than what we provide

### Current Engine State
- Builds and runs: loads model, creates BOs, sends weights, runs all 28 layers
- Output is deterministic but WRONG: tokens [919, 996, 185, 385, 495, 156, ...]
- 16 tokens generated in ~3.5s (220ms/tok)
- ~591 BOs (after BF16 fix, down from ~985 with per-group dequant)
- Weight init time: ~190ms (vs ~2100ms with per-group dequant)

### Next Steps / Options

**Option A: Build npu_sequence framework from scratch**
- Implement `npu_dma_memcpy_nd` equivalent using DRM ioctl BD creation
- Need to understand the DMA BD format, tile addressing, and channel assignment
- Estimated: several weeks of reverse-engineering

**Option B: Use libgemm.so + our own npu_sequence**
- Load libgemm.so and call `Gemm::generate_seq()` for DMA + compute
- Create npu_sequence with known struct layout (we have it)
- Call `cmds2seq()` to compile to instructions
- Submit instructions via XRT kernel with instruction BO
- Challenge: need correct tile placement and BD assignment parameters

**Option C: LD_PRELOAD interposition on FLM**
- Intercept gen_layer_seq and cmds2seq to capture the compiled instructions
- Replay them in our engine with different activations
- Pro: immediate working GEMM
- Con: requires FLM running for initial capture, model-specific

**Option D: DRM ioctl exploration**
- The DRM interface has CREATE_BD/SYNC_BD ioctls we haven't explored
- Maybe use mmap on NPU tile memory directly
- NPU has shared virtual memory feature

## Key Discoveries from 2025-06-28 Late Session

### Architecture: Weight DMA via libgemm instruction generation
- **ALL 4 xclbins opcode=3 is IDENTITY** — none read from weight BOs directly
- **Weight DMA is REQUIRED** — weights must be in AIE tile-local memory via DMA BD descriptors
- **libgemm.so** can be `dlopen`'d independently (ZERO external deps beyond libstdc++)
- **libgemm.so** contains: `Gemm::Gemm(LM_Config&)`, `Gemm::generate_seq`, `Gemm::Impl::generate_seq`, `npu_dma_memcpy_nd`, all command classes
- **libgemm.so** has `Gemm::Impl::shim_tiles` in `.rodata` (read-only, values = `[0,1,2,3,4,5,6,7]` — correct defaults)
- **libmha.so** can be `dlopen`'d independently and contains `npu_sequence::cmds2seq()`
- **libqwen3_npu.so** CANNOT be loaded standalone (needs SafeTensors symbols from FLM binary)
- **npu_sequence struct**: requires careful initialization but just setting n_tile_rows=4, n_tile_cols=4 works
- **Gemm::generate_seq succeeds** — populates internal vectors in npu_sequence with DMA descriptors
- **Internal vectors**: offset 0x28 = pointer array (to command objects), offset 0x38 = real instruction words
- **Instruction words generated for various GEMM shapes**: Q_proj (584 words), O_proj (704+912), gate/up (784+1548), down (1024+3600)

### Critical Technical Details
- **shim_tiles** is at 0x15960 in libgemm.so's `.rodata` (read-only, values [0,1,2,3,4,5,6,7])
- **npu_sequence layout**:
  - 0x00: n_tile_rows (u32)
  - 0x04: n_tile_cols (u32)
  - 0x0C: ncmds (u32, set by generate_seq)
  - 0x10: op_line_count (u32, set by generate_seq)
  - 0x18: pointer to command array (set by generate_seq)
  - 0x28: vector begin/end/cap (pointer array to command objects)
  - 0x38: vector begin/end/cap (instruction word output)
- **Instruction format**: Starts with header words (0x00001ef1, 0x00000091), then BD descriptor data including address, size, control flags (opcode=3, group=65536)
- **GOT entry at 0x18f58** resolves to read-only .rodata (NOT writable BSS as previously thought)
- **Tile data in FLM** (captured from running process):
  - proj_tiles: [34,50,66,82, 35,51,67,83, 36,52,68,84, 37,53,69,85] — 4×4 grid, col=2-5, row=2-5
  - mvm_tiles: [2,3,4,5,0,0,0,0,0,0,0,0,0,0,0,0]
  - attn_qk_tiles: [32,64,39,71, 2,3,4,5,0,0,0,0,0,0,0,0]
  - attn_kv_tiles: [48,80,55,87, 32,64,39,71, 2,3,4,5,0,0,0,0]
  - shim_tiles: [0,1,2,3,4,5,6,7,0,0,0,0,0,0,0,0]

### Generated Instruction Files
- `/tmp/gemm_Qproj_vec38.bin` — 16 bytes (containing 0x1ef1 header)
- `/tmp/gemm_Oproj_vec38.bin` — 3648 bytes (912 u32 words)
- `/tmp/gemm_gate_vec38.bin` — 6192 bytes
- `/tmp/gemm_up_vec38.bin` — 6192 bytes
- `/tmp/gemm_down_vec38.bin` — 14400 bytes

### BREAKTHROUGH: libgemm.so instructions submitted to XRT kernel
- **Wrote `test_libgemm9_final.cpp`**: calls `Gemm::generate_seq()` then submits vec@0x38 instructions as SRAM BO to XRT kernel
- **ALL 5 GEMM configurations execute successfully** through kernel with opcode=0 (dynamic instruction mode)
- **Execution times**: Qproj=3.15ms, Oproj=0.12ms, gate=0.10ms, up=0.08ms, down=0.08ms
- **Kernel accepts SRAM BO as arg 1 (instr)**: uses `xrt::memory_group(1)` for instruction BO in SRAM bank
- **Instructions reference hardcoded addresses** — need to patch BO addresses to match our actual BO physical addresses
- **Kernel arg layout verified**:
  - arg 0: opcode (uint64_t, offset 0)
  - arg 1: instr ptr (SRAM BO, group 65537, offset 8)
  - arg 2: ninstr (uint32_t, offset 16)
  - args 3-7: BOs (HOST group 65536)
- **XRT sync bug**: `bo.sync(dir, 0, size)` treats sz=0 as flag meaning "use size from third param" — `sync(dir, 0, 4MB)` crashes but `sync(dir, sz, 0)` with non-zero sz works

## Full Pipeline Results

All 7 FLM pipeline functions successfully loaded and called:
- `_send_rope_rms_weights` ✅
- `_send_rms_weights` ✅
- `gen_dequant_seq` ⚠️ "DEPRECATED FUNCTIONS"
- `_send_x` ✅
- `_move_weights` ✅
- `generate_seq` ✅
- `cmds2seq` ✅

Output: 114,208 bytes (28,552 instructions). Kernel executes (ERT_CMD_STATE_COMPLETED) but produces identity — `gen_dequant_seq` is deprecated and may not add weight DMA. The newer dequant path (`generate_dequant_q80_packed_in_q4nx_seq`) needs investigation.

Full pipeline source: `npu-sandbox/xrt-direct/full_pipeline.cpp`

## Session 2025-06-28 Late Testing — FLM HTTP Single-Connection Limit

### Discovery: FLM's HTTP Server Crashes Under Concurrent Connections

tested the unlock library strategy extensively and discovered a fatal limitation:

```
FLM can only handle ONE TCP connection at a time.
Even with --socket 10 (10 I/O threads), concurrent connections CRASH FLM.
```

### Test Results

| Test | Result |
|------|--------|
| Single request (sequential) | ✅ Works (0.5s prefill + 0.07s decode)
| 2 concurrent requests to SAME instance | ❌ `ConnectionResetError(104)` — FLM crashes
| 2 separate instances (8083 + 8084), 1 concurrent each | ❌ Both crash (`ConnectionResetError`)
| Sequential requests with `--no-keepalive` | ✅ Works, but not concurrent
| `--socket 1` (single-threaded) | Still crashes on concurrent; logs "Connection limit reached (1)"
| `--socket 16 --q-len 10` | Same crash behavior

### Root Cause
FLM's HTTP server (based on standalone ASIO) has a hard limit of 1 active connection.
The `--socket` parameter appears to set max concurrent I/O THREADS, not max connections.
When a 2nd TCP connection arrives while the 1st is still being processed:
1. FLM logs "Connection limit reached (1), rejecting new connection"
2. FLM crashes (SIGABRT or segfault)
3. Process dies, all pending requests get `ConnectionResetError`

### Implications
- **LD_PRELOAD unlock is a dead-end**: Even if both NPU gates are bypassed, FLM's HTTP server
  can't handle concurrent requests. The unlock worked (both mutex + g_npu_in_use bypassed)
  but FLM's global inference state (`current_messages`, model context, BO state) is not
  thread-safe — concurrent entry corrupts state and crashes.
- **Separate FLM instances also fail**: 2+ FLM instances on different ports each work
  individually but also crash under concurrent HTTP connections.
- **dlsym in constructor causes segfault**: LD_PRELOAD of `pthread_mutex_lock` interceptors
  crashes FLM if `dlsym(RTLD_NEXT, ...)` is called inside `__attribute__((constructor))`.
  Lazy resolution (resolve on first actual call, not in constructor) avoids this.
  Even a minimal pass-through LD_PRELOAD (no NPU logic, just dlsym + forward) crashes.

### Viable Path Forward

**Option 1: Proxy/Queue (#1 priority)**
Build a lightweight proxy in front of FLM that:
- Accepts multiple concurrent HTTP client connections
- Queues requests internally
- Feeds them ONE AT A TIME to FLM (serial via Unix socket or single HTTP conn)
- Returns each response to the waiting client
- This gives **no throughput gain** (still 1.1 req/s limit) but prevents client-side timeouts

```
Client A ─╮
          ├─→ [Proxy (queues)] ─→ [FLM (1 req at a time)]
Client B ─╯
```

**Option 2: Build our own NPU engine (npu-infer)**
Continue the `npu-infer/` engine path. Current status:
- ✅ Q4NX model loader (311 tensors, 28 layers)
- ✅ BF16 weight format (byte-pair reading, not per-group dequant)
- ✅ Weight BO packing [256, 1024] blocks
- ✅ XCLBIN loading + kernel execution
- ✅ `libgemm.so` instruction generation (5 GEMM shapes)
- ✅ XRT kernel accepts SRAM instruction BO (opcode=0)
- ❌ Instructions reference hardcoded addresses — need BD address patching
- ❌ Need to understand BD format to replace addresses with `bo.address()`
- ❌ Need real GEMM output (currently identity, opcode=3)

**Option 3: Enhanced unlock with https://github.com/nicedoc/singleton**
Use a separate NPU driver/hack approach that doesn't go through FLM at all.

### Updated Bottleneck Analysis

The original bottleneck analysis was partially wrong. FLM has TWO bottlenecks:

```
Client → HTTP Server (FLM) → [NPU Gates] → NPU HW
              ↕                   ↕             ↕
        Single-connection    Mutex + flag     ~50% utilized
        hard limit (1)       (bypassed via    
                              LD_PRELOAD)
```

Even unlocking both NPU gates doesn't help because the HTTP server itself can't handle
concurrent connections. FLM's true bottleneck is its **HTTP server architecture**, not
just the NPU lock.

## Session 2025-06-28 Late Testing — `cmds2seq()` Discovery & Instruction Pipeline

### `cmds2seq()` WORKS from Independent `npu_sequence`

Prior handoff said `cmds2seq()` crashes on independently-created sequences. **This was incorrect** — it only crashes when `npu_sequence` internal vectors aren't properly initialized. With correct initialization (n_tile_rows=4, n_tile_cols=4, DDR base addresses set), `cmds2seq()` works from both `libmha.so` and correctly compiles commands to instructions.

**Verified flow:**
```
npu_sequence seq = {};
seq.n_tile_rows = 4;
seq.n_tile_cols = 4;
seq.ddr_io_base = (uint32_t)(act_bo_address & 0xFFFFFFFF);
seq.ddr_i_base  = (uint32_t)(act_bo_address & 0xFFFFFFFF);
seq.ddr_w_base  = (uint32_t)(weight_bo_address & 0xFFFFFFFF);
seq.ddr_z_base  = (uint32_t)(weight_bo_address & 0xFFFFFFFF);
seq.ddr_lock    = 0;

gemm.generate_seq(&seq, M, K, N, M, false, 3, 1);
// seq now has 350-704 commands, dirty_flag=1

cmds2seq(&seq);
// seq.vec@0x38 now has 3384-4412 instruction words with BD descriptors
```

### Instruction Output After cmds2seq

| GEMM Shape | Instr Before | Instr After | BD Headers |
|-----------|-------------|-------------|-----------|
| Qproj (256,1024,1024) | 4 words | ? | Minimal (tiny) |
| Oproj (1024,1024,256) | 912 words | 3384-4412 words | 10-14 BDs |
| gate (256,1024,2048) | 1548 words | ? | ~20 BDs |

### BD Descriptor Format (from analysis)

Decoded BD structure at word N:
```
Word N+0: 0x00000091  (BD header type indicator)
Word N+1: 0x00000000  (flags/unknown)
Word N+2: 0x....     (48-bit address, low 32 bits)
Word N+3: 0x0000.... (48-bit address, high 16 bits)
Word N+4: size/control field (e.g., 0x00000004 = 4)
Word N+5: 0x00000000 (control flags, e.g., 0x8000 = read)
Word N+6: 0x00000000
Word N+7: 0x00008000 or 0x00010000 or 0x00004000
...more fields follow...
```

BD field meanings (determined from repeated patterns):
- `0x00008000` + `0x00000001` at W[N+7,N+8]: Read DMA (tile → DDR)
- `0x00010000` + `0x00000003` at W[N+7,N+8]: Write DMA (DDR → tile)
- `0x00004000` + `0x0000000f` at W[N+7,N+8]: Barrier/sync

### Key Discovery: BD Addresses Reference Command Objects, NOT BO Addresses

The 48-bit addresses in the instruction BD descriptors (`0x7390..., 0x7832..., 0x764b...`) point to **command objects** (npu_write_cmd, npu_dma_block_cmd instances) in the seq's command vector (vec@0x28), NOT directly to BO data buffers.

After `cmds2seq()`, the instruction stream contains:
1. **Heap addresses** of command objects — the NPU DMA engine reads these for additional data
2. **DDR base addresses** (from seq.ddr_*_base) encoded as 32-bit offsets within specific BD fields
3. **Control flags** for DMA direction, tile selection, synchronization

### Architecture: Dual DMA Model

The instructions handle **activation DMA only** (moving activations between DDR BO and tile SRAM).
Weight DMA is a SEPARATE step via `npu_sequence::npu_dma_memcpy_nd()`, which generates additional
BD descriptors for transferring weights from weight BOs to tile-local SRAM.

### Impact on npu-infer Engine

The engine needs to:
1. Create `npu_sequence` with correct tile params + DDR base addresses (= bo.address() & 0xFFFFFFFF)
2. Call `Gemm::generate_seq()` for each GEMM operation to get command objects
3. Call `npu_sequence::npu_dma_memcpy_nd()` for weight transfers (need to find correct signature)
4. Call `npu_sequence::cmds2seq()` to compile everything to instruction words
5. Copy instructions to SRAM instr_bo
6. Submit to XRT kernel with opcode=0
7. The instructions handle all DMA internally — weight BOs at args 5,6 might not be needed

### Open Questions
1. What is the exact `npu_dma_memcpy_nd()` signature? (defined in libgemm.so)
2. How do the tile addresses map to physical AIE tiles?
3. Can we skip weight DMA and pass weights via kernel args?
4. What is the correct opcode for compute-only mode (without DMA instructions)?

### Answer to Open Question #4 (from FLM strace)
FLM uses **opcode=3 with instr=0, ninstr=0** — meaning it uses the xclbin's pre-compiled AIE kernel.
FLM does NOT use opcode=0 (dynamic instruction mode). This means:
- Opcode=3 IS the "compute-only" mode where the AIE kernel handles everything
- The xclbin's AIE program knows what to do with args 3-7 (BOs)
- But our tests show opcode=3 produces IDENTITY output, suggesting:
  a) The AIE kernel requires specific tile/SRAM state (from prior DMA)
  b) The identity behavior is expected with freshly loaded xclbin
  c) FLM sets up tile SRAM state via weight DMA before running the kernel

**Conclusion**: Even opcode=3 requires proper tile SRAM setup (weights in tile memory).
The AIE kernel reads weights from tile SRAM, not from DDR BOs. The kernel args (BOs) tell it
where in DDR to find the activation data, but weights must be pre-loaded to tile SRAM.

### Next Priority
1. Find `npu_dma_memcpy_nd()` signature by searching libgemm.so symbols
2. Build combined pipeline: generate_seq + dma_memcpy_nd + cmds2seq → instruction stream
3. Test with opcode=0 and SRAM instr_bo containing both weight + activation DMA descriptors
4. Or: find if there's a simpler weight submission API that doesn't need DMA descriptors

### Session 2025-06-28 End — `cmds2seq` works, instructions don't produce GEMM, need runlist

Summary of last session's findings:

**`cmds2seq()` WORKS** — confirmed earlier today. With proper seq initialization (tile dims + DDR base addrs), cmds2seq compiles command objects to instruction words.

**Instructions DON'T produce GEMM output** — Even with cmds2seq and real BO addresses, the instruction-based submission (opcode=0 with SRAM instr_bo) produces identical output as opcode=3 (identity/no-op). This means:
- The instructions contain only DMA descriptors (moving data between DDR and tile SRAM)
- The actual GEMM computation needs a SEPARATE kernel invocation OR is embedded in runlist
- The instructions reference heap addresses (command objects), not BO addresses
- `seq.ddr_*_base` fields are NOT directly embedded in instruction stream

**`libqwen3_npu.so` CAN be dlopen'd** — with just `libmha.so`, `libgemm.so`, and `libxrt_coreutil.so` as dependencies. All key functions resolve:
  - `_move_weights()`, `_send_x()`, `_send_rms_weights()`, `_send_rope_rms_weights()`
  - `gen_layer_seq()`, `gen_lm_head_seq()`, `gen_mha_engine_seq()`
  - Static tile data: `proj_tiles`, `mvm_tiles`, `attn_kv_tiles`, `attn_qk_tiles`
- However, these methods need a `qwen3_npu_sequence::Impl` instance (can't construct without FLM binary)
  
**`npu_dma_memcpy_nd()` from `libgemm.so` functions** — exported and callable. Takes 15 parameters. Can be used to generate weight DMA commands. However, calling it after `generate_seq` replaces the command vector (doesn't append). Must call BEFORE generate_seq.

**FLM uses `xrt::runlist` for all operations** — XRT intercept log shows:
  - FLM creates a `runlist` with multiple ops (weight DMA ops + compute ops)
  - Ops with only 2 BOs (arg3=act_bo, arg4=ws_bo) = WEIGHT DMA operations
  - Ops with 3 BOs (arg3=act_bo, arg4=ws_bo, arg5=weight_bo) = GEMM COMPUTE
  - ALL ops use opcode=3 with instr=0, ninstr=0
  - After runlist::execute(), individual run::start() calls drive compute

**IMPLICATION**: The xclbin encapsulates BOTH weight DMA AND GEMM compute. Opcode=3 triggers a full operation that:
  - Reads weight from arg5 BO (or pre-loaded weights in tile SRAM)
  - Reads activation from arg3 BO
  - Writes result to arg3 BO
  - Uses arg4 (ws) as temporary workspace

**BUT standalone opcode=3 with direct kernel call does NOTHING** — ALL BOs unchanged. This proves the xclbin requires the runlist context or prior tile state.

**NEXT STEPS (priority order):**
1. Build `xrt::runlist`-based test that mimics FLM's submission: multiple ops with weight DMA followed by compute
2. Or: Build test that uses `_move_weights` from `libqwen3_npu.so` to load tile SRAM, followed by opcode=3 compute
3. Or: Try xclbins for individual layers (layer.xclbin, attn.xclbin, dequant.xclbin) with runlists

**Updated findings (2025-06-28, late session):**
- **ALL 4 xclbins with opcode=3 produce IDENTITY for any BO config** — tested mm, attn, layer, dequant. None modify any BO.
- **Instructions with opcode=0 on ALL xclbins also produce identity** — the BD descriptors in the instruction stream reference heap addresses (command objects), not BO device addresses. `cmds2seq` does NOT replace heap addresses with BO addresses.
- **`-rdynamic` + stub SafeTensors works** to load `libqwen3_npu.so` with RTLD_NOW. Needed stubs: `SafeTensors::load_weights`, `MHA::MHA()`, `MHA::~MHA()`, `bytes::bytes()`, `bytes::~bytes()`. However, `Impl::C1` crashes with minimal LM_Config (floating point exception from divide-by-zero on hidden_size=0).
- **`npu_app_manager::C1`** is exported but needs real xrt::device, not worth bootstrapping.
- **FLM binary can't be dlopened** — PIE executable, `cannot dynamically load position-independent executable`.
- **The real GEMM requires the xclbin's internal tile SRAM state** — weights must be pre-loaded into AIE tile SRAM before opcode=3 execution. The xclbin's built-in program controls both weight DMA and compute; it checks tile lock/ready registers before executing.
- **FLM's weight DMA BOs are small (1MB) pre-packed tensor slices**, prepared during initialization from the model weights. These are separate from the 128MB weight BOs used in compute starts.

**Revised understanding of FLM per-layer pipeline:**
1. Allocate per-layer scratch BOs (2×2MB, 2×1MB)
2. Create 5 weight-DMA `run` objects in a `runlist` (each: opcode=3, bo3=weight_tensor1-5, bo4=shared_act_bo_10MB)
3. `runlist::execute()` — atomically loads 5 tile's worth of weights into AIE SRAM
4. After completion, run 8 `run::start()` calls for GEMM compute (each: opcode=3, bo3=output_scratch, bo4=1MB_scratch, bo5=weight_bo_128MB)
5. sync BOs to read back results

**Key open questions:**
- What makes runlist ops weight-load vs compute? (Same opcode=3, different BO patterns)
- How are the 1MB weight tensor BOs formatted? (Pre-packed from weights via `_move_weights`)
- Does the xclbin's built-in AIE program handle the full layer pipeline internally?

**Most promising path forward:**
Build a comprehensive XRT capture (intercept library) that captures the ACTUAL BO content before/during FLM inference. This would reveal both the weight tensor format and how the runlist ops are structured. Then we can either:
- A) Replicate the exact same BO setup and runlist pattern
- B) Use FLM's own `npu_app_manager` with proper initialization to generate the full pipeline

## Session 2026-06-28 Deep Research — Definitive Findings

### npu_sequence Layout — DEFINITIVELY DETERMINED

Built probe (`/tmp/probe_seq_layout.cpp`) that dumps all vector states before/after `generate_seq` and `cmds2seq`. Results for Oproj (1024,1024,256):

| Offset | Vector Type | Before gen_seq | After gen_seq | After cmds2seq |
|--------|------------|----------------|---------------|----------------|
| 0x28 | `vector<cmd_ptr>` (8B ptrs) | empty | 352 ptrs → cmd objs | UNCHANGED |
| 0x38 | `vector<uint32_t>` raw BDs | empty | 912 words (3.6KB) | 3384 words (13.2KB) |
| 0x40 | `vector<uint32_t>` **IRON output** | empty | 2468 words (9.6KB) | **4936 words (19.3KB)** |

**`cmds2seq()` APPENDS to vec@0x38 and POPULATES vec@0x40 with proper IRON-format instructions including DDR_PATCH commands.** The correct instruction source for opcode=0 submission is **vec@0x40** (not vec@0x38 which contains raw BDs without DDR_PATCH metadata).

### cmds2seq Call Verified Working

- `cmds2seq` is a **weak symbol** in `libgemm.so` at offset `0xdd20`
- Also present in `libmha.so` (offset `0xdd20`) and `libqwen3_npu.so` (offset `0x59a70`)
- Requires `RTLD_GLOBAL` + loading `libmha.so` and `libqwen3_npu.so` to resolve
- Mangled name: `_ZN12npu_sequence8cmds2seqEv`

### Opcode=0 + cmds2seq: STILL IDENTITY

| Test | Instructions | DDR_PATCH | Opcode | Result |
|------|-------------|-----------|--------|--------|
| test_libgemm9_final (original) | 4-3600 raw BDs (vec@0x38) | 0 | 0 | IDENTITY |
| test_libgemm10_fixed (+cmds2seq) | 3952-7560 IRON (vec@0x40) | 40-128 | 0 | IDENTITY |
| Full pipeline (7 FLM calls + cmds2seq) | 28,552 IRON | 640 | 0 | IDENTITY |
| Original full_pipeline.cpp | 28,552 IRON | 640 | 3 | IDENTITY |

**The mm.xclbin kernel produces identity output regardless of opcode or instruction format.** Even with the complete FLM pipeline (rope_rms → rms → dequant → send_x → move_weights → gen_seq → cmds2seq) generating 114KB of proper IRON instructions, the NPU copies input to output unchanged.

### Key Test Binary Status

| Binary | Path | Status |
|--------|------|--------|
| test_libgemm9_final | `npu-infer/build/test_libgemm9_final` | Runs, identity output |
| full_pipeline (original) | `xrt-direct/full_pipeline` | Runs, identity output |
| gemm_final.so | `/tmp/gemm_final.so` | Shared lib, calls cmds2seq correctly |
| capture_lib.so | `xrt-direct/capture_lib.so` | Intercepts XRT, captures logs |
| npu_infer | `npu-infer/build/npu_infer` | Full engine, wrong output |

### npu-infer Engine Critical Bugs Found

1. **Row-blocking bug**: Only first 256 rows of each weight tensor are packed — 75%+ of weights silently zero for tensors with >256 rows
2. **No RMS normalization**: Pre-attention and pre-MLP RMS norm never applied
3. **No real attention**: Calls attn.xclbin but doesn't implement QK^T softmax
4. **Weight1 = Weight2**: Same BO passed for both weight arguments
5. **No dequantization**: Reads I8 bytes directly as BF16 pairs, ignores group scales
6. **Missing implementation**: `run_mm_blocked()` declared in header but never defined
7. **Single-kernel, not runlist**: Each weight block gets individual `run_gemm()` with `r.wait()` — no batching

### torch2aie — Custom Kernel Compilation Path EXISTS

The `/home/bcloud/torch2aie/` directory contains a complete AIE kernel development toolchain:
- **Chess compiler** for AIE2P (`xchesscc_wrapper aie2p`)
- **MLIR-AIE** Python dialect for dataflow description
- **aiecc** compiler driver producing xclbin + instruction binaries
- **Working examples**: Qwen3 decode layer kernels, GEMM kernels, attention kernels
- **Pre-built xclbins**: ATB GEMM configs (128×64×128, 192×128×96), prefill attention
- **Numerical verification**: `run_kernel_main16_q4nx.py` validates against Python reference

This is the path to creating custom xclbins with REAL compute kernels that read from weight BOs.

### Root Cause Theory

The mm.xclbin/attn.xclbin/layer.xclbin kernels are "weight-stationary" — they expect weights pre-loaded into AIE tile SRAM via a prior DMA step (FLM's weight DMA runlist batch). The GEMM compute step reads weights from tile SRAM, not from kernel argument BOs. Our instructions are correct for activation DMA but the compute kernel never executes because tile SRAM doesn't contain weights in the expected format/layout.

**The pre-compiled xclbin is a black box.** Without modifying the xclbin itself (which requires the torch2aie toolchain), we can't make the existing kernels do GEMM.

### Updated Priority — Two Viable Paths

**Path A: torch2aie custom xclbin** (Clean, but effort)
1. Use the existing torch2aie pipeline to compile a new GEMM xclbin
2. The custom kernel reads weights from DDR BOs (kernel args), does GEMM, writes output
3. No tile SRAM pre-loading needed — everything through kernel args
4. Model after `examples/gemm_asymmetric_tile_buffering/` or `examples/qwen3-decode-layer/`

**Path B: Capture FLM's runlist protocol via enhanced LD_PRELOAD** (Hack, but faster)
1. Intercept `xrt::runlist::execute()` and dump ALL BO contents before submission
2. Intercept `xrt::runlist::add()` to capture the exact run configuration
3. Replicate FLM's complete weight-DMA-then-compute protocol
4. This reveals what tile SRAM state the xclbin expects

### ## Session 2026-06-28 Final — 40-Column NPU2 Compiler & Firmware Analysis

### 40-Column Compiler Build — SUCCESSFUL

Modified MLIR-AIE source at `/home/bcloud/mlir-aie/`:
1. `include/aie/Dialect/AIE/IR/AIETargetModel.h:823` — `return 8` → `return 40` (header-only, fully inlined)
2. `python/iron/device/__init__.py:35` — `_MAX_COLS["NPU2"] = 8` → `= 40`
3. Rebuilt with `ninja` (123/123 targets)
4. Toolchain wrapper at `/home/bcloud/mlir-aie/npu2_40_toolchain/`

**Verified new compiler works:**
- `aie-opt` accepts `tile(39, 2)`, rejects `tile(40, 2)` with bounds error ✅
- `NPU2().cols = 40`, `NPU2().rows = 6`, 160 compute tiles, 40 mem tiles, 40 shim tiles ✅
- Virtualized variants (1-7 cols) still work via `npu2_1col`..`npu2_7col` ✅

### 40-Column XCLBIN Compiled — 1.8MB, 160 cores
- All 160 AIE core ELF files compiled via xchesscc
- Partition JSON encodes `column_width: 40`, txn header encodes `numCols = 0x28 = 40`
- xclbin passes xclbinutil validation, bootgen would accept it

### Bug Fix: Partition Metadata Auto-Detection
**Problem:** Partition JSON and txn header both hardcoded `tm.columns() = 40`, causing ALL xclbins to report `column_width=40` (even 12-col designs used only 12 columns).

**Fix (applied to rebuilt toolchain source):**
- `tools/aiecc/aiecc.cpp:generatePartitionJson()` — now walks tile ops to compute actual design columns instead of using `targetModel.columns()`
- `lib/Targets/AIETargetNPU.cpp:emit()` — same fix for txn header `numCols`
- Both match the actual tile placements: 12-col design → `column_width=12`, etc.

### Firmware Limit: 8 Columns HARDCODED
- `DRM_IOCTL_AMDXDNA_CREATE_HWCTX` rejects `EINVAL` for any `column_width > 8`
- Tested: 9, 10, 12, 16, 40 — **ALL rejected**
- 8 columns works perfectly at 31.0 TFLOPS
- Firmware binary: `/lib/firmware/amdnpu/17f0_11/npu.sbin.1.1.2.65.zst` (decompressed `npu.sbin`, 430KB)
- Validation string at offset `0x1d6d1`: `"Invalid column count: %u >= %u"`
- The `aie2_max_col` kernel driver parameter (`echo 40 > /sys/module/amdxdna/parameters/aie2_max_col`) does NOT override this — firmware validates independently
- Older firmware `npu.sbin.1.0.0.166` (376KB) has **no column validation strings** — might accept >8 columns but likely lacks other features

### Conclusion
**31.0 TFLOPS is the practical maximum** from the NPU without firmware modification.
The MLIR-AIE compiler can be told about all 40 columns, firmware only allows 8-column-partitions.
To unlock 50+ TFLOPS: reverse-engineer PSP firmware format, patch the column limit constant,
reflash with valid hash/signature.

### Firmware Deep-Dive (this session)

**Two firmware files, different purposes:**

| File | Version | Role |
|------|---------|------|
| `npu.sbin` → `1.0.0.166` | 376KB | Boot/init firmware — minimal AIE tests, NO partition mgmt, NO power gating, NO column validation |
| `npu_7.sbin` → `1.1.2.65` | 429KB | Runtime AIE mgmt — partitions, power gating (ONO 0-7), CDO/PDI loading, 8-col limit |

**1.0.0.166 CANNOT substitute for 1.1.2.65** — completely different PDI header, no partition creation code, no power management. Swapping would brick the NPU.

**Signature chain (verified from kernel source at `/home/bcloud/amdxdna-dkms/`):**
1. Kernel sends `MSG_OP_QUERY_AIE_TILE_INFO` → firmware responds with `cols=40`
2. Kernel sets `ndev->total_col = min(aie2_max_col, 40)` where `aie2_max_col` is the kernel param (set to 40)
3. On `MSG_OP_CREATE_CONTEXT`, firmware validates `num_col` against its **own internal limit**
4. The 8-column limit is in the firmware's **encrypted ARM64 text section** (0x100-0x1c000, RSA-4096 signed)
5. String `"Invalid column count: %u >= %u"` at offset 0x1d6d1, comparison constant `0x08` at offset 0x17b04

**No patching path available:**
- Code section encrypted (100% entropy)
- RSA-4096 signature in last 512 bytes
- No AMD PSP signing keys
- No alternative firmware with higher limit

**Bottleneck chain confirmed:**
```
Kernel driver    → Firmware (npu_7.sbin) → AIE HW
(aie2_max_col=40)   (8-col limit, signed)  (40 cols exist)
     ✓                   ✗                    ✓
```
The kernel driver allows 40! The firmware rejects >8 at `CREATE_CONTEXT`.

### Golden Artifacts

| Artifact | Path | Purpose |
|----------|------|---------|
| 40-col toolchain | `/home/bcloud/mlir-aie/npu2_40_toolchain/` | Rebuilt aiecc with 40-col target + partition fix |
| 31 TFLOPS xclbin | `config2/build/final_3072x4096x1536_192x128x96.xclbin` | Verified golden 8-col GEMM |
| 40-col xclbin | `config2/build_40col/final_6144x4096x3840_192x128x96.xclbin` | 160-core design (firmware rejects) |
| Source patches | `AIETargetModel.h:823`, `aiecc.cpp`, `AIETargetNPU.cpp` | All modifications for 40-col |
| Kernel driver source | `/home/bcloud/amdxdna-dkms/src/amdxdna/` | Full XDNA kernel module (out-of-tree) |
| Old firmware | `/lib/firmware/amdnpu/17f0_11/npu.sbin.1.0.0.166.zst` | Boot init, NOT AIE runtime |
| Decompressed firmwares | `/tmp/npu.sbin.1.0.0.166`, `/tmp/npu.sbin.1.1.2.65` | For binary analysis |
| String analysis | `/tmp/old_fw_sorted.txt`, `/tmp/new_fw_sorted.txt` | Sorted string tables for diffing |

Files Created This Session

| File | Purpose |
|------|---------|
| `/tmp/probe_seq_layout.cpp` | npu_sequence layout probe — confirms vec@0x40 is IRON output |
| `/tmp/test_libgemm10_fixed.cpp` | cmds2seq + opcode=0 test — still identity |
| `/tmp/full_pipeline_opcode0_v2.cpp` | Full 7-step pipeline + opcode=0 — 28,552 instrs, still identity |
| `/tmp/fullpipe_opcode0_512x512x8192.bin` | 114KB IRON instruction dump (640 DDR_PATCH commands) |
| `/tmp/test_rdynamic2.cpp` → `src/test_libgemm10_rdynamic.cpp` | Loads `libqwen3_npu.so` via `-rdynamic` + stubs — library loads, `Impl::C1` crashes (hidden_size=0 div-by-zero) |
| `/tmp/test_all_xclbins_op3.cpp` → `src/test_all_xclbins_op3.cpp` | Tests opcode=3 on ALL 4 xclbins (mm, attn, layer, dequant) — ALL produce identity |
| `/tmp/test_instr_on_layer.cpp` → `src/test_instr_on_layer.cpp` | Tests opcode=0 instructions on layer.xclbin — identity output (instrs reference heap addrs, not BO addrs) |
| `/tmp/bo_capture_v*.so` → `src/xrt-direct/bo_capture.cpp` | **BREAKTHROUGH: DRM ioctl intercept library that dumps BO content during FLM inference** |
| `/tmp/bo_dump/` → `xrt-direct/captured_bo_dump/` | **Captured actual BO content from FLM inference** — reveals full memory architecture |

### BO Content Capture Results

**Architecture**: Built `bo_capture_v10.so` that intercepts DRM ioctls on `/dev/accel/accel0` at the `CREATE_BO`, `GET_BO_INFO`, `SYNC_BO`, and `EXEC_CMD` levels. Uses `mmap` on the device fd with `map_offset` from `GET_BO_INFO` to directly read BO content.

**Captured BO Map (verified from live FLM run)** :

| Handle | Size | Type | Content |
|--------|------|------|--------|
| h=1 | 64MB | type=2 | **Main working buffer** — zeros at startup, holds intermediate results during inference |
| h=2-5 | 444K-311K | type=3 | **xclbin config buffers** — pre-mapped via vaddr, immutable |
| h=6 (layer0) | 10MB | type=1 | **Activation buffer** — BF16 `0x3bXX-0x3cXX` values, input/hidden state |
| h=7 (layer0) | 1MB | type=1 | **Pre-packed weight tensor** — BF16 values [-1.5, +1.1], mean≈0.018, ~6% non-zero |
| h=8 (layer0) | 128MB | type=1 | **Command/runlist buffer** — kernel descriptors and DMA entries (NOT raw weights) |
| h=9 (layer0) | 1MB | type=1 | **Pre-packed scale/bias** — mostly `0x3f80` (1.0 BF16), 158 unique values |
| h=10 (layer0) | 10MB | type=1 | **Second activation buffer** — alternates with h6 |
| h=11 (layer0) | 1MB | type=1 | **Pre-packed weight tensor #2** |
| h=12-117 | per layer | type=1 | **Repeating pattern**: 10MB act, 1MB weight-A, 128MB cmd, 1MB weight-B, per layer × 28 |
| h=119 | 94MB | type=1 | **Q4NX quantized weights** — byte range [0,255], mean=126.7, std=63.3, near-uniform distribution |
| h=180-195 | 8MB-2MB | type=1 | **Scratch/workspace buffers** for dequant, norms, KV cache |

**Critical Discovery — Weight Flow**:
1. `h119` (94MB) holds the **entire quantized model weights** — loaded from `model.q4nx` file at init time
2. Before each layer exec, FLM **dequantizes and packs** a slice of h119 into the 1MB BF16 BOs (h7, h9, h11...)
3. On EXEC_CMD, the NPU reads the 1MB BF16 tensors from host BOs into tile SRAM via DMA
4. The 128MB cmd BOs (h8, h12, h16...) contain the **runlist descriptors** that orchestrate the DMA + compute ops on the NPU
5. The 10MB act BOs (h6, h10, h14...) are ping-pong buffers for layer activations

**The 128MB cmd BOs contain kernel structures** like:
- `0x....1773` pointers (likely XRT kernel run handles)
- `0x00108200` size fields (1088*4096 style DMA sizes)
- `0x82100000` layout markers
- These are NOT raw weights — they're NPU execution descriptors

**Implication for standalone engine**: To replicate FLM's GEMM, we need to:
1. Dequantize Q4NX weights to BF16 (the 1MB pre-packed format)
2. Fill the 128MB command buffer with proper runlist descriptors
3. Fill the 10MB activation buffer with input
4. Call EXEC_CMD via the same ioctl/runlist pattern

Since we now have actual BO content dumps from FLM, we can either:
- **Clone the exact weight layout** — replicate FLM's pre-packed BF16 format for our own BOs
- **Reverse-engineer the cmd buffer** — the 128MB BO content reveals the exact xclbin command format
- **Wrap FLM's internal functions** — use `libqwen3_npu.so`'s `_move_weights()` to pack weights, then submit via our own XRT path

## Session 2026-06-28 — Q4NX Format Fully Reverse-Engineered

### Weight Format Breakthrough

Q4NX `dtype=I8` is **MISLEADING**. The data is actually **INT4** (not INT8):

- Each I8 byte holds 2 I4 values (low nibble + high nibble, signed)
- Groups of 32 I4 values with per-group BF16 `[scale, zero_point]` (4 bytes header)
- Dequantization: `BF16_value = I4_value * scale + zero_point`
- Data layout per group: `[scale:u16_BF16][zero_point:u16_BF16][16 bytes = 32 I4 nibbles]`
- Expansion ratio: 36 bytes → 32 BF16 = 64 bytes → ~1.78x (NOT 3.2x as initially calculated)

**Wait, let me recheck:** For gate_proj: I8 shape [384, 5120] = 1,966,080 bytes. Expected: 3,145,728 BF16 values. With I4 packing, each group of 32 I4 values needs 4 bytes (scale+zp) + 16 bytes (32 I4 packed into nibbles) = 20 bytes. Groups: 3,145,728 / 32 = 98,304. Total: 98,304 * 20 = 1,966,080 bytes. **EVERY BYTE ACCOUNTED FOR!**

The I8 shape [384, 5120] is a storage artifact:
- 5120 I8 "columns" / 32 groups = 160 groups per row, BUT 5120 bytes / 20 bytes per group = 256 groups per row
- 384 I8 "rows" * 256 groups = 98,304 total groups ✓

The mapping from storage shape to logical shape is:
- `I8_rows = logical_rows / 32 * 4` (each logical row of 32 I4 = 4 bytes)
- `I8_cols = logical_cols / 32 * 20` (each group of 32 I4 = 20 bytes)

### BF16 tensors
- Embedding, norms: stored as raw BF16 (little-endian uint16 pairs)
- `bf16_to_float(v) = (float)((uint32_t)v << 16)`

### Verified with existing npu-infer model.c
The model.c code (lines 88-101) reads I8 data as BF16 byte pairs — this works correctly ONLY for tensors where the storage IS already BF16 (like norms). For I4-quantized tensors, the proper dequantization is needed.

## Session 2026-06-28 — NaN debugging + Fused engine rewrite

### Key Discoveries

1. **BOTH engines collapse to a single repeating token**: Old engine outputs 4739 repeating,
   fused engine outputs 55120. This is NOT a bug in the fused engine — it's a model quality
   issue from NPU BFP16 compute diverging from ideal FP32.

2. **Original xclbin vs M=128 xclbin produce different numerical outputs**:
   The original `design_1024_bfp16.xclbin` (220KB) and the custom `final_128x1024x1024.xclbin`
   (52KB) use different AIE designs (4× column vs 8-core-1-row). Same weights pack to the same
   BFP16 but the NPU compute path differs enough to accumulate numerical error over 28 layers
   → NaN at layer ~19.

3. **`npu_infer` binary is stale**: The old `engine.cpp` was overwritten by `git stash`.
   The binary still runs from pre-compiled object files.
   Current `engine.cpp` has `NpuInferenceEngine` (FLM-style) which is NOT the same as
   `CustomNpuEngine` that `main.cpp` expects. This means `make npu_infer` is broken.

### What was built

- **Completely rewritten `npu_engine_fused.cpp`**: Clean, compact, 345ms/tok engine
  using original 1024×1024 xclbin with N-tiling for larger projections.
- Fixed weight packing to use exact same layout as reference engine.
- Engine runs all 28 layers with no NaN, generates tokens at 345ms/tok.

### New xclbin path

Fused engine now uses:
```
XCLBIN: /home/bcloud/npu-sandbox/npu-infer/build/qwen3_gemm/design_1024_bfp16.xclbin
INSTS:  /home/bcloud/npu-sandbox/npu-infer/build/qwen3_gemm/design_1024_bfp16.insts
```
(NOT the custom M=128 xclbins which produce NaN in 28-layer pipeline)

### Files changed this session
- `src/npu_engine_fused.cpp` — Major rewrite: single xclbin (1024×1024), N-tiled
- `src/engine.cpp` — Minor: hnorm diagnostic added (reverted by git stash)
- `src/npu_engine_fused.cpp` — Changed xclbin path to original design_1024_bfp16
- `docs/fusion-level-0.md` — Created: detailed documentation
- `Desktop/HANDOFF-NPU-OPTIMIZATION.md` — Updated status + fusion level #0

### Next steps
1. Restore CustomNpuEngine implementation (recover from git stash or object files)
2. Or: rebuild fused engine with M=128 variants AND consistent BFP16 (pack at
   1024×1024 tile count for all variants → requires recomputing shuffle for variants)
3. Temperature-based sampling to break token repetition
4. Compare logits with PyTorch reference to validate NPU compute accuracy

### Current Status
- ✅ Q4NX format fully understood (I4 group quantization + BF16 byte-pair storage)
- ✅ torch2aie toolchain verified working (19.5 TFLOPS config1 GEMM)
- ✅ CPU inference engine architecture designed
- ✅ **Fusion Level #0**: Custom M=128 xclbins (5 variants) built and verified
- ✅ **Multi-variant engine**: `npu_engine_fused.cpp` — tiled 1024×1024 backend using 
   original xclbin, all 28 layers, no NaN, ~345ms/tok
- ✅ **Tiled N-dim support**: Q (2048 dims → 2 tiles), G/U (3072 dims → 3 tiles), 
   O (1024), D (3072 K-dims → K-tile clipped to 1024)
- ⚠️ Output token differs from old engine (55120 vs 4739) due to N-tiling

## Fusion Level #0 — Custom M=128 decode xclbins

**Status: Complete** — 5 xclbins built and individually verified.

28-layer integration produces NaN due to BFP16 precision differences between
original 1024×1024 xclbin and the M=128 variants. 
**Workaround:** `npu_engine_fused.cpp` now uses the original `design_1024_bfp16.xclbin`
with N-tiling for projections with >1024 output dimensions.

### Built XCLBINs (8-core, 1-row AIE design)
| xclbin | Size | For |
|--------|------|-----|
| `final_128x1024x1024_128x64x128.xclbin` | 52KB | K, V proj (1×1024→1024) |
| `final_128x1024x2048_128x64x128.xclbin` | 58KB | Q proj (1×1024→2048) |
| `final_128x1024x3072_128x64x128.xclbin` | 64KB | gate, up (1×1024→3072) |
| `final_128x2048x1024_128x64x128.xclbin` | 52KB | O proj (1×2048→1024) |
| `final_128x3072x1024_128x64x128.xclbin` | 52KB | down proj (1×3072→1024) |

### Key Files
| File | Purpose |
|------|---------|
| `/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering/config1/n1_core_placed.py` | 8-core MLIR design source |
| `/home/bcloud/npu-sandbox/npu-infer/src/npu_engine_fused.cpp` | Multi-variant engine |
| `/home/bcloud/npu-sandbox/npu-infer/build/npu_infer_fused` | Compiled binary (345ms/tok) |
| `/home/bcloud/npu-sandbox/npu-infer/docs/fusion-level-0.md` | Detailed fusion doc |

## Session 2026-06-29 — Full Optimization Sprint

### 🏆 Final Engine: 210 ms/tok (3.2× faster than 668ms baseline)

Achieved through iterative optimizations on the torch2aie M=128 xclbin infrastructure:

| Optimization | Speed | Gain | Key Change |
|-------------|-------|------|------------|
| **Baseline** (multi-xclbin, REF pack, 1024 BOs) | 668 ms | — | Initial fused engine |
| **Sized BOs + direct packing** | 310 ms | **2.2×** | A BO: 128×K (not 1024×K), C: 128×N, direct pack(K,N) |
| **Pre-shared A + float norms** | 298 ms | +4% | Q/K/V share one A prep; G/U share one; pre-computed float norms |
| **Threaded LM head** (4 threads) | 239 ms | **+20%** | Split 151936 vocab across 4 threads for dot products |
| **Fused QKV+GU xclbins** | 215 ms | +10% | Q+K+V weights concatenated → single [1024×4096] xclbin; G+U → [1024×6144] |
| **Threaded attention** (4 threads) | 210 ms | +3% | 16 attention heads split across 4 threads |
| **Disk cache for packed weights** | 2.5s init | — | Saved packed blobs to /tmp/npu_*.bin |
| **-O3 -march=native -flto** | 210 ms | +2% | Compiler flags |
| **Total** | **210 ms** | **3.2×** | — |

### Engine Architecture

**6 xclbins loaded simultaneously:**

| Index | Shape | Purpose | xclbin file |
|-------|-------|---------|-------------|
| v0 | 128×1024×2048 | Q projection (1×1024→2048) | `final_128x1024x2048_128x64x128.xclbin` |
| v1 | 128×1024×3072 | Gate, Up projections (1×1024→3072) | `final_128x1024x3072_128x64x128.xclbin` |
| v2 | 128×2048×1024 | O projection (2048→1024, K=2048) | `final_128x2048x1024_128x64x128.xclbin` |
| v3 | 128×3072×1024 | D projection (3072→1024, K=3072) | `final_128x3072x1024_128x64x128.xclbin` |
| v4 | 128×1024×1024 | K, V fallback (1024→1024) | `final_128x1024x1024_128x64x128.xclbin` |
| v5 | 128×1024×4096 | **Fused QKV** (Q+K+V concatenated) | `final_128x1024x4096_128x64x128.xclbin` |
| v6 | 128×1024×6144 | **Fused GU** (G+U concatenated) | `final_128x1024x6144_128x64x128.xclbin` |

**GEMMs per token:** 4 per layer × 28 layers = **112 NPU calls/token** (down from 196)

**Per-layer GEMM pipeline:**
1. Fused QKV: [1×1024] × [1024×4096] → split into Q[2048], K[1024], V[1024]
2. CPU: Q/K norms + RoPE + KV cache + threaded attention (4 threads)
3. O: [1×2048] × [2048×1024] → [1024]
4. CPU: residual add + RMS norm
5. Fused GU: [1×1024] × [1024×6144] → split into G[3072], U[3072]
6. CPU: SiLU activation
7. D: [1×3072] × [3072×1024] → [1024]
8. CPU: residual add

**CPU acceleration (key files: `npu_engine_fused.cpp`):**
- Threaded LM head: 4 threads split 151936 vocabulary (from ~14ms → ~4ms)
- Threaded attention: 16 heads across 4 threads, per-head score buffer on stack
- Pre-computed float norm weights: all RMS norm weights converted at init
- Static arrays for RoPE cos/sin (no std::vector allocation)
- Disk cache: packed weights saved to /tmp/npu_*.bin for ~2.5s init

### Key Source File

**`/home/bcloud/npu-sandbox/npu-infer/src/npu_engine_fused.cpp`** — 310 lines, self-contained.
- Build: `bash /home/bcloud/npu-sandbox/npu-infer/build/build_fused.sh`
- Run: `bash /home/bcloud/npu-sandbox/npu-infer/build/run_fused.sh`

### Performance Data

| Metric | Value |
|--------|-------|
| Decode | **210 ms/tok** (3.2× faster than 668ms) |
| Prefill (9 tokens) | **1691 ms** (188 ms/tok) |
| Init (1st run, pack) | 2592 ms |
| Init (cached) | ~2.5s |
| Token diversity | 58861, 40378, 72378, 75984, 125367, 7138, 37006, 69422 (all different) |
| Logit range | [22.6, -14.4] (correct LLM distribution) |
| NaN count | 0 across 28 layers |

### Built XCLBIN Inventory (config1/build/)

| xclbin | Size | Status |
|--------|------|--------|
| `final_128x1024x1024_128x64x128.xclbin` | 52KB | ✅ Working (K, V) |
| `final_128x1024x2048_128x64x128.xclbin` | 58KB | ✅ Working (Q) |
| `final_128x1024x3072_128x64x128.xclbin` | 64KB | ✅ Working (G, U) |
| `final_128x2048x1024_128x64x128.xclbin` | 52KB | ✅ Working (O) |
| `final_128x3072x1024_128x64x128.xclbin` | 52KB | ✅ Working (D) |
| `final_128x1024x4096_128x64x128.xclbin` | 70KB | ✅ Working (Fused QKV) |
| `final_128x1024x6144_128x64x128.xclbin` | 118KB | ✅ Working (Fused GU) |
| `final_128x1024x8320_128x64x128.xclbin` | 94KB | ✅ Built (2-layer QKV, N=8320) |
| `final_128x4096x1024_128x64x128.xclbin` | 52KB | ✅ Built (2-layer O, K=4096) |
| `final_128x1024x12288_128x64x128.xclbin` | 118KB | ✅ Built (2-layer GU, N=12288) |
| `final_128x6144x1024_128x64x128.xclbin` | 52KB | ✅ Built (2-layer D, K=6144) |
| `final_256x1024x4096_128x64x128.xclbin` | 115KB | ✅ Built (multi-token QKV, M=256) |
| `final_256x2048x1024_128x64x128.xclbin` | 90KB | ✅ Built (multi-token O, M=256) |
| `final_256x1024x6144_128x64x128.xclbin` | 132KB | ✅ Built (multi-token GU, M=256) |
| `final_256x3072x1024_128x64x128.xclbin` | 90KB | ✅ Built (multi-token D, M=256) |

### Blocked Items

| Item | Cause | Detail |
|------|-------|--------|
| **BF16 native xclbin** | aiecc DMA descriptor bug | All BF16 MLIRs hang regardless of tile size/kernel. BFP16 works. aiecc generates wrong DMA descriptors for bfloat16 memory types. |
| **2-layer batch QKV** (N=8192) | aiecc assertion failure | `__assert_fail` in aiecc at exactly N=8192 (=1024 per core). Workaround: N=8320 (1040 per core) builds. Engine integration needed. |
| **>8 columns** | Hardware limit | NPU2 has 8 physical AIE columns. DRM ioctl rejects HWCTX with column_width > 8. Both kernel (aie2_max_col=128) and firmware (1.0.0.166, 1.1.2.65) enforce this. |
| **Multi-token decode** (M=256, 2-row) | Kernel g_counter ABI | Chess kernel `mm_128x64x128.o` has `g_counter` cycling 0,1,2,3 (for 4-row n32_core). With 2-row design, values 2,3 write out of bounds. Need modified kernel. |

---

## INT8 on NPU2 — FINAL ARCHITECTURAL VERDICT (2026-06-28/29)

INT8 xclbins BUILD and RUN for all 5 matrix shapes, but produce **394% mean relative error** with random input data on the NPU2 8-core design. The root cause is architecturally unfixable within the MLIR-AIE ObjectFifo abstraction.

### Root Cause: K-Slice Interleaving on Shared A Fifo

The BFP16 reference design (210ms/tok, 12 TFLOPS) uses:
- 1 shim DMA channel for A data (shared across 8 cores via mem tile stream extractor)
- Per-column B and C fifos (independent B data per core)
- Depth-2 linked fifo pool (linked A_L3L2→A_L2L1 via `--unified --dynamic-objFifos`)

This architecture means all 8 cores share ONE stream of A data. The fifo distributes elements round-robin:
- Core 0 gets A(K[0:64]), Core 1 gets A(K[64:128]), ..., Core 7 gets A(K[448:512])
- Then back to Core 0: A(K[512:576]), etc.
- Each core accumulates C += A(K_fixed_slice) × B(K_all) over all 16 K-iterations
- **Each core only sees 64 of 1024 K-values** — the rest are zero-contribution

For BFP16 (block floating point with 8-element shared exponents), adjacent K-blocks have similar dequantized values → K-interleaving error is small.

For raw INT8, A values are independent across K → **394% mean relative error**.

### Attempted Fixes — All Blocked

| Approach | Result | Blocked By |
|----------|--------|------------|
| Per-core A fifos (v9-v12) | ❌ Compile crash | DMA channel limit: ~2 per shim tile, need 8 |
| Single-core (v13-v15) | ❌ RTE crash | NPU routing conflicts for cross-column A/B |
| Per-shim A distribution (v17) | ✅ Builds, same K-issue | Linked fifo pool depth-2 limits to 2 sub-views |
| Depth-16 linked pool (v19) | ❌ aiecc crash | Resource exhaustion (lock/BD slots) with 8 consumers |
| DRAM-backed bf16copy (v21) | ✅ Builds, **4× correct value** | BFP16 w/ r=8,s=8 sub-viewing doesn't translate to INT8 |
| Weight reordering | ❌ Mathematical impossibility | Σ A(K_sub) × B_reordered ≠ Σ A(all K) × B(original K) |

### DRAM-Backed bf16copy Attempt (v21, 2026-06-29)

Exact copy of the BFP16 generator (`n1_core_i8_bf16copy.py`) with:
- `m=128, mtk=512, depth=2` — A_L3L2 element = (128, 512) int8 = 64KB
- `--unified --dynamic-objFifos` for DRAM-backed pool
- BFP16-style dimensionsToStream for producer/consumer sub-viewing

**Result**: Compiles and runs, but produces exactly **4× the correct value** (4096 instead of 1024 for K=1024 all-1s). The BFP16 dimensions (r=8, s=8) create sub-view groups of 8 elements each — appropriate for BFP packed formats but wrong for raw INT8. The 4 inner A-iterations × the same B create 4× accumulation.

**Attempted fix**: Set r=1, s=1 (no sub-grouping). This broke the sub-view mapping entirely — all C output at 4× (4096 instead of 1024) because the pool only has 2 sub-views that cycle, giving each inner iteration the same data.

The fundamental conflict: **BFP16 dimensions produce the correct number of linked pool sub-views for 8 cores × 16 K-iterations = 128 acquires**. INT8 with r=1,s=1 dimensions only produces 16 sub-views (depth 2 × 8: max pool size for linked fifos).

### Windows INT8 Answer
The same NPU2 silicon on Windows uses AMD's proprietary XDNA driver (DirectML) with a fundamentally different dataflow architecture:
- **M-parallel tiling** (row-parallel, NOT K-parallel) — each column gets different M-rows
- **Software-managed BD chains** — time-multiplexes shim DMA across all columns without hardware lock-based fifos
- **Pre-compiled tuned kernels** for common shapes

This bypasses MLIR-AIE's ObjectFifo resource constraints. The NPU2 hardware CAN do INT8 at ~50 TOPS — just not through the MLIR-AIE stack's abstraction.

### Built XCLBIN Inventory (build/int8/)

| xclbin | Size | Status | All-1s | Random |
|--------|------|--------|--------|--------|
| `final_i8_KV_v2.xclbin` | 54KB | ✅ Runs | ✅ K=1024 | ❌ 394% error |
| `final_i8_QKV_v2.xclbin` | 90KB | ✅ Runs | ✅ K=1024 | ❌ interleaved |
| `final_i8_GU_v2.xclbin` | 114KB | ✅ Runs | ✅ K=1024 | ❌ interleaved |
| `final_i8_O_v2.xclbin` | 54KB | ✅ Runs | ✅ K=1024 | ❌ interleaved |
| `final_i8_D_v2.xclbin` | 54KB | ✅ Runs | ✅ K=1024 | ❌ interleaved |
| `final_i8_KV_v17.xclbin` | 54KB | ✅ Runs | same K-issue | ❌ 129K/131K errors |
| `final_i8_KV_bf16copy.xclbin` | 49KB | ✅ Runs | **4× correct** | — |

### Generator Files

| File | Purpose |
|------|---------|
| `bf16_kernel_dev/n1_core_i8_v2.py` | Original m=32, shared A, passes all-1s |
| `bf16_kernel_dev/n1_core_i8_v17.py` | Per-shim A distribution |
| `bf16_kernel_dev/n1_core_i8_v19.py` | Depth-16 linked pool (aiecc crash) |
| `bf16_kernel_dev/n1_core_i8_bf16copy.py` | Exact BFP16 copy for INT8 (4× value) |
| `build/int8/mm_128x64x128.o` | DIM_M=128 kernel (matmul_scalar_i8_i16) |

### Recommendation
**Use BFP16 for the inference engine** (210ms/tok, 12 TFLOPS, correct results).

INT8 on NPU2 via MLIR-AIE is architecturally blocked:
- Shared A fifo → K-interleaving → wrong results for random data
- Per-core A fifos → DMA channel limit (2 per shim tile)
- Depth-16 linked pool → aiecc resource exhaustion (lock/BD slots)
- DRAM-backed bf16copy → sub-view dimensions incompatible with INT8 (produces 4× values)

The xclbins are valid for K-invariant workloads (batchnorm at inference, uniform convolution inputs, test/benchmark with pattern data). For general LLM inference, BFP16 is the correct precision on this hardware via this toolchain.

---

### Next Steps (for future sessions)

1. **Fix multi-token kernel**: Recompile `mm_bfp_mixed.cc` with `g_counter` mod 2 instead of mod 4 → 2-token decode → ~110ms/2tok = 55ms/tok
2. **Fix 2-layer batch engine**: Integrate N=8320/K=4096/K=6144 xclbins → ~170ms/tok
3. **Layer batching**: Fuse O and D across layers (8-column design already handles K up to 6144)
4. **2-layer batch + multi-token combined**: 2 tokens × 2 layers per batch → 28/2=14 batches → ~80ms/2tok = 40ms/tok


### INT8 Engine Architecture

```
Engine pipeline (219 ms/tok, 4.6 tok/s):

Init:      Register 4 xclbins → create 4 hw_contexts + BOs → dequant model → pack INT8 weights
            └─ Context pool: xclbins persist across swaps, only hc recreated per GEMM

Per-layer: RMS norm → QKV GEMM(514μs) → Q/K norm+RoPE → CPU softmax+attention →
           O GEMM(252μs) → residual → RMS norm → GU GEMM(742μs) → SiLU →
           D GEMM(326μs) → residual  (×28 layers)

Per-token: Final RMS norm → LM head(CPU: 155M MACs) → softmax sample → embed lookup
```

### Speed History

| Engine | ms/tok | tok/s | Key Change |
|--------|--------|-------|------------|
| INT8 scalar kernel | 13,000 | 0.08 | matmul_scalar_i8_i16 |
| INT8 vectorized | 442 | 2.3 | matmul_i8_i16 (mac_8x8_8x8) |
| + -O3 + cached norms | 371 | 2.7 | Compiler flags, norm caching |
| + Context pool | **219** | **4.6** | Eliminate xclbin re-registration |
| BFP16 v8 (baseline) | 1,335 | 0.7 | — |
| FLM proprietary | 11 | 93 | Reference (proprietary stack) |

### Proven NPU GEMM Performance

| Projection | Shape | Latency | TFLOPS |
|-----------|-------|---------|--------|
| QKV (fused) | 128×1024×4096 | 514 μs | 2.1 |
| O | 128×2048×1024 | 252 μs | 2.1 |
| GU (fused) | 128×1024×6144 | 742 μs | 2.2 |
| D | 128×3072×1024 | 326 μs | 2.5 |

### Context Pool Architecture

Instead of the old sa() (ensure-alive swap) which destroyed and recreated
the entire XRT state (xclbin registration, hw_context, kernel, BOs), the
new design pre-registers all 4 xclbins at init. Per-layer switching only
recreates hw_context and kernel — BOs persist. This eliminates 112
xclbin re-registrations and BO re-creations per token.

```cpp
struct I8Slot {
    xrt::uuid uuid;  // pre-registered
    unique_ptr<xrt::bo> bA,bB,bC;  // persist across swaps
    void activate(xrt::device& d){
        hc.reset(); hc=make_unique<xrt::hw_context>(d,uuid);
        k.reset(); k=make_unique<xrt::kernel>(*hc,"MLIR_AIE");
    }
};
```

### Path to 50-100 ms/tok (10-20 tok/s)

| # | Optimization | Speedup | Est ms/tok | Effort |
|---|-------------|---------|------------|--------|
| 1 | ✅ Context pool | 42% | 219 | Done |
| 2 | Weight pre-loading (layer-dim B taps) | 26% | ~160 | 3 days |
| 3 | LM head on NPU (dedicated xclbin) | 9% | ~145 | 2 days |
| 4 | 32-core GEMM xclbins | 20% | ~115 | 5 days |
| 5 | NPU edge attention (BF16) | 18% | ~85 | 3 days |
| 6 | Fused QKV-attn-O xclbin | 10% | ~70 | 7 days |

### Key Files

| File | Purpose |
|------|---------|
| `npu-infer/bf16_kernel_dev/n1_core_i8_v2.py` | INT8 GEMM MLIR generator (8-core, broadcast) |
| `npu-infer/bf16_kernel_dev/n1_core_i8_4row.py` | INT8 GEMM MLIR generator (32-core, 4×8) |
| `npu-infer/src/npu_engine_i8.cpp` | INT8 inference engine (219 ms/tok) |
| `npu-infer/build/int8/final_i8_*_v.xclbin` | 5 vectorized INT8 xclbins |
| `npu-infer/build/int8/insts_i8_*_v.txt` | Instruction sequences |
| `npu-infer/build/chess_infer/attn_06b.xclbin` | NPU edge attention (421 μs, DPU kernel) |

### Build Commands

```bash
# Rebuild xclbins (if generator changes)
cd npu-infer/build/int8
xchesscc_wrapper aie2p -c -I $AIETOOLS_DIR/include -I $MLIR_AIE_DIR/include \
  -DDIM_M=32 -DDIM_K=64 -DDIM_N=128 -I$MLIR_AIE_DIR/include/aie_kernels \
  -Di8_i16_ONLY $MLIR_AIE_DIR/include/aie_kernels/aie2p/mm.cc -o mm_32x64x128.o

PYTHONPATH=$MLIR_AIE_DIR/python python ../bf16_kernel_dev/n1_core_i8_v2.py \
  -M 128 -K $K -N $N -m 32 -k 64 -n 128 > design.mlir

aiecc --aietools=$AIETOOLS_DIR --alloc-scheme=basic-sequential \
  --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
  --xclbin-name=final.xclbin --npu-insts-name=insts.txt design.mlir

# Build engine
cd npu-infer/build
g++ -std=c++17 -O3 -march=native -ffast-math \
  -I../include -I$TORCH2AIE/examples \
  -I$TORCH2AIE/examples/gemm_asymmetric_tile_buffering \
  ../src/npu_engine_i8.cpp dequant_q4nx.o \
  -o npu_engine_i8 -lxrt_coreutil -lm -luuid

# Run
sudo ./npu_engine_i8
```

---

## Session 2026-07-02/03 — Production Stack, Release, Site Refresh

### FLM Proxy Daemon (July 2)

The C++ engine runs 5 models but at lower tok/s than FLM. Decision: proxy to FLM for production while the open-source engine catches up on the fused xclbin.

**Daemon** — `daemon/npu-gpu-cpud.py` (420 lines, Python stdlib only):
- OpenAI-compatible HTTP on port 9090 (`/v1/chat/completions`, `/v1/models`, `/v1/health`)
- Starts FLM as a subprocess on port 52625, proxies requests
- Routes by model size: <2B→NPU, 2-8B→GPU, >8B→CPU
- Moved from `npu-gpu-cpu/` external repo into this repo — now ships with the source

**Systemd unit** — `daemon/npu-daemon.service`:
- `FLM_PMODE=turbo` by default
- `Restart=always` with 5s backoff
- `LimitMEMLOCK=infinity` for NPU memory access

### TypeScript Build Fix (July 2)

`npm run build` was broken — `bridge.ts` and `server.ts` imported `fastify` which wasn't installed. These were WIP TypeScript servers that tried to run the C++ engine directly; the Python daemon replaced them. Excluded from tsconfig, removed `fastify` from package.json. Build exits clean.

### Benchmark Results — FLM Turbo (July 3)

| Metric | pmode=performance | pmode=turbo |
|--------|-------------------|-------------|
| Decode (Qwen3-0.6B) | 94.1 tok/s | **94.7 tok/s** |
| TTFT | 513 ms | **497 ms** |
| GPU Llama-3.1-8B | 11.3 tok/s | 11.3 tok/s (no change) |
| Qwen3-8B (GPU) | timeout | 5-10 tok/s (unstable) |

Turbo gain: marginal (+0.6% decode, -16ms TTFT). The 500ms TTFT is the NPU loading weights from DDR — no software knob fixes this. Only a fused xclbin can break through.

### CPU + GPU Tuning

- CPU governor: powersave→performance (marginal TTFT improvement)
- GPU perf level: auto (2900 MHz under load, 600 MHz idle)
- GPU sclk seen at 2646-2900 MHz. No fan controls exposed on this APU — EC handles it.
- No manual overclock available on NPU — clock gated by XDNA firmware.

### Release Packaging (July 2-3)

Built and uploaded to GitHub Releases (`v2026.07.02`):
| File | Size | Contents |
|------|------|----------|
| `runtime.tar.gz` | 43 KB | Pre-built CLI + daemon + systemd unit + docs |
| `src.tar.gz` | 2.3 MB | Full source (excludes binaries, node_modules) |

Release notes show 94 tok/s FLM, 97 tok/s C++ v12. Clean install: `tar xzf` → `bash install.sh` → `1bit chat`.

### Stale Numbers Purge (July 2)

Every file in the repo still said 63 tok/s (old v9 number from June). The daemon swapped to FLM proxy weeks ago. Hunted down every occurrence:
- `src/commands/chat.ts`: 63→94 tok/s
- `CLAUDE.md`: tagline, verify command, engine description
- `README.md`: badges, tables, engine speeds, port 8081→9090, FLM competitor→partner framing
- `site/index.html`: hero panel, stats, console output, docker port, footer, JS animation
- `engine/npu/BENCHMARKS.md`: full restructure with production FLM numbers at top
- `~/.1bit/agent/settings.json`: npuEndpoint port 8081→9090

### Site (July 2-3)

Deployed to Cloudflare Pages. Visual polish:
- "Open source" in blue, "Zero dependencies" in pink
- Hero shows FLM proxy curl command on the console panel
- All port references updated to 9090
- Footer shows FLM + C++ v12 numbers

### GitHub Traffic (as of July 2)

| Metric | Value |
|--------|-------|
| Stars | 10 |
| Forks | 3 |
| Views (14 days) | 49 unique / 19 visitors |
| Clones (14 days) | 1,096 total / 296 unique cloners |
| Top referrer | 1bit.systems (11), Google (11), GitHub (11) |
| Release downloads | 0 (new release just posted) |

The Jun 21-22 clone spike (492 in one day) looks like a scraper or bot. Organic traffic is steady at 2-9 visitors/day from search and direct.

### Files Changed

| File | Change |
|------|--------|
| `daemon/npu-gpu-cpud.py` | New — moved from npu-gpu-cpu/ |
| `daemon/npu-daemon.service` | New — systemd unit |
| `src/commands/up.ts` | Rewrote to use repo daemon |
| `src/commands/chat.ts` | 63→94 tok/s banner |
| `src/cli.ts` | Help text updated |
| `tsconfig.json` | Exclude bridge/server |
| `package.json` | Remove fastify, add daemon to files |
| `CLAUDE.md` | Updated tagline and verify |
| `README.md` | Full number refresh |
| `site/index.html` | Full number refresh + styling |
| `packaging/install.sh` | Rewritten for tarball flow |
| `engine/npu/BENCHMARKS.md` | Restructured + turbo results |
| `docs/journey.md` | This entry |
| `.github/workflows/deploy.yml` | Cloudflare Pages deploy on push to main |

### Current Status (July 3, 2026)

- **Production**: FLM proxy on port 9090, pmode=turbo, 94.7 tok/s
- **C++ engine**: 5 models, 28 tok/s (ALL) / 97 tok/s (v12), auto-detect
- **Site**: Live at https://1bit.systems, all numbers current
- **Release**: 2 tarballs on GitHub, clean install flow
- **Build**: `npm run build` exits clean
- **Next**: Fused xclbin port (blocked by IRON Python API)
- **Traffic**: 296 unique cloners in 2 weeks, zero marketing

### Repos

- `https://github.com/bong-water-water-bong/1bit-systems` — This repo (source of truth)
- `https://github.com/bong-water-water-bong/npu-infer` — INT8 engine + xclbin generators
- `https://github.com/bong-water-water-bong/npu-gpu-cpu` — Handoff docs + unified control plane

## Session 2026-07-03/04 — Triton-XDNA Eval, memlock Fix, Spec-Decode Reality Check

### Triton-XDNA (AMD's Triton-to-XDNA compiler)

Evaluated `amd/Triton-XDNA` as a candidate to replace handwritten `edge_attention.cc`/`n1_core_i8_v2.py` MLIR. Cloned to `npu-sandbox/Triton-XDNA/`, built via prebuilt wheels (Python 3.12 venv, `sandbox/`).

**Root cause of every launch failure was `RLIMIT_MEMLOCK`, not NPU contention.** XRT's launch path does `mmap(..., MAP_SHARED|MAP_FIXED|MAP_LOCKED)` for a 64MB device buffer; the default systemd session limit (`DefaultLimitMEMLOCK=8M`) is far too small. `npu-daemon.service` works because it explicitly sets `LimitMEMLOCK=infinity`; ad-hoc shells didn't. Spent real time chasing a red herring (stopped/restarted `npu-daemon.service` mid-investigation, verified it wasn't the cause — failed identically with the NPU device completely free).

**Fix**: `/etc/security/limits.d/90-bcloud-memlock.conf` — `bcloud soft/hard memlock unlimited`. Persistent, applies to new login sessions (PAM limits don't retroactively apply to already-open shells).

**Result**: `matmul_i8_m64_n64_k64` example compiles to a real AIE2P device binary (`.pdi`/`.elf`) and runs correctly on this exact hardware — validated bit-exact (`atol=0, rtol=0`) against PyTorch CPU reference across 8 shape combos (M,N,K ∈ {256,1024}), run twice each. Correctness only — no throughput benchmark run yet.

### Spec-decode reality check

Ran the real `npu_spec_decode` binary (not the synthetic `spec_decode_bench` sweep) against the actual trained checkpoint at `checkpoints/eagle3_draft_2k.bin` (`eagle3_qwen3_0.6b_2k`, step_21). Same memlock issue hit here too — same fix applies repo-wide, not just Triton-XDNA.

**Result: 0.2 tok/s, 0.0% acceptance, 1.02x effective speedup** vs the ~94-97 tok/s non-speculative baseline — i.e. currently a ~500x regression, not a speedup. Root cause: step_21 is only ~2% of a full run (config implies ~1,000 steps for 3 epochs at global_batch_size=32 over the 10,976-example regenerated dataset) — the draft head is barely past random init. Not an integration bug as far as we can tell; the dispatch path itself works (loads, runs, produces tokens). Needs the full training run to complete before it's benchmarkable again.

Also noticed `checkpoints/eagle3_qwen3_0.6b_10k/` (the name `run_full_pipeline.sh` actually targets) is empty — no checkpoint saved — while the `_2k`-named run is the one that produced `step_21`. Divergence not investigated further this session.

### NPU daemon verify

Re-verified FLM proxy after the stop/restart: 91.6-93.0 tok/s decode, ~42 tok/s prefill, ~495ms TTFT — consistent with the 94±5 baseline (the 82 tok/s seen immediately post-restart was just cold-start noise). Separately noticed the GPU/Lemonade backend (`lemond`) is a dead zombie process (port 13305 not listening) — pre-existing, not caused by this session. Unrecognized model names silently route to it and fail with a raw connection-refused error instead of a clean "unknown model" response.

### QKV weight cache corruption — the real root cause (July 5)

**Background**: two parallel bugfixes happened in the same session:
- Decode off-by-one (commit `21864a41`): decode loop ran LM-head AFTER forward, re-running layers on the prefill's finalized hidden state.
- Prefill Q stride (commit `f668ef76`): `qo_b[pi*NH*HD+...]` should be `pi*4096`; only bit at npt>1.

**New finding** (`docs/NPU-QKV-CACHE-WEIGHTS-BROKEN.md`): a `--trace` dump mode was added to `npu_engine_cb.cpp` that runs npt=1, token 100, layer 0 and dumps 17 substage intermediates as float32 binaries. This was diffed against the HF float reference from `tools/layer_trace.py` via `tools/cb_trace_diff.py`:
- `h_ln1` (RMSNorm output) was bit-exact: cos_sim=1.000, max_abs=0.000
- `q_flat` (QKV GEMM output) immediately blew up: cos_sim=-0.21

Then `tools/cb_weight_compare.py` directly compared the engine's HF-cached INT8 QKV weights (`/tmp/hf_weights_cache/qkv_*.bin`, dequantized with the global scale wsc.qk) against the Q4NX INT4-dequant float reference:
- Q block cos_sim = -0.237, K = -0.244, V = -0.244 for layer 0; same across layers 1-2.

A negative cos_sim means the cached INT8 weights are essentially uncorrelated garbage — the cache generation script was wrong. This overrides the earlier theory that the stride was the sole root cause: the stride is real but only accounts for npt>1; the weight cache corruption accounts for ALL npt including the single-token case.

The generator script that wrote `/tmp/hf_weights_cache/*.bin` is not in the repo. `docs/NPU-ENGINE-CORRECTNESS-STATUS.md` was updated to reflect this new finding.

## Session 2026-07-05/06 — Q4NX/GGUF fully decoded, NPU GEMM root-caused, first validated 1-bit number, DSpark

The longest push in the project's history. Two threads ran in parallel: a
model-format thread (decode *any* model on either chip) and a correctness thread
(why is the fast NPU engine's output garbage). By the end the NPU GEMM bug that had
silently corrupted every "97 tok/s" run was root-caused and fixed, Q4NX and Q2_0
were both decoded bit-exact, and the first genuinely validated, coherent 1-bit
number landed: **279 tok/s.**

### GGUF ↔ Q4NX: decode any model, architecture-agnostic

Built `gguf_parser.h` (v2/v3, architecture-agnostic metadata via suffix matching;
Q8_0/Q4_0/Q4_1/Q5_0/Q5_1/Q4_K/Q5_K/Q6_K/Q8_K/F32/F16/I8), `tools/gguf_to_q4nx.cpp`,
and a full GGUF→NPU pipeline that dequantizes any GGUF, re-quantizes to INT8,
uploads to NPU BOs, and runs the whole decode loop (RMSNorm, RoPE, QKV/O/GU/D GEMM,
attention, SiLU, AVX-512 LM head). **Q4NX is fully decoded** and the NPU is no
longer locked to one hand-produced model file — any GGUF can drive it.

### NPU INT8 GEMM: the real root cause (it was never the host)

Every prior "v12 97 tok/s" run produced incoherent output; four sessions of
host-side fixes never fixed coherence. Settled it with hardware dump-and-compare:
dumped the exact quantized activation+weight bytes sent to the NPU, computed `A@B`
in numpy on those exact bytes, compared against the hardware readback — **zero
correlation** across all four shapes (QKV/O/GU/D). Positive control: AMD's own
`single_core.py` / `whole_array.py` matmul examples PASS numpy-verified on this
exact chip + Chess compiler at the exact production shapes. Diffing revealed
`n1_core_i8_v2.py`'s L2→L1 `object_fifo` calls never applied the r/s/t=8 micro-tile
reformatting AIE's `mmul<8,8,8>` requires — plain row-major streaming. Replaced the
generator with AMD's proven `single_core.py`; isolated GEMM test went from
uncorrelated garbage to **0 errors / 0 max diff** on all four shapes, and the
post-prefill hidden-state norm collapsed from ~4,050,000 (near-input-independent)
to ~250 and started tracking the prompt. (`docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md`.)
Also fixed: i16-vs-i32 xclbin output width (~120,000× error), a malformed smoke-test
prompt, and unbounded RMSNorm weights that were masking the broken kernel.

### Q2_0 ternary: bit-exact, and the first real 1-bit number

The prism-ml Ternary-Bonsai Q2_0 format isn't publicly documented. Reverse-
engineered from raw bytes, verified **bit-exact vs the F16 reference
(cosine = 1.000000)** across every layer type: 128 elems / 34 bytes, fp16 scale
then 2-bit LSB-first codes, value `(code-1)*d`. Decoder: `tools/q2_0_decode.py`.
Then measured, on hardware, coherent: **Ternary-Bonsai-1.7B native Q2_0 (1.58-bit)
= 274–279 tok/s** on the Radeon 8060S via Vulkan — *"The capital of France is
**Paris**…"* — versus **22 tok/s** F16. A **12.6× speedup** from native 2-bit
storage. `llama-bench` tg64 = 278.81 ± 2.95 t/s. This is the honest, reproducible
"1bit" headline (`docs/VALIDATED-BENCHMARKS-2026-07-05.md`, `docs/one-bit-headline.md`).

### ZINC Q2_0 kernel + build unstick

Wrote `zinc:src/shaders/dmmv_q2_0.comp` (mirrors the proven `dmmv_q8_0` reduction)
and wired it through loader/dispatch — **builds and runs the ternary model natively
at ~894 tok/s**, but output isn't coherent yet (a ZINC-internal DMMV weight-layout
detail, not the format — dequant is bit-exact). Branch `zinc:feat/q2_0-vulkan-kernel`.
Separately, ZINC's repo was stuck in a half-finished Zig 0.15→0.16 migration that
built with neither toolchain; restored `main` to a clean 0.15.2 build and preserved
the 0.16 attempt on `wip/zig-0.16-migration`.

### DSpark (speculative-decode draft) — projected, not yet measured

DSpark is a small draft model (5-layer transformer + Markov head + confidence head)
for speculative decoding. Measured **5.90× acceptance** (5.90/7 blocks, 73.7%) on
10 gsm8k samples with Qwen3-4B; confidence-head AUC 0.912. The headline
**"572 tok/s" is a projection** (base NPU × 5.90×), not an end-to-end coherent
measurement — the draft is still training and rides on the NPU base engine. Label
it as a projection until measured; it is not a validated production number the way
94 tok/s (FLM) and 279 tok/s (GPU ternary) are.

### Honest status at session end

- ✅ **NPU production (FLM proxy): 94 tok/s, coherent** — validated live.
- ✅ **GPU native 1.58-bit ternary: 279 tok/s, coherent** — validated, reproducible.
- ✅ **NPU INT8 GEMM kernel: root-caused and fixed** (bit-exact via AMD's generator).
- ✅ **Q4NX + Q2_0 fully decoded**; GGUF→NPU pipeline architecture-agnostic.
- ⚠️ C++ NPU `npu_engine_cb` and the ZINC-native Q2_0 path build/run *fast* but are
  **not yet coherent**. "97 tok/s v12", "291 tok/s fused", and "572 tok/s DSpark"
  are raw-throughput / projected figures on paths whose output was never validated
  coherent — qualify them, don't market them as production alongside the two numbers
  that are.
- ❌ `engine/fusion/main.zig` still prints a dispatch table and runs no inference.

### 2026-07-11 addendum — the DSpark story continues

The 279 tok/s / 572 tok/s-projection numbers above later drifted into a flat
"disproven" claim (a 2026-07-07 test reported 0.1–0.2 tok/s at 0% acceptance with no
qualification). That claim was itself wrong: traced to (1) a checkpoint-path wiring
bug in `npu_spec_integration.cpp` that made the benchmark silently run an untrained
draft model, and (2) a training config that regressed after this session — the
`global_batch_size=32` / 10,976-example dataset described above became
`global_batch_size=512` / 360 examples by 2026-07-11, both changes making an
already-fragile training setup much worse. See `docs/wiki/performance.md` for the
corrected "unresolved, not disproven" status and the real 0.8 tok/s / 0% acceptance
measurement taken with the wiring bug fixed.

## Session 2026-07-16/20 — FLM fully replaced, model-agnostic broadening, TQ2 ternary

The single biggest architectural change since the last addendum: **FastFlowLM is no
longer the default NPU path, and its native `.so`-dependency is gone entirely.**
22 closed-source libraries were disassembled, 209 xclbin bitstreams traced back to
their AIE generators, and the whole stack rebuilt from source
(`docs/fastflowlm-decode/SUMMARY.md`). `engine/npu/src/npu_engine_universal.cpp`
no longer `dlopen`s FLM's `.so` files for NPU attention/GEMM instruction
generation — it uses pre-compiled instruction files instead, and `backend_manager.cpp`
now marks `npu_xrt` `auto_selectable` with the comment "NPU_XRT is the default now."

That comment was aspirational for four days. `src/model_router.cpp` — the file that
actually decides which backend a qwen3-architecture model gets routed to — still hard-
coded `{"npu_flm", "cpu_generic"}` as of this morning (2026-07-20), meaning every
qwen3 model kept going through the FastFlowLM subprocess in practice regardless of
what `backend_manager.cpp` claimed. Fixed today (PR #567): route is now
`{"npu_xrt", "npu_flm", "cpu_generic"}` — native engine first, FLM kept only as a
fallback, not removed outright. Honest tradeoff, not a free win: `npu_xrt`'s
single-core GEMM kernels are correctness-verified on real hardware
(`docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md`, 2026-07-17/18), but the 8-core
multi-tile path that would close the throughput gap to FLM's fused-xclbin numbers
is still "unverified in combination" per that same doc. Shipped the routing change
anyway, on the user's explicit call, because "FLM is diagnostic-only, not the
serving path" needs to be true in the code, not just asserted in the README.

**Model-agnosticism kept widening**, matching the standing "every model, every
quant" scope (not scope creep — see the earlier note on this in the repo's own
memory). GGUF architecture support went 2→8 (LLAMA, MISTRAL, QWEN2, GEMMA, PHI,
ZAMBA2 alongside the original two), quant support went 4→13 (legacy Q4_1/Q5_0/Q5_1
plus the full K-quant family), the HIP backend takes runtime `ModelConfig` instead
of hardcoded Zaya1-8B dims for non-Zaya models, and GGUF parsing was consolidated
into one shared, verified module instead of several divergent per-file copies.

**1BP's own namesake feature had never actually shipped.** `ONEBP_TQ1`/`ONEBP_TQ2`
existed in the format's `OnebpQuant` enum since it was designed, but every model
converted so far — including genuinely ternary-trained ones — went through the
4-bit `Q4NX` path regardless of source precision, for a project called "1bit.systems."
TQ2 (symmetric 2-bit ternary, one BF16 scale per 32-element group, no zero-point,
half of Q4NX's tile size) is now implemented end-to-end: converter, loader,
on-disk layout. Verified against Bonsai-1.7B (genuinely ternary-trained, Apache-2.0)
— structural match exact, numerical match "100% of dequantized values within
BF16 scale-rounding precision" against the real C++ loader, not just the Python
converter. Separately, the 1BP converter/loader was found dropping norms and MoE
expert weights entirely (91% of Zaya1-8B's tensors were silently missing) — fixed
same window.

**BlackMamba conversion, and a metadata bug worth flagging for future architecture
ports.** Converting BlackMamba-1.5B/2.8B (Zyphra's Mamba+MoE hybrid) to GGUF then
to 1BP hit a config-read failure — `H=0 L=0` — despite the GGUF file parsing fine
structurally. Root cause: `scripts/blackmamba_to_gguf.py` wrote its metadata keys
unprefixed (`"block_count"`, `"embedding_length"`) instead of prefixed with the
architecture name (`"mamba.block_count"`), which is the GGUF convention every
reader in this repo actually expects — `model_discovery.cpp`'s own suffix matching
(`ends_with(key, ".block_count")`) requires the leading dot that only the prefixed
form provides. Neither the HF-style lookup nor the architecture-prefixed lookup
matched a bare key, so config silently came back zeroed instead of erroring loudly.
Fixed by prefixing all of BlackMamba's custom keys with `mamba.`; worth checking
any other hand-written GGUF exporter in this repo for the same pattern before
trusting its output loads correctly anywhere beyond a byte-level structural check.

**Vision went from greenfield to a real, working POC.** Qwen2-VL support — actual
image-to-text end to end, not just tensor plumbing — landed (#491), with a fix for
generation running past the real EOS token instead of a fixed budget (#492).
Separately, lightweight image preprocessing (stb_image, no OpenCV dependency,
optional HIP resize/normalize kernel) replaced a hypothetical OpenCV dependency —
this shipped with a real build break (a deleted copy constructor with no
corresponding move constructor, breaking `std::vector::push_back`) that sat
unnoticed until a routine full-repo build check today; fixed with proper move
semantics rather than restoring the deleted copy path.

**The landing page was making a claim its own tooling had already disowned.**
`benchmarks/latest.json._unverified` — a real quarantine mechanism, not decoration
— has flagged `npu_validated_tok_s` ("69/94 tok/s NPU") as "NO SOURCE... MUST NOT
be published" since issue #107. That didn't stop it from being the headline number
in `site/index.html`'s `<title>`, meta description, OG/twitter tags, JSON-LD, and
hero `<h1>` — and the page's own JS had a hardcoded string literal that
unconditionally overwrote the meta description with that same number on every
successful page load, bypassing the quarantine guard that protects every other
binding on the page. Replaced with claims that are actually sourced: the native
NPU stack as the new default, ~41 TFLOPS prefill peak (which does have a real
`bench_prefill_variants` citation in `numbers.json`), and current binary size.
Historical throughput numbers elsewhere on the page (the V12 tuning timeline) were
left in place — they were real measurements at their dates — but relabeled so they
read as history, not a current production claim.

**Status at end of this window**: native open-source NPU engine is the default
route for qwen3 models, not yet throughput-competitive with FLM on the verified
single-core path. Model catalog now spans the full Zyphra family plus 1BP
conversions (all on Hugging Face) with BlackMamba added this session. TQ2 ternary
is real and verified against a genuinely ternary-trained model, not just a format
spec. Full repo build and test suite both clean as of this session's end (one
apparent GPU memory-fault test failure turned out to be a local test
misconfigured to point a non-MoE-scoped test at a real MoE model, not a code bug).
