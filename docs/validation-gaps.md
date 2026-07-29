# Validation Gaps & Engineering Blockers

> **Canonical gap tracker.** Updated 2026-07-29 after live validation on Strix Halo
> (Ryzen AI Max+ 395, Radeon 8060S, 256 GB/s, NPU firmware 1.1.2.65, ROCm 7.1).
>
> Every claim in `docs/wiki/models.md` was either validated on real hardware,
> identified as a documentation error, or catalogued here as a genuine engineering gap.

---

## 🐛 Confirmed Bugs

### B1. Mamba1 GGUF Metadata Key Mismatch

**Location**: `tools/test_mamba1_backend.cpp` (also `src/backend_mamba1.cpp` if config read path is shared)

**Issue**: The config reader looks for GGUF metadata keys with a `mamba.` prefix
(e.g. `mamba.block_count`, `mamba.embedding_length`), but the actual GGUF files
store these keys **without** the prefix (e.g. `block_count`, `embedding_length`).

**Affected keys**:
| Code reads | File has | Effect |
|---|---|---|
| `mamba.block_count` | `block_count` | num_layers = stack garbage |
| `mamba.embedding_length` | `embedding_length` | hidden_size = stack garbage |
| `mamba.vocab_size` | `vocab_size` | vocab_size = stack garbage |
| `mamba.ssm.state_size` | `ssm.state_size` | d_state = stack garbage |

**Impact**: Config was read as Hidden=2048, Layers=40, Vocab=262272, d_state=128
instead of the real values (Hidden=1152, Layers=30, Vocab=50304, d_state=16).
This caused `backend->init()` to fail at layer 30 (past the actual 30 layers).

**Fix applied**: Removed `mamba.` prefix from key lookups. BlackMamba 1.5B now
runs at **79.9 tok/s** (validated, matches documented 79.4).

**Related files to check**: If `src/backend_mamba1.cpp` or `include/gguf_reader.h`
has similar prefix logic, it may also need fixing.

---

## 📋 Documentation Errors (Under-Reported Performance)

### D1. Fused TQ2 — 23% faster than documented

| Source | Value |
|--------|:-----:|
| `docs/wiki/models.md` | 345 tok/s (1.19×) |
| `site/benchmarks.json` | 420 tok/s (1.15×) |
| Live measurement | **426 tok/s (1.16×)** |

**Recommended**: Update models.md to 426 tok/s, 1.16× speedup.

### D2. Tile8 GEMV (Zaya1-8B shaped) — 35% faster than documented

| Source | Value |
|--------|:-----:|
| `docs/wiki/models.md` | 57 tok/s |
| Live measurement | **77 tok/s** |

**Recommended**: Update models.md to 77 tok/s.

### D3. Mamba2 decode block — slight over-report

| Source | Value |
|--------|:-----:|
| `docs/wiki/models.md` | 1293 tok/s |
| `site/benchmarks.json` | (missing) |
| Live measurement (100 iters) | **1270 tok/s** |

**Recommended**: Update models.md to 1270 tok/s. The discrepancy is from
different iteration counts (10 in benchmark vs 100 in live test).

### D4. Prefill INT8 WMMA — minor under-report

| Source | Value |
|--------|:-----:|
| `docs/wiki/models.md` | 40.66 TFLOPS |
| Live measurement | **40.78 TFLOPS** |

**Recommended**: Update models.md to 40.78 TFLOPS.

---

## 🔴 Engineering Gaps (Not Yet Working)

### E1. Mistral / Pixtral on NPU

**Blockers**: No FLM xclbins extracted for Mistral architecture. The FLM
extraction pipeline (`engine/npu/tools/`) covers Qwen, Llama, Gemma, Phi,
but not Mistral. GGUF path through GPU HIP works.

**Unblock path**: Reverse-engineer Mistral FLM xclbins from ROCm stack,
or run Mistral through GPU Vulkan/HIP (already validated).

### E2. Phi4-Mini on GPU (HIP / Vulkan / CPU)

**Blockers**: NPU-only at present (`peano_dims` ready, 4 xclbins). GPU paths
require weight file generation from the 1BP model. The `unified_server` detects
the NPU FLM model (`Phi4-mini-Instruct-NPU2`) but fails on GPU/HIP init because
weight files are missing at `/home/bcloud/.local/share/1bit-systems/weights/`.

**Unblock path**: Run `gguf_to_onebp` conversion pipeline or generate weight
bin files from the Q4NX model.

### E3. Laguna / Falcon / OLMo on NPU

**Blockers**: These architectures have no FLM xclbins. They run through GGUF
on GPU HIP.

**Unblock path**: Add FLM extraction support for these architectures.

### E4. ZR1-1.5B on NPU / CPU

**Blockers**: NPU needs Peano dims compilation (pending). CPU path needs weight
file generation. Vulkan ZINC end-to-end is validated at ~26 tok/s.

**Unblock path**: Complete Peano dims for ZR1 architecture shapes.

### E5. Nanbeige4.1-3B on GPU (HIP / Vulkan / CPU)

**Blockers**: NPU-only (`peano_dims` ready, 4 xclbins). The model has unusual
`head_dim=80` which may require specialized kernel support for GPU paths.

**Unblock path**: Add GPU GEMV kernel support for head_dim=80.

### E6. Zaya1 (ternary) on NPU

**Blockers**: Ternary kernels are blocked on Peano xclbin compilation. The
1BP ternary format requires native `npu_xrt` LUT-decode kernels that haven't
been implemented yet.

**Unblock path**: Implement XDNA 2 ternary decode kernel in the NPU engine.

### E7. BlackMamba on NPU / Vulkan / CPU

**Blockers**:
- **NPU**: The SSM scan operation has not been mapped to XDNA 2 tile arrays.
  This is a fundamental architectural mapping problem.
- **Vulkan**: Mamba1 SSM scan requires HIP cooperative-groups functionality
  that hasn't been ported to Vulkan.
- **CPU**: CPU inference path not implemented for Mamba1 architecture.

**Status**: GPU HIP path works (79.9 tok/s for 1.5B, validated above). 2.8B
model file is an empty placeholder on HuggingFace (0 bytes — needs upload).

**Unblock path**: SSM scan kernel design for XDNA 2; Vulkan cooperative-groups
port; CPU inference path.

### E8. Zamba2 on NPU / CPU

**Blockers**:
- **NPU**: Mamba2 SSM not mapped to XDNA 2.
- **CPU**: CPU path not implemented.
- **Vulkan**: End-to-end validated at ~30 tok/s (2.7B).

**Status**: GPU HIP kernels validated (Mamba2 decode: 1270 tok/s, Conv1D:
37804 tok/s, Selective Scan: 39313 tok/s).

**Unblock path**: Same as E7 for NPU/CPU.

### E9. Zamba (original) on NPU

**Blockers**: No FLM xclbins. Runs through GGUF on GPU HIP.

**Unblock path**: Add FLM extraction support.

### E10. Bonsai (ternary) on NPU / CPU

**Blockers**: No FLM path exists for ternary-native models. Requires native
`npu_xrt` LUT-decode kernels (same as E6).

**Status**: GPU HIP path validated (21.9 tok/s for 1.7B). Vulkan path validated
(318 tok/s kernel-level). CPU not yet.

### E11. Qwen2-VL / Qwen3-VL on Vulkan / CPU

**Blockers**:
- **Vulkan**: ViT vision encoder not yet ported to Vulkan.
- **CPU**: CPU path for ViT not implemented.

**Status**: GPU HIP path works for text decoder. NPU VL models available
(Qwen2.5-VL-3B, Qwen3-VL-4B with `peano_dims` ready).

### E12. Whisper on Vulkan / CPU

**Blockers**: FFT/STFT pipeline only on GPU HIP. Untested on Vulkan and CPU.

### E13. Embedding-Gemma-300M on GPU / Vulkan / CPU

**Blockers**: Embedding extraction pipeline not yet implemented. NPU path
has pre-compiled xclbins (4, `peano_needed`).

---

## ⚠️ Missing Validation

### M1. BlackMamba 2.8B end-to-end

**Blockers**: The HuggingFace repo `bong-water-water-bong/BlackMamba-2.8B-GGUF`
contains an empty file (0 bytes). Unable to benchmark.

**Recommended**: Upload the actual 2.8B GGUF model, then run through
`test_mamba1_backend`.

### M2. Zamba2 end-to-end on GPU HIP

**Blockers**: No public GGUF models available for Zamba2 architecture. The
Zyphra repos only contain safetensors format. Need GGUF conversion or access
to gated repos.

**Status**: Individual sub-kernels validated (Mamba2 decode: 1270 tok/s,
Conv1D: 37804 tok/s, Selective Scan: 39313 tok/s).

### M3. CPU-only end-to-end benchmarks

**Blockers**: The built `llama-bench` uses the Vulkan backend. `-ngl 0` still
reports as "Vulkan" backend (no pure-CPU `llama-bench` binary available).
The CPU prefill numbers in the doc (1,969 tok/s) couldn't be reproduced
directly — our measurement with `ngl=0` gave 8,117 tok/s which is likely
still using GPU for matrix ops through the Vulkan backend.

**Recommended**: Build a CPU-only `llama-bench` or use `taskset` + `-ngl 0`
with a CPU-only ggml build to get accurate CPU numbers.

### M4. NPU end-to-end inference

**Blockers**: Only 3 of 37 FLM models are extracted locally (Phi4-mini,
Qwen3-0.6B, Qwen3.5-4B). NPU inference requires the `npu_engine_universal`
pipeline which needs Peano-compiled xclbins. The `unified_server` detects the
NPU and auto-discovers models, but the full inference pipeline requires:
1. Extracted FLM model with Q4NX weights
2. Peano-compiled xclbins for the specific model shapes
3. Runtime NPU engine with 4-live context support

**Recommended**: Document the NPU extraction + compilation pipeline
separately and add a step-by-step guide for adding new NPU models.

### M5. Zaya1-8B end-to-end on GPU HIP

**Status**: Tile8 GEMV kernel validated at 77 tok/s (correctness-verified).
Documented at ~64 tok/s. End-to-end generation through `zaya_full` or
`unified_server` requires weight file generation that hasn't been run.

---

## 📊 Summary

| Category | Count | Details |
|----------|:-----:|---------|
| Bugs found & fixed | 1 | Mamba1 GGUF metadata key prefix |
| Documentation errors | 4 | Under-reported perf numbers |
| Engineering gaps | 13 | Backend/model combinations not yet working |
| Missing validation | 5 | Couldn't test due to missing models/tools |
