# ROCm GPU Inference Engine — Handoff

## Current State (master @ 9ed49db)

The GPU engine (`rocm_engine.cpp`) produces all-identical tokens (82291 repeated) instead of the correct output. After 15+ hours of systematic debugging, the root cause has been isolated with high confidence.

### What's Working
- **GPU GEMM kernels**: Verified accurate (< 0.00008 max diff from CPU reference)
- **Compilation and runtime**: Engine builds with system `hipcc`, runs on Radeon 8060S at ~106 ms/tok, no segfaults/NaN/Inf
- **CPU reference match**: A standalone CPU reference (`cpu_ref_infer.cpp`, `/tmp/cpu_ref_debug.cpp`) produces EXACTLY the same output as the GPU engine — confirming the engine logic itself is self-consistent
- **NPU engine works**: `npu_infer` (built from `engine.cpp` + xclbins) produces correct diverse output: `919 996 185 385 495 ...` from the same model file

### What's Wrong

**All gate and up projection outputs are 100% negative → SiLU(gate) ≈ 0 → FFN contributes nothing → hidden state blows up.**

| Metric | Value |
|--------|-------|
| Gate outputs | 100% negative (min -76.5, mean -31.7) |
| Max \|SiLU(gate)\| | 0.000000 (completely saturated at zero) |
| Hidden state after layer 27 | ±60M (vs expected ~±10) |
| Token output | 82291 repeated every step |

### Root Cause: Q4NX Format Incompatibility

The `.q4nx` model file is a **proprietary NPU weight format** for XDNA 2's AIE kernels. `dequant_i8_to_float` interprets it as INT4 with BF16 scale+zero-point per 32-element group, producing weight values that are 94% negative for gate/up projections.

**The AIE kernel dequantizes differently internally**, producing different float values. This is proven by:
1. The NPU's own CPU dequant function (`npu_dequant_block` in `model.c`) produces **NaN/inf** when reading the same bytes — yet `npu_infer` works because the xclbin AIE kernel handles dequant internally
2. All 8 tested GEMM indexing patterns (NPU stride, transpose, etc.) produce 100% negative gate outputs — the weight values are fundamentally wrong, not the GEMM layout
3. Both GPU engine and CPU reference (using same `dequant_i8_to_float`) produce identical wrong output

### Key Investigations Completed

1. **Weight loading fixes applied and verified**: QKV/GU transpose is correct (0 diff from reference), O/Down use direct memcpy (also correct). ALL weight loading is now correct for the `dequant_i8_to_float` interpretation.

2. **BF16 endianness check confirmed**: Scales are little-endian BF16 (correct), NOT big-endian. The tiny scale values (0.007–0.019) are valid. Dequantization IS working correctly for the format it assumes.

3. **CPU reference matches GPU exactly**: Both use `gw[k*3072+i]` NPU stride pattern, both produce all-negative gate. The bug is in the *weight values*, not the forward pass logic.

4. **NPU weight BO analysis**: The NPU loads raw I8 rows (no dequantization on CPU) and sends them to the xclbin. The `engine.cpp` code packs weights using `npu_pack_weight_bo` which copies I8 bytes directly into a BO. The AIE kernel does its own dequant.

### Files and Artifacts

| File | Purpose |
|------|---------|
| `src/rocm_engine.cpp` | GPU engine with debug dumps and correct weight loading |
| `tools/cpu_ref_infer.cpp` | Standalone CPU reference (compile: `g++ -O2 cpu_ref_infer.cpp dequant_q4nx.o -lm`) |
| `/tmp/debug_qkv_test.cpp` | Standalone weight verification test |
| `/tmp/dequant_vs_raw.cpp` | Compares dequant vs raw-BF16 weight reading |
| `/tmp/all_layouts.cpp` | Tests all 8 GEMM indexing patterns |
| `/tmp/cpu_full_prefill.cpp` | Full CPU pipeline for prefill (compares with engine) |
| `docs/superpowers/plans/2026-07-01-debug-findings.md` | Detailed debug findings |
| `docs/superpowers/plans/2026-06-30-fix-output-correctness.md` | Previous plan (outdated) |

### Build & Run

```bash
cd /home/bcloud/npu-sandbox/npu-infer/build
hipcc -D__HIP_PLATFORM_AMD__ -std=c++17 -O2 -I../include -I/opt/rocm/include \
    -c ../src/rocm_engine.cpp -o rocm_engine.o
hipcc -o rocm_engine rocm_engine.o dequant_q4nx.o -lm
LD_LIBRARY_PATH=/opt/rocm/lib ./rocm_engine
```

## Recommended Next Steps

### Option A: Export HF weights (Recommended — cleanest path)

Download original Qwen3-0.6B from HuggingFace, export to a format the GPU engine can load directly:

1. Use `transformers` to load Qwen3-0.6B safetensors
2. Export weights as raw float32 or BF16 binary files
3. Update `rocm_engine.cpp` to load from these files, skipping `dequant_i8_to_float` entirely
4. This removes all dependency on the Q4NX format

**Expected effort**: ~2-4 hours (download + script + load code changes)
**Risk**: Low (standard HF → binary export is well-understood)

### Option B: Reverse-engineer Q4NX format

Study the xclbin AIE kernel to understand how it dequantizes. This requires:
1. Dumping the xclbin's weight arguments after `npu_pack_weight_bo` writes them
2. Running the AIE kernel with known inputs and capturing outputs
3. Reverse-engineering the I8→BF16 conversion formula

**Expected effort**: 20+ hours (very difficult, no documentation)
**Risk**: High (the xclbin is proprietary, may be obfuscated)

### Option C: Use NPU for inference, GPU only for debug

The NPU engine already works correctly. Use the GPU engine purely for validation/debugging while focusing on:
1. Integrating the working NPU engine into the production pipeline
2. Benchmarking and optimizing the NPU path (currently 174 ms/tok)
3. Building the GPU engine only for test workloads with known-good HF weights

## Critical Notes for Next Worker

- **The forward pass logic in `rocm_engine.cpp` is CORRECT** — verified by CPU reference matching it exactly and all GEMM kernels passing accuracy tests
- **The weight loading into the engine's 'layers_[l].gate' arrays is CORRECT** for the `dequant_i8_to_float` interpretation — verified by standalone C++ tests comparing against reference computation (diff=0)
- **DO NOT spend more time debugging weight layout/transpose/stride** — all 8 patterns produce 100% negative gate, this is a dead end
- **The Q4NX format contains valid data** — the NPU works, proving the model file is correct. Our dequant interpretation is simply wrong for CPU/GPU use
- **When loading HF weights** (Option A), the model architecture is: 28 layers, 16 Q heads, 8 KV heads (GQA=2), HD=128, H=1024, IM=3072, NV=151936, RoPE base=1000000
