# Phase 1b: Productionize NPU Engine + GPU Hybrid Architecture

## Context

Phase 1a proved INT8 GEMM on the XDNA2 NPU (18/18 tests, 100% correct, 442 ms/tok on Qwen3-0.6B). The NPU is **not** the primary inference engine — it's a coprocessor for tiny always-on models. The **Radeon 8060S (40 CU, ~20 TFLOPS)** with ROCm/hipBLAS is the real workhorse.

This phase cleans up the NPU engine (engine.cpp/engine.h reconciliation, NpuGemm refactor), creates the GPU inference engine (ROCm/hipBLAS), and builds the unified hybrid architecture.

## Global Constraints

- `include/engine.h` declares `CustomNpuEngine` — `src/engine.cpp` must match it exactly
- NPU xclbins remain as-is: 5 INT8 xclbins (QKV, O, GU, D, KV-reuse), 100% verified
- GPU engine uses ROCm 7.1, gfx1151 target, hipBLAS for GEMM
- GPU models use llama.cpp GGUF format (Q4_K_M, IQ1_S for 1-bit ternary)
- `npu_engine_i8.cpp` is the single active NPU engine — other `src/engine_v*.cpp` are stale
- All tests pass before commit (NPU tests 18/18)

## Tasks

### Task 1: Reconcile engine.h / engine.cpp

**Spec:**
1. `include/engine.h` declares `class CustomNpuEngine` with `Init(model_path)` and `Generate(input_ids, input_len, output_ids, max_len)` — this is the interface contract
2. `src/engine.cpp` currently defines a completely different class `class NpuInferenceEngine` — it must be rewritten to implement `CustomNpuEngine` exactly
3. The new `engine.cpp` delegates to `npu_engine_i8.cpp`'s `I8Ctx` structs (QKV, O, GU, D contexts) for actual NPU execution
4. Remove the old `AdaptiveModel`, `TokenStreamer`, `FLM` references — the NPU engine is standalone
5. Engine weights are loaded from `.q4nx` files (same format as FLM, dequantized to float at init)
6. Keep all 5 xclbins loaded in `I8Ctx` — reuse the working inference pipeline from `npu_engine_i8.cpp`

**Files to modify:**
- `src/engine.cpp` — rewrite to match `engine.h`
- `include/engine.h` — verify/update if needed

### Task 2: Extract NpuGemm class

**Spec:**
1. Read `src/npu_engine_i8.cpp` and extract the `I8Ctx` struct + `init_xrt`/`run_gemm`/`cleanup_xrt` into a reusable class `NpuGemm` in `include/npu_gemm.h`
2. `NpuGemm` has: `init(xclbin_path, inst_path, M, K, N)`, `run(A, B, C)`, `cleanup()`
3. `NpuGemm` handles BO allocation, DMA, kernel execution, result readback
4. `npu_engine_i8.cpp`'s `main()` becomes a thin wrapper that creates 5 `NpuGemm` instances
5. Preserve all working DMA logic: broadcast-mode A, sequential per-group drain, correct K-loop

**Files to create/modify:**
- `include/npu_gemm.h` — new NpuGemm class
- `src/npu_engine_i8.cpp` — refactored to use NpuGemm

### Task 3: Create GPU inference engine

**Spec:**
1. New file: `src/gpu_engine.cpp`, header: `include/gpu_engine.h`
2. `class GpuInferenceEngine` with same `Init(model_path)` and `Generate(...)` interface
3. Uses ROCm 7.1 + hipBLAS for GEMM ops
4. Reads GGUF model weights directly (llama.cpp GGUF parser or minimal loader)
5. Implements same Qwen3-0.6B architecture as the NPU engine: RMSNorm, RoPE, SwiGLU FFN, MHA
6. Compiles with `hipcc -std=c++17 -O2 -I../include -I/opt/rocm/include -L/opt/rocm/lib -lhipblas -lm`
7. Test: produce identical logits to NPU engine (within FP32 tolerance)

### Task 4: Multi-token decode NPU (M=256)

**Spec:**
1. Generate M=256 xclbins from `bf16_kernel_dev/n1_core_i8_v2.py` — 2× batch size
2. QKV: 256×1024×4096, O: 256×2048×1024, GU: 256×1024×6144, D: 256×3072×1024
3. Build with `bf16_kernel_dev/build_all_int8.sh` or manual aiecc invocation
4. Verify correctness with test_correctness.cpp (constant data)
5. Update npu_engine_i8.cpp to use M=256 xclbins when batched, M=128 for single-token

### Task 5: Clean stale engine variants

**Spec:**
1. Remove these stale files:
   - `src/engine_v2.cpp` through `src/engine_v10.cpp` (if they exist)
   - `src/engine_fused.cpp`, `src/engine_batched.cpp`, `src/engine_chess.cpp`, `src/engine_attn.cpp`
   - Any other `src/engine_*.cpp` except `engine.cpp`, `npu_engine_i8.cpp`, `gpu_engine.cpp`
2. Remove stale generator scripts from `bf16_kernel_dev/`:
   - Keep: `n1_core_i8_v2.py`, `n32_core_placed.py`, `n1_core_bf16.py`
   - Remove: any `gen_*.py` or `mm_*.py` that are superseded by the above
3. Remove unused chess kernel variants from `bf16_kernel_dev/` or `include/`
4. Each deletion as a separate commit for traceability

## Parallel Task Groups

Tasks 1+2 are sequential (engine.cpp depends on NpuGemm class). Tasks 3+5 are independent. Task 4 requires xclbin build (may need NPU hardware).

## Verification

| Test | Command | Passes |
|------|---------|--------|
| NPU tests (all 18) | `pytest tests/ -v` | 18/18 |
| NPU engine compile | `g++ -std=c++17 -O2 -Iinclude -I/path/to/torch2aie/examples src/engine.cpp src/npu_engine_i8.cpp -lxrt_coreutil -lm -o npu_engine` | compiles |
| GPU engine compile | `hipcc -std=c++17 -O2 -Iinclude -I/opt/rocm/include -L/opt/rocm/lib src/gpu_engine.cpp -lhipblas -lm -o gpu_engine` | compiles |
| GPU output match | `python3 tests/cpu_ref_gemm.py` vs GPU logits | within 1e-5 |
| M=256 xclbins verify | `./tools/verify_gemm.py` | 100% correct |

---

## Technical Background: The Full Stack

```
┌─────────────────────────────────────────────┐
│              Custom Application              │
├─────────────────────────────────────────────┤
│          Unified Engine API                  │
│  CustomNpuEngine::Init/Generate             │
│  GpuInferenceEngine::Init/Generate          │
├──────────────────┬──────────────────────────┤
│  NPU Backend     │  GPU Backend             │
│  (xrt, 5 xclbins)│  (ROCm 7.1, hipBLAS)    │
│  Tiny 0.6B only  │  7B-200B, video, vision │
├──────────────────┴──────────────────────────┤
│          128 GB Unified Memory              │
│  Qwen3-0.6B INT8 (0.6GB) — always loaded    │
│  Qwen3-7B Q4_K_M (4GB) — GPU primary       │
│  Qwen3-32B Q4_K_M (18GB) — GPU reasoning   │
│  200B at 1-bit ternary (50GB) — fits        │
│  SDXL / Hunyuan / Whisper / TTS             │
└─────────────────────────────────────────────┘
```

The NPU engine is already proven (Phase 1a complete). Phase 1b productionizes it and brings the GPU online in parallel, then unifies both into one coherent system.

