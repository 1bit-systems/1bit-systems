# INT8 XCLBIN Investigation — Complete Findings

## Status: INT8 WORKS (Patched & Verified)

**INT8 xclbins compile and run on Strix Halo NPU.** All 5 models use INT8 GEMMs in production. The MLIR toolchain was patched to accept `i8` types, and 23 xclbins were successfully built across 5 model families.

## Summary

The NPU hardware fully supports INT8 (proven by 31 TFLOPS BFP16 and the working IRON API INT8 matmul at 64×64×64). Initially, building INT8 xclbins was blocked because the MLIR parser only validated `v8bfp16ebs8` and `v16bfp16ebs16` types — `i8` and `i16` were rejected at parse time.

**The fix**: Patched `AIEXDialect.cpp` and `AIETargetModel.cpp` in the aiecc source, rebuilt with ninja. INT8 xclbins now compile and run successfully.

## The Hardware Reality

| Format | Hardware Support | Toolchain Support | Status |
|--------|-----------------|-------------------|--------|
| **INT8** (i8) | ✅ Native DMA + compute (50 TOPS) | ✅ Patched MLIR dialect | ✅ **Running — 5 models** |
| **BFP16** (v8bfp16ebs8) | ✅ Native DMA + compute | ✅ MLIR dialect + aiecc | ❌ Abandoned — 17% GEMM error |
| **BF16** (bfloat16) | ✅ Native compute | ❌ DMA hangs (bad descriptors) | ❌ Blocked |

## What Was Fixed

### Path 6: Patch MLIR Parser — SUCCESS
1. Found type validation in `AIEXDialect.cpp` and `AIETargetModel.cpp`
2. Added `i8`, `i16` to accepted element types
3. Rebuilt aiecc with ninja
4. Successfully built INT8 xclbins for all projections

### Verified INT8 xclbins (23 total)
| Model | H | IM | QKV | O | G | U | D | Combined GU | Status |
|-------|---|---|-----|---|---|---|---|------------|--------|
| Qwen3-0.6B | 1024 | 3072 | ✅ | ✅ | — | — | ✅ | ✅ | 28 tok/s |
| Qwen3-8B | 4096 | 12288 | ✅ | ✅ | ✅ | ✅ | ✅ | — | 8 tok/s |
| Qwen3-VL-4B | 2560 | 9728 | ✅ | ✅ | ✅ | ✅ | ✅ | — | 11 tok/s |
| Llama-3.1-8B | 4096 | 14336 | ✅ | ✅ | ✅ | ✅ | ✅ | — | 10 tok/s |
| Gemma4-E2B | 1536 | 6144 | ✅ | ✅ | — | — | ✅ | ✅ | 16 tok/s |

### Remaining INT8 Work
- **GU_2layer xclbin** (N=12288): blocked by AIE core program memory overflow — needs kernel split
- **>8 column xclbins**: blocked by firmware signing requirement

## All Paths Explored (Historical)

### Path 1: Custom n1_core MLIR Generator (n1_core_i8.py)
- Created INT8 variant of the standard n1_core_placed.py
- Changed all `bfloat16` → `np.int8` / `np.int16`
- **Result**: Initially rejected by aiecc MLIR parser. Now works after patches.

### Path 2: Kernel Swap (BFP16 xclbin + INT8 kernel .o)
- Build standard BFP16 xclbin with `mm_128x64x128.o`
- Replace kernel object with INT8-compiled `mm_i8.o`
- **Result**: Buffer sizes differ — BFP16 DMA reads 9216 bytes, INT8 needs 8192 bytes.

### Path 3: Peano-Compiled INT8 Kernel + --no-xchesscc
- Compile INT8 `mm.cc` with Peano's clang++
- **Result**: Peano kernel .o contains Chess-specific ELF sections. lld rejects them.

### Path 4: MLIR Type Sed (BFP16 MLIR → INT8 via text replace)
- Take standard BFP16 MLIR, replace `bf16` → `i8`/`i16` via sed
- **Result**: Same as Path 1 — `i8` type rejected. Now works after patches.

### Path 5: IRON API @iron.jit (Direct Python)
- Use `aie.iron` Python API with `kernels.mm(input_dtype=np.int8, output_dtype=np.int32)`
- Works for small tiles (64×64×64) — exact match, error=0
- **Result**: Blocked for large tiles (>32KB SRAM) — ObjectFifo doesn't support L2/L1 streaming hierarchy.

## Files

| File | Purpose |
|------|---------|
| `/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/n1_core_i8.py` | INT8 MLIR generator |
| `/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/n1_core_i8_v2.py` | INT8 v2 generator |
| `/home/bcloud/npu-sandbox/npu-infer/build/int8/` | All 23 xclbins + MLIR + insts |
| `/home/bcloud/mlir-aie/lib/Dialect/AIEX/IR/AIEXDialect.cpp` | Patched MLIR parser |
| `/home/bcloud/mlir-aie/lib/Dialect/AIE/IR/AIETargetModel.cpp` | Patched target model |
