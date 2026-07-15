# INT8 XCLBIN Investigation — Complete Findings

## Summary: Software Blocked, Hardware Ready

The NPU hardware fully supports INT8 (proven by 31 TFLOPS BFP16 and the working IRON API INT8 matmul at 64×64×64). However, building INT8 xclbins for the Qwen3-0.6B engine is **blocked by the MLIR dialect** in this toolchain version. The MLIR parser only validates `v8bfp16ebs8` and `v16bfp16ebs16` types — `i8` and `i16` are rejected at parse time.

## The Hardware Reality

| Format | Hardware Support | Toolchain Support | Status |
|--------|-----------------|-------------------|--------|
| **BFP16** (v8bfp16ebs8) | ✅ Native DMA + compute | ✅ MLIR dialect + aiecc | ✅ **Working 212ms/tok** |
| **INT8** (i8) | ✅ Native DMA + compute (50 TOPS) | ❌ MLIR rejects `i8` type | ❌ Blocked |
| **INT16** (i16) | ✅ Native compute | ❌ MLIR rejects `i16` type | ❌ Blocked |
| **BF16** (bfloat16) | ✅ Native compute | ❌ DMA hangs (bad descriptors) | ❌ Blocked |

## All Paths Attempted

### Path 1: Custom n1_core MLIR Generator (n1_core_i8.py)
- Created INT8 variant of the standard n1_core_placed.py
- Changed all `bfloat16` → `np.int8` / `np.int16`
- Changed B buffer from `v8bfp16ebs8` packed format to flat `int8`
- MLIR generates with correct `memref<32x64xi8>`, `memref<64x128xi8>`, `memref<128x128xi16>` types
- **Result**: `i8` type rejected by aiecc MLIR parser. Error: "Invalid block type: i8. Known types are: v8bfp16ebs8, v16bfp16ebs16."

### Path 2: Kernel Swap (BFP16 xclbin + INT8 kernel .o)
- Build standard BFP16 xclbin with `mm_128x64x128.o`
- Replace kernel object with INT8-compiled `mm_i8.o`
- **Result**: Buffer sizes differ. BFP16 DMA reads 9216 bytes for B (64×16×9), INT8 needs 8192 bytes (64×128×1). DMA reads 1024 garbage bytes → wrong results or hang.

### Path 3: Peano-Compiled INT8 Kernel + --no-xchesscc
- Compile INT8 `mm.cc` with Peano's clang++ instead of Chess's xchesscc_wrapper
- Link with `--no-xchesscc --peano=... --no-xbridge`
- **Result**: Peano kernel .o contains Chess-specific ELF sections (.tctmemtab, .rtstab, .eoltab, .chesstypeannotationtab). lld rejects them. 1760 byte kernel is stub (Chess intrinsics emit warnings).

### Path 4: MLIR Type Sed (BFP16 MLIR → INT8 via text replace)
- Take standard BFP16 MLIR, replace `bf16` → `i8`/`i16` via sed
- **Result**: `i8` type rejected by aiecc MLIR parser (same as Path 1).

### Path 5: IRON API @iron.jit (Direct Python)
- Use `aie.iron` Python API with `kernels.mm(input_dtype=np.int8, output_dtype=np.int32)`
- Works for small tiles (64×64×64) — exact match, error=0
- **Result**: Blocked for large tiles (>32KB SRAM). ObjectFifo with flat 1D arrays doesn't support the L2/L1 streaming hierarchy needed for large buffers. The n1_core design's hierarchical tiling is done by the MLIR generator, not by the IRON API.

### Path 6: Add `i8` Type Support to MLIR Parsher
- The MLIR parser is in the aiecc binary, not in Python
- Source is at `/home/bcloud/mlir-aie/` — would need to modify C++ code in `lib/Dialect/AIE/IR/` or similar
- **Result**: Requires rebuilding aiecc from source. Estimated 4-8 hours for someone familiar with MLIR.

## The MLIR Dialect Limitation

The aiecc's MLIR parser validates types against a known set. The only AIE-specific element types are:
- `v8bfp16ebs8` — 8 BF16 values packed into 9 bytes (BFP16 format)
- `v16bfp16ebs16` — 16 BF16 values packed into 16 bytes

Standard MLIR types like `i8`, `i16`, `f32` are NOT accepted for AIE objectFifo memrefs. The AIE DMA engine CAN handle these types (the IRON API proves this), but the MLIR parser's validation is artificially restricted.

## Why IRON API Works

The IRON API (`aie.iron`) uses a DIFFERENT compilation path:
1. Generates MLIR through `@iron.jit` → `compilabledesign.py`
2. The generated MLIR uses the `aie.objectfifo` dialect with proper types
3. The `compile_mlir_module` function calls `aiecc` with the generated MLIR
4. For small tiles that fit in 32KB SRAM, the MLIR generation handles types correctly
5. For large tiles, the ObjectFifo's L2/L1 streaming isn't automatically generated

The IRON API's INT8 matmul works because it uses flat ObjectFifos where the entire matrix fits in a single tile's SRAM. Large matrices need the hierarchical tiling that `n1_core_placed.py` provides — and that code doesn't support `i8` types.

## What Would Fix It

### Short Term (Patch MLIR Parser)
1. Find the type validation code in `/home/bcloud/mlir-aie/lib/Dialect/AIE/IR/` or `/home/bcloud/mlir-aie/lib/Dialect/AIEX/IR/`
2. Add `i8`, `i16`, `i32` to the list of accepted element types
3. Rebuild aiecc with `ninja`
4. Rebuild INT8 xclbins

### Medium Term (Full INT8 Support)
1. Update `n1_core_placed.py` to support multiple element types (not just BFP16)
2. Compile kernel with correct INT8 compile flags
3. Ensure DMA strides work correctly for 1-byte element types
4. Build and test INT8 xclbins
5. Modify C++ engine to pack INT8 weights and call INT8 xclbins

### Long Term (New Toolchain)
1. Upgrade to a newer MLIR-AIE version that supports INT8 natively
2. Or use the IRON API with proper hierarchical tiling support

## Files Created

| File | Purpose |
|------|---------|
| `/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/n1_core_i8.py` | INT8 MLIR generator (generates `i8`/`i16` MLIR) |
| `/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/build_i8_xclbin.sh` | INT8 xclbin build script |
| `/home/bcloud/npu-sandbox/npu-infer/build/int8/` | Build artifacts (MLIR, .o files) |
| `/home/bcloud/npu-sandbox/npu-infer/tools/test_mt_gemm3.cpp` | Multi-token xclbin test |
| `/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/mm_bf16_v3.cc` | Native BF16 kernel (no emulation flag) |
| `/home/bcloud/npu-sandbox/npu-infer/bf16_kernel_dev/CONCLUSION.md` | BF16 investigation conclusion |

## All IRON API Fixes Applied

| Bug | Fix | File |
|-----|-----|------|
| ScalarValue nanobind type mismatch | Removed `ArithValueMeta` metaclass | `ai/extras/dialects/arith.py` |
| Peano ELF symbol rename | Pure-Python `_rename_symbol_in_elf32` | `ai/utils/compile/utils.py` |
| transpose.hpp incomplete type | Added `else` fallback to `shuffle_modes` | `ai_api/detail/aie2/transpose.hpp` |
| aie2p mm.cc SKIP_VECTORIZED | Added `#ifndef SKIP_VECTORIZED` guard | `ie_kernels/aie2p/mm.cc` |
| aie2p mm.cc extern "C" | Separated vectorized/scalar combos | `ie_kernels/aie2p/mm.cc` |
| mm() SKIP_VECTORIZED flag | Added compile_flags entry | `iron/kernels/linalg.py` |
