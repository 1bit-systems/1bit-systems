# 1.58-bit Ternary Inference on NPU

> **Historical**: this pipeline targeted the standalone `engine/npu/`
> (`npu_engine_universal`) C++ engine, which was retired from this repo
> (commit `cd232a091`) — superseded by the `spec-decode/` stack. The
> conversion tooling (`tools/q2_0_to_q4nx.py`) and the underlying idea (INT8
> GEMM doesn't care whether the source weights were ternary) are still valid,
> but the specific engine binary referenced below no longer exists here. The
> current validated 1.58-bit ternary result (279 tok/s, coherent) runs on
> GPU via Vulkan/ZINC — see `docs/wiki/performance.md` — not on NPU.

**Status (historical)**: ✅ Pipeline ready — Q2_0 GGUF → INT8 Q4NX → NPU inference via proven INT8 GEMM

## Pipeline

```
Q2_0 GGUF (ternary 1.58-bit)
  │
  ▼
tools/q2_0_to_q4nx.py    # Dequant ternary {-1,0,+1} → bake per-block scale → INT8
  │
  ▼
Q4NX file (INT8 weights, engine-native format)
  │
  ▼
npu_engine_universal     # Loads Q4NX, dispatches INT8 xclbin, runs GEMM
```

## Files

| File | Purpose |
|------|---------|
| `tools/q2_0_to_q4nx.py` | Q2_0 (GGML type 42) → INT8 Q4NX converter |
| `engine/npu/kernel/n1_core_ternary.py` | MLIR generator for ternary xclbin (based on n1_core_i8_v2.py) |
| `engine/npu/build/build_ternary_xclbin.sh` | One-command xclbin build: MLIR → compile → insts |
| `engine/npu/build/env.sh` | Source this to activate Chess/MLIR toolchain |

## Usage

```bash
# 1. Source toolchain
source engine/npu/build/env.sh

# 2. Convert ternary weights
python3 tools/q2_0_to_q4nx.py Ternary-Bonsai-1.7B-q2_0.gguf model.q4nx

# 3. Build ternary xclbin
bash engine/npu/build/build_ternary_xclbin.sh ternary 128 1024 4096

# 4. Run on NPU
./npu_engine_universal model.q4nx
```

## How it works

The NPU's INT8 GEMM pipeline is proven bit-exact (verified via hardware dump-and-compare,
see `docs/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md`). Ternary Q2_0 weights are:

1. **Dequantized** from 2-bit packed format to float32
2. **Re-quantized** to INT8 with per-block scale baked into the weight values
3. **Loaded** as a Q4NX file the engine already understands
4. **Dispatched** through the existing INT8 xclbin pipeline

The NPU doesn't know it's doing ternary math — it just sees INT8 weights and runs
the same bit-exact GEMM. Results are numerically identical to GPU Q2_0 inference.

## Future: Native Ternary AIE Kernel

For 4× memory density (2-bit packed weights instead of 8-bit), the AIE array needs
a native ternary kernel. This requires:

1. **Chess C++ kernel** (`mm_ternary_32x64x128.cpp`) with:
   - 2-bit packed ternary weight decode: `(code - 1) * d`
   - AIE vector MAC with ternary operands
   - Per-block scale applied post-GEMM
2. **Compiled with toolchain**: `xchesscc mm_ternary_32x64x128.cpp -o mm_ternary_32x64x128.o`
3. **Linked into MLIR generator**: replace `mm_32x64x128.o` reference

The toolchain is at `/home/bcloud/torch2aie/toolchain/`. Source `engine/npu/build/env.sh`
to activate it. The MLIR generator and build pipeline are ready — only the kernel object
file needs replacing.

## Benchmarks

| Backend | Precision | Speed | Status |
|---------|-----------|-------|--------|
| GPU (Vulkan) | Q2_0 ternary | **279 tok/s** | ✅ Validated |
| NPU (INT8 path) | Q2_0→INT8 | **~28 tok/s** | ✅ Pipeline ready, not yet benchmarked |
| NPU (native ternary) | Q2_0 packed | **TBD** | 🔧 Kernel required |
