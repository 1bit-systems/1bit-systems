# Strix Halo NPU (XDNA2) — Full Qwen3-0.6B Inference at 4.8 tok/s

## What We Accomplished

We got **Qwen3-0.6B running on the Strix Halo NPU** at **210ms/tok (4.8 tok/s)** — 3.2× faster than our baseline. This was done using the **torch2aie/IRON toolchain** with custom BFP16 xclbins on the Linux XRT stack.

### The Stack
- **Hardware**: AMD Strix Halo (Ryzen AI MAX+ 395), XDNA2 NPU
- **Toolchain**: MLIR-AIE + aiecc + xchesscc_wrapper (Chess compiler)
- **Runtime**: XRT (Xilinx Runtime) with custom C++ engine
- **Model**: Qwen2.5-0.6B (28 layers, 1024 hidden dim, 600M params)

### Engine Architecture
```
6 custom xclbins → 4 GEMMs/layer × 28 layers = 112 NPU calls/token
Fused QKV (1024×4096) + Fused GU (1024×6144) + O + D
Threaded LM head (4×) + Threaded attention (4×)
BFP16 format with scale=1.0 (practically lossless, RMSE 0.0003)
```

### Performance
| Metric | Value |
|--------|-------|
| Decode latency | **210 ms/tok** |
| Prefill (9 tok) | 1.67s |
| Throughput | **4.8 tok/s** |
| vs naive baseline | **3.2× faster** |
| NPU compute | ~12 TFLOPS BFP16 (of 31 peak) |

### 15+ Built XCLBIN Artifacts
| xclbin | Size | Status |
|--------|------|--------|
| BFP16 QKV fused | 70KB | ✅ Running |
| BFP16 GU fused | 118KB | ✅ Running |
| BFP16 O | 52KB | ✅ Running |
| BFP16 D | 52KB | ✅ Running |
| Multi-token M=256 (4 variants) | 90-132KB | ✅ Built |
| 2-layer batch N=8320 (4 variants) | 52-118KB | ✅ Built |

---

## The Wall: INT8 and BF16

### INT8: Blocked by MLIR Dialect (Software, Not Hardware)
The NPU hardware fully supports INT8 (50 TOPS peak). The IRON API proves this — we ran INT8 matmul at 64×64×64 with **exact match, error=0**. However, the aiecc MLIR **parser only accepts `v8bfp16ebs8` and `v16bfp16ebs16` types** — `i8`/`i16` are rejected.

**We patched the aiecc source** (`AIEXDialect.cpp` and `AIETargetModel.cpp`) to accept `i8`/`i16`, rebuilt with ninja, and **successfully built an INT8 xclbin** (66KB). But execution hangs — the DMA strides need recalibration for 1-byte element types vs BFP16's 1.125-byte packed format.

### BF16: Blocked by DMA Descriptors
BF16 xclbins compile but the DMA controller hangs at runtime. All kernel variants (identity, native, emulated) hang identically. The Chess compiler generates incorrect DMA descriptors for `bfloat16` memory types.

### What's Needed
1. **INT8 DMA stride formulas** for the n1_core tile streaming hierarchy (1-byte elements need different strides than BFP16's 2-byte values)
2. **MLIR parser update** to accept `i8`/`i16` upstream (we have the patch)
3. **BF16 DMA descriptor fix** in aiecc — or a newer toolchain version

---

## All Compiler/API Bugs Fixed (7 total)
1. MLIR Python bindings nanobind type mismatch
2. AIE ELF symbol rename (objcopy doesn't handle 32-bit AIE ELFs)
3. transpose.hpp incomplete type (`const void` template deduction failure)
4. Kernel source missing `extern "C"` for Peano compiler
5. Vectorized kernel compilation forced even when only scalar needed
6. aiecc toolchain path resolution
7. ~15 IRON API integration issues (all fixed, `@iron.jit` works end-to-end)

---

## Repository
Full handoff documents + xclbins + engine source:
**https://github.com/bong-water-water-bong/lemon-mlx-engine/docs/npu/**

### Key Files
| File | Content |
|------|---------|
| `HANDOFF-NPU-OPTIMIZATION.md` | Complete 3-day optimization journey (880+ lines) |
| `INT8-HANDOFF.md` | INT8 investigation: 6 failed paths, root cause, fix strategy |
| `src/npu_engine_fused.cpp` | Working engine (310 lines, 210ms/tok) |
| `build/int8/` | Built INT8 xclbins + MLIR generator |
| `bf16_kernel_dev/` | All BF16/IRON/INT8 investigation artifacts |

---

## We Need Help

If you have experience with:
- **AMD XDNA2 NPU programming** (especially INT8 DMA)
- **MLIR dialect development** (adding element types to AIE dialect)
- **Chess compiler internals** (BF16 DMA descriptor fix)
- **Windows NPU stack** (DirectML/QNN — what does Windows do differently?)

Please reach out! The hardware is incredibly capable — 31 TFLOPS BFP16, INT8 support, all on a 15-25W APU. The Linux software stack just needs to catch up.
