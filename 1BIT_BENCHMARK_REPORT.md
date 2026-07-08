# 🏎️ 1-bit Model Benchmark Report
**2026-07-08 06:44:25** — Strix Halo (RyzenAI-npu5 / gfx1151)

## 📦 GGUF Models

### 1-bit / Ternary
| Model | Size | Tensors | Format |
|-------|------|---------|--------|
| Bonsai-1.7B-Q1_0.gguf | 248 MB | 310 | Q1_0 (1-bit binary) |

### Standard Quant (for reference)
| Model | Size | Tensors |
|-------|------|---------|
| ggml-model-q4_0.gguf | 32716 MB | 281 |
| ggml-model-f16.gguf | 17739 MB | 1161 |
| ggml-model-zaya.gguf | 5566 MB | 1041 |
| ggml-model-q4_k_m.gguf | 5565 MB | 921 |

## 🧠 NPU XCLBins (Native Ternary)

| XCLBin | Size |
|--------|------|
| npu/build/build/ternary_32core/ternary_32core.xclbin | 310 KB |
| npu/build/build/bitnet_micro/design.xclbin | 22 KB |
| npu/build/build/bitnet_ternary/design.xclbin | 22 KB |
| npu/build/build/ternary_pyapi/design.xclbin | 16 KB |
| npu/build/build/ternary_objfifo/design.xclbin | 16 KB |
| npu/build/build/ternary/design.xclbin | 15 KB |

### XCLBin Architecture
| Variant | Cores | Pattern | M rows |
|---------|-------|---------|--------|
| Single (ternary/design.xclbin) | 1 | aie.flow + writebd | 32 |
| Single objfifo (ternary_objfifo) | 1 | object_fifo | 32 |
| 8-core (ternary_pyapi) | 8 | column-parallel | 32 |
| **32-core (ternary_32core)** | **32** | **4×8 row-broadcast + slice** | **128** |
| BitNet micro | 1 | scheduler microbench | 32 |
| BitNet scheduler | 1 | 7-phase full layer | 32 |

## 🎮 GPU Kernels (HIP / RDNA 3.5)

### Source Kernels
| Kernel | Packing | Compute | Info |
|--------|---------|---------|------|
| ternary_gemm_v2.hip | — | — | Wave32 |
| ternary_gemv.hip | 2-bit (4 ternaries/byte, 2 bpw) | fused add/sub/skip (no fp mul) | Wave32, block=128 |
| ternary_gemv_phase5.hip | — | — | Wave32, block=128 |
| ternary_gemv_phase5_16row.hip | — | — | block=128 |
| ternary_gemv_phase5_8row.hip | — | — | block=128 |
| ternary_gemv_phase5_dot4.hip | — | — | block=128 |
| ternary_gemv_phase5_halo.hip | — | — | block=128 |
| ternary_gemv_phase5_i4a.hip | — | — | block=128 |
| ternary_gemv_phase5_multi.hip | — | — | Wave32, block=128 |
| ternary_gemv_sherry.hip | — | — | Wave32, block=128 |
| ternary_gemv_tq1_halo.hip | base-3 (5 ternaries/byte, 1.6 bpw) | — | DP4A (int8 dot product), block=128 |
| ternary_gemv_v2.hip | — | — | DP4A (int8 dot product), block=128 |
| zaya_moe_ternary_gemv.hip | base-3 (5 ternaries/byte, 1.6 bpw) | — | block=128 |

### Compiled Objects (1bit/build/)
| Object | Size |
|--------|------|
| ternary_gemv_phase5_dot4.hip.o | 20 KB |
| ternary_gemv_phase5_halo.hip.o | 26 KB |
| ternary_gemv_sherry.hip.o | 25 KB |
| ternary_gemv_tq1_halo.hip.o | 24 KB |
| zaya_moe_ternary_gemv.hip.o | 29 KB |
| bonsai_gemv_scalar_ref.hip.o | 20 KB |
| bonsai_q1_gemv.hip.o | 17 KB |
| bonsai_tq2_gemv.hip.o | 17 KB |
| ternary_gemm_smallm.hip.o | 49 KB |
| ternary_gemm_smallm_scalar_ref.hip.o | 18 KB |
| ternary_gemv_launchers.hip.o | 5 KB |
| zaya_moe_launcher.hip.o | 2 KB |
| test_bonsai_e2e.cpp.o | 19 KB |
| test_bonsai_gemv.cpp.o | 27 KB |
| test_ternary_gemm_smallm.cpp.o | 26 KB |
| test_zaya_moe_gemv.cpp.o | 13 KB |

## 🎯 Spec-Decode Draft Models

| Model | Size |
|-------|------|
| final_model.pt | 1736 MB |
| best_model.pt | 1736 MB |
| eagle3_draft_npu_1k.pt | 1346 MB |
| eagle3_draft_npu.pt | 1346 MB |

## 🔬 Bit-Exact Verification

| Check | Result |
|-------|--------|
| Q2_0 decoder cos_vs_F16 | **1.000000** (4 tensors, Bonsai-1.7B) |
| Q2_0 → Q4NX converter | ✅ Round-trip lossless (INT8 passthrough) |
| Native ternary all-ones test | ✅ **256.0000 exactly** (bit-exact) |
| Vulkan ternary GEMM (279 tok/s) | ✅ Validated |
| mm_ternary BF16 precision | ✅ max error < 1e-3 vs CPU reference |

## 📐 Ternary Packing Format Reference

| Format | Bits/weight | Encoding | Use |
|--------|------------|----------|-----|
| Q1_0 (GGUF type 41) | 1.0 | 1 bit: +d/-d per value | Bonsai-1.7B-Q1_0 |
| Q2_0 (GGUF type 42) | 2.0 | 2-bit: {-1,0,+1,+2} × scale | PrismML ternary |
| NPU packed uint8 | 2.0 | 4×2bit/byte: 00=-1,01=0,10=+1,11=-1 | mm_ternary kernel |
| TQ1 halo v4 | 1.6 | base-3: 5 values/byte (3⁵=243) | ternary_gemv_tq1_halo |
| Sherry v3 | 1.25 | 3:4 sparsity (training-time only) | ternary_gemv_sherry |
| ZAYA MoE | 2.0 | 2-bit DP4A fused | zaya_moe_ternary_gemv |

## 📊 Summary

| Category | Count | Detail |
|----------|-------|--------|
| 1-bit GGUF models | 1 | Bonsai-1.7B-Q1_0 (248 MB) |
| Standard GGUF models | 4 | ZAYA1-8B variants |
| Native ternary NPU xclbins | 6 | 15-310 KB, 1-32 cores |
| HIP ternary kernel sources | 13 | 12 variants, 5 packing formats |
| HIP compiled objects | 16 | 14 .o files, RDNA 3.5 |
| Vulkan ternary shaders | 2 | ternary_gemm.comp + .spv |
| Spec-decode draft models | 4 | eagle3 + dspark |

_Scan completed in 0.0s_
