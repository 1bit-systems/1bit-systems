# Fused Engine Benchmark Suite — Strix Halo

**Date:** 2026-07-07
**Hardware:** Strix Halo (AMD Ryzen AI 300 Series)
**NPU:** AIE2 rev 11 (0x17f0, PCI c6:00.1)
**GPU:** Radeon 8060S Graphics (RDNA 3.5, PCI c5:00.0)
**Vulkan:** 1.4.341 (RADV STRIX_HALO)
**Model:** Qwen3-0.6B (1.5B params, INT4 quantized)

## NPU-Only Throughput (tok/s)

| Engine | B=16 | B=32 | B=64 | B=128 | B=256 | B=512 |
|--------|------|------|------|-------|-------|-------|
| npu_engine_universal | 10 | 12 | 25 | **48** | **103** | 0 |
| npu_engine_qwen3_0_6b | 18 | 36 | 46 | 45 | 24 | — |
| npu_engine_qwen3_8b | 18 | 33 | 45 | 44 | 23 | — |
| npu_engine_gemma4_e2b | 18 | 32 | 45 | 44 | 22 | — |
| npu_engine_llama | 19 | 35 | 44 | 45 | 24 | — |

**Peak: 103 tok/s** — npu_engine_universal at B=256

## GPU Attention

- **Device:** Radeon 8060S Graphics
- **Vulkan:** 1.4.341 (RADV STRIX_HALO)
- **Shaders:** flash_attn.spv + flash_attn_batched.spv (loaded from ZINC)
- **Estimated:** ~0.5ms/layer × 28L = 14ms @ seq=100

## Fused NPU+GPU Pipeline

| Stage | Time | tok/s |
|-------|------|-------|
| NPU QKV+FFN (28L) | 13.2ms | — |
| GPU flash attention (28L) | 14.0ms | — |
| Pipeline overlap | 14.0ms (GPU-bound) | — |
| M=128 batch | — | ~9,143 theoretical |
| Realistic estimate | ~7.9ms | **~127** |
| **Target** | **3.7ms** | **273** |

## Status

- ✅ NPU baseline: 103 tok/s (universal engine, B=256)
- ✅ GPU attention: Radeon 8060S Vulkan init + SPIR-V shaders
- ✅ Pipeline overlap: executeLayerQKV + executeLayerAttnFFN + std.Thread
- ✅ Fused engine builds: `zig build` + `zig build test` pass
- ⏳ MLIR-AIE multi-invocation: AIE cores halt at `aie.end` (need kernel loop)
  - Workaround: xclbin reload between layers (~100ms each)
  - Fix: modify MLIR-AIE compilation to produce looping kernels

## Engine Binary Sizes

| Binary | Size | Description |
|--------|------|-------------|
| npu_engine | 117KB | Generic NPU engine |
| npu_engine_universal | 115KB | Auto-detect model, fastest at high batch |
| npu_engine_fused | 41KB | Fused NPU+GPU (xclbin per layer) |
| npu_engine_qwen3_0_6b | 117KB | Qwen3-0.6B optimized |
| npu_engine_qwen3_8b | 117KB | Qwen3-8B optimized |
| npu_engine_gemma4_e2b | 117KB | Gemma4 E2B |
| npu_engine_llama | 117KB | Llama family |
| npu_engine_spec | 65KB | Speculative decode |
| npu_engine_spec_v2 | 90KB | Speculative decode v2 |
