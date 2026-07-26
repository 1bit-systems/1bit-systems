> **📜 Hackathon submission** — This document was created for the AMD Radeon Hackathon 2026-07 and reflects the project state at that time (July 2026). Numbers like "97 tok/s" NPU and FastFlowLM references are historical — see the [README](../README.md) and [current benchmarks](../docs/wiki/performance.md) for up-to-date data.
>
# AMD AI DevMaster Hackathon — Track 3 Submission
## AI Acceleration & Performance

**Team**: 1bit.systems  
**Project**: 1bit.systems — Open-Source AMD XDNA 2 NPU Stack + ROCm GPU Kernels  
**Date**: July 2026  
**Hardware**: AMD Ryzen AI Max+ 395 (Strix Halo) — Radeon 8060S GPU (gfx1151) + 32 XDNA 2 NPU tiles + 128 GB unified LPDDR5X

---

## 1. Application Scenarios

1bit.systems delivers **maximum inference performance on consumer AMD hardware** by reverse-engineering the XDNA 2 NPU stack from binary and building custom ROCm HIP kernels — all open-source, zero proprietary code.

| Scenario | Description |
|----------|-------------|
| **NPU-Native Inference** | Full LLM inference on XDNA 2 NPU (32 tiles, 50 TOPS INT8) — no FastFlowLM dependency. Reverse-engineered 22 proprietary `.so` libraries and 209 xclbin bitstreams. Entire stack rebuilt from source. |
| **GPU-Accelerated Mamba1** | First open-source Mamba1 GPU backend for AMD. Alternating SSM + MoE layers on Radeon 8060S. BlackMamba 1.5B at 79.8 tok/s. |
| **Fused Ternary Decode** | TQ2 (2-bit ternary) fused QKV+GU kernel at 415 tok/s on ROCm HIP. Vulkan ZINC ternary backend at 318 tok/s. |
| **Mixed Backend Routing** | Token Router dispatches each layer to the optimal backend — NPU for dense matmul, GPU for ternary decode, CPU for attention at low token counts. |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    1bit.systems Acceleration Stack           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Token Router (per-layer dispatch)        │   │
│  │  auto-detect model → route layers to fastest backend  │   │
│  └────┬────────────┬──────────────┬────────────┬─────────┘   │
│       │            │              │            │             │
│  ┌────▼────┐ ┌────▼────┐  ┌──────▼─────┐ ┌───▼────────┐   │
│  │ XDNA 2  │ │ ROCm    │  │ Vulkan     │ │ CPU        │   │
│  │ NPU     │ │ HIP     │  │ ZINC       │ │ OpenMP     │   │
│  │ 97 t/s  │ │ 417 t/s │  │ 318 t/s    │ │  5 t/s     │   │
│  │ 32tiles │ │ gfx1151 │  │ RDNA 3.5   │ │ fallback   │   │
│  └─────────┘ └─────────┘  └────────────┘ └────────────┘   │
│       │            │              │                         │
│  ┌────▼────────────▼──────────────▼──────────────────┐     │
│  │              128 GB unified LPDDR5X                │     │
│  │        Zero-copy DMA between all backends          │     │
│  └───────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 3. Key Innovations

### 3.1 XDNA 2 NPU Reverse Engineering (4 Days, Zero Docs)

This is the project's flagship achievement — full reverse-engineering of AMD's **proprietary** XDNA 2 NPU stack:

- **22 proprietary `.so` libraries** disassembled and analyzed
- **209 xclbin bitstreams** traced back to their AIE generator sources
- **Q4NX format** fully decoded: 311 tensors, 4-bit groups of 32 with bf16 scales, 32×256 NPU tile layout
- **NPU instruction set** documented: 8 DMA channels, 8 BD slots, compute vs. data tiles
- **Entire stack rebuilt from source**: 87.8 MB closed binary → 17.5 MB open-source

**Perf**: NPU v12 achieves **97 tok/s** (M=32 batch decode, INT8 GEMM on 32 XDNA 2 tiles) — beats FLM Kraken Point by 46%.

### 3.2 ROCm HIP GPU Kernels

| Kernel | Speed | Notes |
|--------|-------|-------|
| Q1 GEMV (fused) | **417 tok/s** | Binary 1-bit fused kernel |
| TQ2 Fused (QKV+GU) | **415 tok/s** | 2-bit ternary, all projections in one dispatch |
| BlackMamba 1.5B e2e | **79.8 tok/s** | Alternating SSM + MoE on Radeon 8060S |
| BlackMamba 2.8B e2e | **46.4 tok/s** | 36 layers (18 SSM + 18 MoE) |
| Prefill INT8 WMMA | **42.2 TFLOPS** | Wave32 Matrix Multiply-Accumulate |

### 3.3 Mamba1 GPU Backend (Novel)

First open-source Mamba1/SSM GPU backend for AMD hardware:

- Alternating SSM layers (rmsnorm → in_proj → conv1d/SiLU → selective_scan → gate → out_proj) and MoE FFN layers (router → top-1 expert dispatch → SiLU → scale-add residual)
- Pure HIP kernels — no external library dependency
- Zero-copy state passing between SSM blocks
- Supports BlackMamba 1.5B (30 layers) and 2.8B (36 layers)

### 3.5 TheRock 7.15.0a Integration

First project to adopt and validate **TheRock** — AMD's nightly pip-installable HIP SDK for gfx1151:

- Drop-in replacement for system ROCm 7.2.4
- Native gfx1151 code generation (no HSA override needed)
- pip install — no apt repo required
- Full build validation at 79.8 tok/s BlackMamba inference

---

## 4. Performance Benchmarks

### 4.1 Kernel-Level Microbenchmarks (synthetic 28-layer weight buffer)

| Benchmark | Value | Backend | Engine |
|-----------|:-----:|---------|--------|
| Q1 GEMV | **417 tok/s** | ROCm HIP | Fused kernel |
| Fused TQ2 | **415 tok/s** | ROCm HIP | QKV+GU fused |
| GPU ternary | **318 tok/s** | Vulkan | ZINC SPIR-V |
| NPU v12 | **97 tok/s** | XDNA 2 | 32 tiles |
| Prefill | **42.21 TFLOPS** | INT8 WMMA | 32 tiles |

### 4.2 End-to-End Model Inference

| Model | Tool/s | Backend | Architecture |
|-------|:------:|---------|-------------|
| BlackMamba 1.5B | **79.8 tok/s** | ROCm HIP | Mamba1 SSM + MoE |
| BlackMamba 2.8B | **46.4 tok/s** | ROCm HIP | Mamba1 SSM + MoE |
| Zaya1-8B | **64 tok/s** | ROCm HIP | Dense Transformer |
| Qwen3 0.6B | **64 tok/s** | ROCm HIP | Dense Transformer |
| Qwen3 27B Q4_K | **30 tok/s** | ROCm HIP | Speculative decode |
| Qwen3 35B MoE Q4_K | **20 tok/s** | ROCm HIP | Speculative decode |

### 4.3 NPU v12 vs. FLM Kraken Point

| Metric | 1bit NPU v12 | FLM Kraken Point | Advantage |
|--------|:------------:|:----------------:|:---------:|
| Throughput | 97 tok/s | 66 tok/s | **46% faster** |
| M=32 latency | 244 → 10 ms/tok | — | **24× speedup** |
| Stack size | 17.5 MB open | 87.8 MB closed | **5× smaller** |
| License | MIT | Proprietary | **Open source** |

---

## 5. Open-Source NPU Stack Comparison

| Component | FastFlowLM (AMD) | This work |
|-----------|-----------------|-----------|
| Size | 87.8 MB | 17.5 MB |
| License | Proprietary | MIT |
| Source | Closed binary | Open C++23 |
| Dependencies | 22 `.so` files | XRT only |
| Model format | Q4NX only | Q4NX, Q4NX, 1BP, GGUF |
| Architecture | Black-box | Documented |
| NPU instruction set | Unknown | Fully documented |
| xclbin generation | Proprietary generator | AIE MLIR + open tools |

---

## 6. Project Links

- **Source**: https://github.com/bong-water-water-bong/1bit-systems
- **NPU engine**: `engine/npu/src/npu_engine_v12.cpp`
- **Mamba1 GPU backend**: `src/mamba1_engine.hip`
- **NPU ISA docs**: `fastflowlm_analysis/NPU_ISA.md` (274 lines)
- **Q4NX format spec**: `fastflowlm_analysis/Q4NX_FORMAT.md`
- **FLM reverse-engineering audit**: `docs/journey.md` (1800+ lines)
- **Performance SSOT**: `docs/wiki/performance.md`
- **Site**: https://1bit.systems

---

## 7. Setup

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems
# Install TheRock 7.15.0a — native gfx1151 HIP SDK
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ rocm[devel,libraries]
# Build
cmake -B build -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build -j$(nproc)
# Run Mamba1 GPU inference
./build/test_mamba1_backend models/blackmamba-1.5b.gguf 64
# Run NPU inference
./build/zaya_server --port 8088
```

*Generated for AMD AI DevMaster Hackathon — Track 3 Submission*
