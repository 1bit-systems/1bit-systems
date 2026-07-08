# v0.2.0 — Native Ternary NPU: Hardware Verified on Strix Halo

**2026-07-08** | **RyzenAI-npu5** | **32-core bit-exact BF16**

## 🎉 Breakthrough

Native ternary (2-bit packed) kernel runs on AMD NPU hardware with
bit-exact BF16 precision. 32-core xclbin processes 128 rows × 256
ternary values in 118.9 µs.

| Configuration | Latency | Throughput | GMACs/s | XCLBin | All-ones |
|--------------|---------|------------|---------|--------|----------|
| 1 core | 68.3 µs | 14,636/s | 0.120 | 16 KB | 32/32 ✅ |
| **32 cores** | **118.9 µs** | **8,410/s** | **0.276** | **314 KB** | **128/128 ✅** |

## ✅ Verified

- Single-core: **-256.0000 exactly** (32/32 outputs)
- 32-core: **-256.0000 exactly** (128/128 outputs)
- Q2_0 decoder: **cos=1.000000** vs F16 (4 tensors)

## 📦 What's Included

### Kernel & MLIR
- `mm_ternary_32x64x128` Chess C++ kernel with `row_start`/`num_rows` tiling
- 32-core MLIR generator (4×8 grid, per-column DMA, object_fifo)
- Single-core MLIR generator (debug/verify)

### Tools
- Q2_0 bit-exact GGUF decoder (type 42 ternary format)
- Q2_0→Q4NX INT8 passthrough converter
- NPU unlock tool (UEFI NVRAM patching for Strix Halo)
- 1-bit model benchmark suite

### Engine
- `npu_ternary_target.h` — TargetModelInterface for spec-decode
- 32-core test harness

## 🏎️ Models Benchmarked

| Model | Size | Format | Hidden | Layers | 1-bit? |
|-------|------|--------|--------|--------|--------|
| **Bonsai-1.7B-Q1_0** | 250 MB | Q1_0 | 2048 | 28 | 🔵 YES |
| ZAYA1-8B-zaya | 5.6 GB | Q4_K/F32 | 2048 | 40 | — |
| ZAYA1-8B-Q4_K_M | 5.6 GB | Q4_K | 2048 | 40 | — |

## 🧠 GPU Kernels

13 HIP kernel sources (5 packing formats: Q1_0, Q2_0, TQ1, Sherry, ZAYA MoE).
16 compiled objects for RDNA 3.5 (gfx1151). Vulkan ternary GEMM shader.

---

**Strix Halo XDNA2. AMD Ryzen AI Max+ 395.**
